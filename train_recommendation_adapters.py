import argparse
import gc
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

PRICE_ORDER = {"low": 0, "medium": 1, "high": 2}

SYSTEM_PROMPT_TEMPLATE = """
You are a tourism recommendation assistant for Pula, Croatia.
Your task is to return ONLY valid JSON (no markdown, no prose outside JSON).

You will receive one user profile and a candidate place list.
Recommend top {top_k} places from candidates only.

Output schema (strict):
{{
  "recommendations": [
    {{
      "name": "<exact place name from candidates>",
      "type": "<place type>",
      "score": <integer 0-100>,
      "short_answer": "<very short answer>",
      "reason": "<short grounded reason>",
      "matched_tags": ["<tag1>", "<tag2>"]
    }}
  ]
}}

Rules:
1. Return exactly {top_k} recommendations whenever enough candidates exist.
2. Use exact place names from candidates.
3. Ground reasons in the provided profile preferences and candidate metadata.
4. Keep short_answer concise (max 8 words) and reason concise (1 sentence).
5. Do not output any text before or after JSON.
6. Do not use <think> tags.
""".strip()


@dataclass
class Place:
    name: str
    place_type: str
    tags: List[str]
    rating: Optional[float]
    price: Optional[str]
    description: str
    opening_hours: Optional[str]
    street: Optional[str]
    housenumber: Optional[str]


def normalize_token(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_tags(raw_tags: Any) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, list):
        tags = raw_tags
    elif isinstance(raw_tags, str):
        tags = re.split(r"[;,]", raw_tags)
    else:
        return []

    cleaned = []
    for tag in tags:
        token = normalize_token(tag)
        if token:
            cleaned.append(token)
    return sorted(set(cleaned))


def load_places(file_path: str) -> List[Place]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    places: List[Place] = []
    for item in data:
        props = item.get("properties", {})
        name = (props.get("name") or "").strip()
        if not name:
            continue

        rating = props.get("rating")
        if rating is not None:
            try:
                rating = float(rating)
            except (TypeError, ValueError):
                rating = None

        price = props.get("price")
        if isinstance(price, str):
            price = normalize_token(price)
            if price not in PRICE_ORDER:
                price = None
        else:
            price = None

        place_type = normalize_token(item.get("type", "unknown")) or "unknown"
        tags = normalize_tags(props.get("tags"))
        tags.append(place_type)
        if props.get("outdoor_seating") is True:
            tags.append("outdoor_seating")
        if props.get("indoor_seating") is True:
            tags.append("indoor_seating")

        places.append(
            Place(
                name=name,
                place_type=place_type,
                tags=sorted(set(tags)),
                rating=rating,
                price=price,
                description=(props.get("description") or "").strip(),
                opening_hours=props.get("opening_hours"),
                street=props.get("addr:street"),
                housenumber=props.get("addr:housenumber"),
            )
        )
    return places


def load_profiles(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_and_score_candidates(
    profile: Dict[str, Any],
    places: List[Place],
    strict_price: bool,
) -> List[Tuple[Place, float, Dict[str, Any]]]:
    likes = {normalize_token(x) for x in profile.get("likes", []) if x}
    dislikes = {normalize_token(x) for x in profile.get("dislikes", []) if x}

    min_rating = profile.get("min_rating")
    try:
        min_rating = float(min_rating)
    except (TypeError, ValueError):
        min_rating = None

    pref_price = normalize_token(profile.get("price_preference", ""))
    if pref_price not in PRICE_ORDER:
        pref_price = ""

    scored: List[Tuple[Place, float, Dict[str, Any]]] = []

    for p in places:
        tag_set = set(p.tags)
        like_overlap = likes.intersection(tag_set)
        dislike_overlap = dislikes.intersection(tag_set)

        if dislike_overlap:
            continue
        if min_rating is not None and p.rating is not None and p.rating < min_rating:
            continue

        price_match = None
        if pref_price:
            if p.price is not None:
                price_match = p.price == pref_price
                if strict_price and not price_match:
                    continue

        score = 0.0
        score += 3.0 * len(like_overlap)
        if p.rating is not None:
            score += p.rating
            if min_rating is not None and p.rating >= min_rating:
                score += 1.0
        if price_match is True:
            score += 1.5
        elif price_match is False:
            score -= 0.5

        diagnostics = {
            "like_overlap": sorted(like_overlap),
            "price_match": price_match,
        }
        scored.append((p, score, diagnostics))

    if not scored:
        for p in places:
            scored.append((p, 0.0, {"like_overlap": [], "price_match": None, "fallback": True}))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def candidate_block(candidates: Sequence[Place]) -> str:
    rows = []
    for p in candidates:
        address_parts = [x for x in [p.street, p.housenumber] if x]
        address = " ".join(address_parts) if address_parts else "unknown"
        rows.append(
            {
                "name": p.name,
                "type": p.place_type,
                "rating": p.rating,
                "price": p.price,
                "tags": p.tags,
                "address": address,
                "opening_hours": p.opening_hours,
                "description": p.description[:240] if p.description else "",
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def build_user_prompt(profile: Dict[str, Any], candidates: Sequence[Place], top_k: int) -> str:
    compact_profile = {
        "id": profile.get("id"),
        "name": profile.get("name"),
        "travel_style": profile.get("travel_style"),
        "persona": profile.get("persona"),
        "likes": profile.get("likes", []),
        "dislikes": profile.get("dislikes", []),
        "price_preference": profile.get("price_preference"),
        "min_rating": profile.get("min_rating"),
        "query": profile.get("input", ""),
    }

    return (
        f"User profile:\n{json.dumps(compact_profile, ensure_ascii=False)}\n\n"
        f"Candidates:\n{candidate_block(candidates)}\n\n"
        f"Return top {top_k} recommendations as strict JSON."
    )


def build_target_payload(
    scored_candidates: Sequence[Tuple[Place, float, Dict[str, Any]]],
    top_k: int,
) -> Dict[str, Any]:
    recommendations = []
    for rank, (place, score, diagnostics) in enumerate(scored_candidates[:top_k], start=1):
        overlap = diagnostics.get("like_overlap", [])
        overlap_text = ", ".join(overlap[:2]) if overlap else "your profile"
        short_answer = f"Great match for {overlap[0]}" if overlap else "Good fit for your preferences"
        normalized_score = max(0, min(100, int(round(60 + score * 4 - rank))))
        recommendations.append(
            {
                "name": place.name,
                "type": place.place_type,
                "score": normalized_score,
                "short_answer": short_answer,
                "reason": f"Matches {overlap_text} and overall preferences.",
                "matched_tags": overlap[:4],
            }
        )

    return {"recommendations": recommendations}


def build_training_examples(
    profiles: Sequence[Dict[str, Any]],
    places: Sequence[Place],
    top_k: int,
    candidate_limit: int,
    strict_price: bool,
    max_profiles: Optional[int],
) -> List[Dict[str, str]]:
    if max_profiles is not None:
        profiles = profiles[:max_profiles]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(top_k=top_k)
    examples = []

    for profile in profiles:
        scored = filter_and_score_candidates(profile, list(places), strict_price=strict_price)
        candidate_places = [x[0] for x in scored[:candidate_limit]]
        target_payload = build_target_payload(scored, top_k=top_k)

        user_prompt = build_user_prompt(profile, candidate_places, top_k)
        examples.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant_json": json.dumps(target_payload, ensure_ascii=False),
            }
        )

    return examples


def _to_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    return []


def _extract_profile_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    profile = row.get("profile") if isinstance(row.get("profile"), dict) else {}
    merged = dict(profile)

    # Promote top-level fields often found in eval/training exports.
    for key in [
        "id",
        "user_id",
        "name",
        "travel_style",
        "persona",
        "likes",
        "dislikes",
        "price_preference",
        "min_rating",
        "input",
        "query",
    ]:
        if key in row and key not in merged:
            merged[key] = row.get(key)

    if "id" not in merged and "user_id" in merged:
        merged["id"] = merged.get("user_id")
    if "input" not in merged and "query" in merged:
        merged["input"] = merged.get("query")

    merged.setdefault("likes", _to_list(merged.get("likes")))
    merged.setdefault("dislikes", _to_list(merged.get("dislikes")))
    merged.setdefault("input", "")
    return merged


def _extract_recommendations_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(row.get("recommendations"), list):
        return row["recommendations"]

    parsed = row.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("recommendations"), list):
        return parsed["recommendations"]

    validated = row.get("validated_recommendations")
    if isinstance(validated, list):
        return validated

    return []


def _normalize_teaching_recommendations(
    recommendations: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in recommendations[:top_k]:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or item.get("place") or item.get("venue_name") or "").strip()
        if not name:
            continue

        place_type = normalize_token(item.get("type") or item.get("venue_type") or "")
        score = item.get("score", 80)
        try:
            score = int(round(float(score)))
        except (TypeError, ValueError):
            score = 80
        score = max(0, min(100, score))

        matched_tags = item.get("matched_tags")
        if not isinstance(matched_tags, list):
            matched_tags = []

        short_answer = str(item.get("short_answer") or item.get("short") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not short_answer:
            short_answer = reason if reason else "Recommended for your preferences"
        if not reason:
            reason = "Matches your stated travel preferences."

        normalized.append(
            {
                "name": name,
                "type": place_type,
                "score": score,
                "short_answer": short_answer,
                "reason": reason,
                "matched_tags": [normalize_token(x) for x in matched_tags if x],
            }
        )

    return normalized


def load_teaching_examples(
    teaching_file: str,
    places: Sequence[Place],
    top_k: int,
    candidate_limit: int,
    strict_price: bool,
    max_profiles: Optional[int],
) -> List[Dict[str, str]]:
    with open(teaching_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Teaching dataset must be a JSON array of records")

    if max_profiles is not None:
        data = data[:max_profiles]

    examples: List[Dict[str, str]] = []
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(top_k=top_k)

    for row in data:
        if not isinstance(row, dict):
            continue

        profile = _extract_profile_from_row(row)
        if not profile:
            continue

        scored = filter_and_score_candidates(profile, list(places), strict_price=strict_price)
        candidate_places = [x[0] for x in scored[:candidate_limit]]
        user_prompt = build_user_prompt(profile, candidate_places, top_k)

        recommendations = _extract_recommendations_from_row(row)
        normalized_recs = _normalize_teaching_recommendations(recommendations, top_k=top_k)
        if not normalized_recs:
            # Skip weak or incomplete rows to keep labels high quality.
            continue

        assistant_payload = {"recommendations": normalized_recs}
        examples.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant_json": json.dumps(assistant_payload, ensure_ascii=False),
            }
        )

    return examples


def extract_model_short_name(model_id: str) -> str:
    name = model_id.split("/")[-1].lower()
    if "llama-3.2" in name:
        return "llama3.2"
    if "llama-3.1" in name:
        return "llama3.1"
    if "llama-3" in name:
        return "llama3"
    if "qwen" in name:
        return name.replace("qwen", "qwen").replace("-instruct", "")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def resolve_lora_target_modules(model) -> List[str]:
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    available = set()
    for module_name, _ in model.named_modules():
        suffix = module_name.split(".")[-1]
        if suffix in preferred:
            available.add(suffix)

    if available:
        return [m for m in preferred if m in available]

    # Fallback for uncommon naming.
    return ["q_proj", "v_proj"]


def maybe_hf_login() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        return
    try:
        login(token=token, add_to_git_credential=False)
    except Exception as exc:
        print(f"Warning: Hugging Face login failed: {exc}")


def make_chat_text(tokenizer, system_prompt: str, user_prompt: str, assistant_json: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_json},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return (
            f"{system_prompt}\n\n"
            f"User:\n{user_prompt}\n\n"
            f"Assistant:\n{assistant_json}"
        )


def train_single_model(args: argparse.Namespace, model_id: str, base_examples: Sequence[Dict[str, str]]) -> None:
    model_short = extract_model_short_name(model_id)
    adapter_type = "lora"
    output_dir = os.path.join(args.output_root, f"{adapter_type}-{model_short}-checkpoints")
    adapter_dir = os.path.join(args.adapters_dir, f"{adapter_type}-{model_short}")

    print("=" * 70)
    print(f"Training model: {model_id}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting training text with model chat template...")
    text_rows = [
        {"text": make_chat_text(tokenizer, ex["system_prompt"], ex["user_prompt"], ex["assistant_json"])}
        for ex in base_examples
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(text_rows))
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map={"": local_rank},
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_lora_target_modules(model),
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=args.max_length,
        packing=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=not args.bf16,
        bf16=args.bf16,
        tf32=args.tf32,
        optim=args.optim,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        group_by_length=True,
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        ddp_find_unused_parameters=True,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    is_main_process = trainer.is_world_process_zero()
    if is_main_process:
        print(f"Train/Eval samples: {len(train_dataset)}/{len(eval_dataset)}")
        trainer.model.print_trainable_parameters()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    os.makedirs(args.adapters_dir, exist_ok=True)
    if is_main_process:
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        eval_losses = [x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x]
        train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        metrics = {
            "model_id": model_id,
            "adapter_dir": adapter_dir,
            "training_time_seconds": round(elapsed, 2),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "final_eval_loss": round(eval_losses[-1], 4) if eval_losses else None,
            "best_eval_loss": round(min(eval_losses), 4) if eval_losses else None,
            "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "max_length": args.max_length,
        }
        metrics_path = os.path.join(adapter_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved adapter: {adapter_dir}")
        print(f"Saved metrics: {metrics_path}")

    del trainer, model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapters for tourism recommendations")

    parser.add_argument("--model-ids", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--places-file", type=str, default="pula_merged.json")
    parser.add_argument("--profiles-file", type=str, default="user_profiles.json")
    parser.add_argument(
        "--teaching-dataset-file",
        type=str,
        default="",
        help="Optional JSON file with gold recommendations per profile. If set, this is used instead of heuristic labels.",
    )
    parser.add_argument("--output-root", type=str, default="./checkpoints")
    parser.add_argument("--adapters-dir", type=str, default="./adapters")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--strict-price", action="store_true")
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="eager")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    maybe_hf_login()

    model_ids = [x.strip() for x in args.model_ids.split(",") if x.strip()]
    if not model_ids:
        raise ValueError("--model-ids must contain at least one model")

    places = load_places(args.places_file)
    profiles = load_profiles(args.profiles_file)
    print(f"Loaded places/profiles: {len(places)}/{len(profiles)}")

    if args.teaching_dataset_file:
        print(f"Using teaching dataset: {args.teaching_dataset_file}")
        base_examples = load_teaching_examples(
            teaching_file=args.teaching_dataset_file,
            places=places,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            strict_price=args.strict_price,
            max_profiles=args.max_profiles,
        )
    else:
        print("No teaching dataset provided, using heuristic pseudo-labels from profiles + places.")
        base_examples = build_training_examples(
            profiles=profiles,
            places=places,
            top_k=args.top_k,
            candidate_limit=args.candidate_limit,
            strict_price=args.strict_price,
            max_profiles=args.max_profiles,
        )
    if not base_examples:
        raise ValueError("No training examples generated")

    print(f"Prepared recommendation examples: {len(base_examples)}")
    for model_id in model_ids:
        train_single_model(args, model_id, base_examples)

    print("All requested models finished.")


if __name__ == "__main__":
    main()
