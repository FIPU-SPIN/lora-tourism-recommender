import argparse
import csv
import gc
import json
import math
import os
import re
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


PRICE_ORDER = {"low": 0, "medium": 1, "high": 2}


class Place:
    def __init__(
        self,
        name: str,
        place_type: str,
        tags: List[str],
        rating: Optional[float],
        price: Optional[str],
        description: str,
        opening_hours: Optional[str],
        street: Optional[str],
        housenumber: Optional[str],
        outdoor_seating: Optional[bool],
        indoor_seating: Optional[bool],
    ):
        self.name = name
        self.place_type = place_type
        self.tags = tags
        self.rating = rating
        self.price = price
        self.description = description
        self.opening_hours = opening_hours
        self.street = street
        self.housenumber = housenumber
        self.outdoor_seating = outdoor_seating
        self.indoor_seating = indoor_seating


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
      "short_answer": "<personalized 1-2 sentence reply the user sees, e.g. Based on your love of X and Y style, <name> is a great match because ...>",
      "reason": "<internal 1 sentence justification citing specific matched tags, rating, or price that drove the score>",
      "matched_tags": ["<tag1>", "<tag2>"]
    }}
  ]
}}

Rules:
1. Return exactly {top_k} recommendations whenever enough candidates exist.
2. Use exact place names from candidates.
3. Ground short_answer in the user's persona, travel_style, and top likes. It is the message the user reads.
4. Ground reason in the candidate's specific tags, rating, or price that matched the profile.
5. Do not output any text before or after JSON.
6. Do not use <think> tags.
""".strip()


def normalize_token(text: str) -> str:
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

        description = (props.get("description") or "").strip()

        places.append(
            Place(
                name=name,
                place_type=place_type,
                tags=sorted(set(tags)),
                rating=rating,
                price=price,
                description=description,
                opening_hours=props.get("opening_hours"),
                street=props.get("addr:street"),
                housenumber=props.get("addr:housenumber"),
                outdoor_seating=props.get("outdoor_seating"),
                indoor_seating=props.get("indoor_seating"),
            )
        )
    return places


def load_profiles(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_gold_labels(file_path: Optional[str]) -> Dict[int, set]:
    if not file_path:
        return {}
    if not os.path.exists(file_path):
        print(f"Warning: labels file not found at {file_path}. Gold metrics disabled.")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: Dict[int, set] = {}
    if isinstance(data, dict):
        for user_id, relevant in data.items():
            try:
                uid = int(user_id)
            except (TypeError, ValueError):
                continue
            if isinstance(relevant, list):
                labels[uid] = {normalize_token(x) for x in relevant if x}
    elif isinstance(data, list):
        for row in data:
            uid = row.get("user_id", row.get("id"))
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                continue

            relevant = (
                row.get("relevant_places")
                or row.get("relevant_place_names")
                or row.get("ground_truth")
                or row.get("labels")
                or []
            )
            if isinstance(relevant, list):
                labels[uid] = {normalize_token(x) for x in relevant if x}

    return labels


def build_place_lookup(places: List[Place]) -> Dict[str, Place]:
    return {normalize_token(p.name): p for p in places}


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
            if p.price is None:
                price_match = None
            else:
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
            "like_overlap_count": len(like_overlap),
            "like_overlap_tags": sorted(like_overlap),
            "price_match": price_match,
            "rating": p.rating,
            "price": p.price,
        }
        scored.append((p, score, diagnostics))

    if not scored:
        for p in places:
            scored.append((p, 0.0, {"fallback": True, "rating": p.rating, "price": p.price}))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def candidate_block(candidates: List[Place]) -> str:
    rows = []
    for p in candidates:
        payload = {
            "name": p.name,
            "type": p.place_type,
            "rating": p.rating,
            "price": p.price,
            "tags": p.tags,
            "description": p.description[:120] if p.description else "",
        }
        rows.append(payload)
    return json.dumps(rows, ensure_ascii=False)


def build_user_prompt(profile: Dict[str, Any], candidates: List[Place], top_k: int) -> str:
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


def build_inputs(system_prompt: str, user_input: str, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        print(f"Warning: No chat template found, using fallback. Error: {e}")
        prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt")

    return inputs


def resolve_max_new_tokens(
    context_length: int,
    input_length: int,
    requested_limit: Optional[int],
) -> int:
    available = context_length - input_length
    if available <= 0:
        raise ValueError(
            f"Prompt length ({input_length} tokens) reaches or exceeds the "
            f"model context length ({context_length} tokens)."
        )
    if requested_limit is not None:
        return min(available, requested_limit)
    return available


def model_context_length(model) -> int:
    context_length = getattr(model.config, "max_position_embeddings", None)
    if context_length is None:
        context_length = getattr(model.config, "n_positions", None)
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("Could not determine the model's maximum context length.")
    return context_length


def load_model_and_tokenizer(model_id: str, offload_folder: str, gpu_memory_gib: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_label = f"multi-GPU 4-bit path ({n_gpus} GPUs)"
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
    else:
        # Single A100-40GB path: 4-bit has a better chance to fit than 8-bit.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_label = "single-GPU 4-bit path"
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": {"": 0},
            "max_memory": {0: f"{gpu_memory_gib}GiB"},
            "trust_remote_code": True,
        }

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, RuntimeError) as e:
        message = str(e)
        raise RuntimeError(
            f"Model loading failed on {load_label}. "
            "CPU/disk offload fallback is disabled due environment hook incompatibility. "
            "Try increasing requested GPUs, reducing --gpu-memory-gib a bit (e.g. 34), or using a smaller model. "
            f"Original error: {message}"
        ) from e

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: Optional[int],
    do_sample: bool,
    temperature: float,
) -> str:
    inputs = build_inputs(system_prompt, user_prompt, tokenizer)
    input_len = inputs["input_ids"].shape[-1]
    max_new_tokens = resolve_max_new_tokens(
        model_context_length(model), input_len, max_new_tokens
    )
    inputs = inputs.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    import torch

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def batch_generate_transformers(
    model,
    tokenizer,
    system_prompt: str,
    user_prompts: List[str],
    max_new_tokens: Optional[int],
    do_sample: bool,
    temperature: float,
) -> List[str]:
    import torch

    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": up},
        ]
        for up in user_prompts
    ]

    try:
        prompts = [
            tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conv in conversations
        ]
    except Exception:
        prompts = [
            f"{system_prompt}\n\nUser: {up}\n\nAssistant:"
            for up in user_prompts
        ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    max_new_tokens = resolve_max_new_tokens(
        model_context_length(model), inputs["input_ids"].shape[1], max_new_tokens
    )
    input_lens = inputs["attention_mask"].sum(dim=1)
    inputs = inputs.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    decoded: List[str] = []
    for i in range(outputs.shape[0]):
        new_tokens = outputs[i][int(input_lens[i]):]
        decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return decoded


def load_vllm_model(
    model_id: str,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 16384,
):
    import os

    # Some schedulers expose CUDA_VISIBLE_DEVICES as GPU UUIDs (GPU-...)
    # while vLLM expects logical integer indices. Remap to 0..N-1 if needed.
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw_visible:
        gpu_ids = [x.strip() for x in raw_visible.split(",") if x.strip()]
        if gpu_ids and all(g.startswith("GPU-") for g in gpu_ids):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(len(gpu_ids))
            )
            print(
                "Detected UUID-based CUDA_VISIBLE_DEVICES; remapped to "
                f"logical indices: {os.environ['CUDA_VISIBLE_DEVICES']}"
            )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
    from vllm import LLM

    visible_devices = [
        d.strip()
        for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if d.strip()
    ]
    n_gpus = len(visible_devices) if visible_devices else 1
    print(f"Loading vLLM model '{model_id}' on {n_gpus} GPUs (tensor_parallel_size={n_gpus})")
    llm = LLM(
        model=model_id,
        tensor_parallel_size=n_gpus,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    return llm


def batch_generate_vllm(
    llm,
    system_prompt: str,
    user_prompts: List[str],
    max_new_tokens: Optional[int],
    do_sample: bool,
    temperature: float,
    max_model_len: int,
) -> List[str]:
    from vllm import SamplingParams

    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": up},
        ]
        for up in user_prompts
    ]

    tokenizer = llm.get_tokenizer()
    sampling_params = []
    for conversation in conversations:
        prompt_tokens = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
        )
        generation_limit = resolve_max_new_tokens(
            max_model_len, len(prompt_tokens), max_new_tokens
        )
        sampling_params.append(
            SamplingParams(
                max_tokens=generation_limit,
                temperature=temperature if do_sample else 0.0,
            )
        )

    # Disable Qwen3 chain-of-thought thinking if the chat template supports it.
    try:
        outputs = llm.chat(
            conversations,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        outputs = llm.chat(conversations, sampling_params)

    return [o.outputs[0].text.strip() for o in outputs]


def extract_json_payload(text: str) -> Tuple[Dict[str, Any], str]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<｜begin▁of▁sentence｜>.*?<｜Assistant｜>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    if not cleaned:
        return {}, "empty_response"

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, ""
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1))
            if isinstance(parsed, dict):
                return parsed, ""
        except json.JSONDecodeError:
            pass

    start = cleaned.find("{")
    if start == -1:
        return {}, "no_json_object_found"

    depth = 0
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:idx + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed, ""
                except json.JSONDecodeError as e:
                    return {}, f"json_decode_error:{e.msg}"

    return {}, "unbalanced_json_braces"


def normalize_recommendation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    name = item.get("name") or item.get("venue_name") or item.get("place") or ""
    item_type = item.get("type") or item.get("venue_type") or ""
    score = item.get("score", 0)
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    matched_tags = item.get("matched_tags") or []
    if not isinstance(matched_tags, list):
        matched_tags = []

    short_answer = str(item.get("short_answer") or item.get("short") or "").strip()
    reason = str(item.get("reason", "")).strip()

    return {
        "name": str(name).strip(),
        "type": normalize_token(item_type),
        "score": score,
        "short_answer": short_answer,
        "reason": reason,
        "matched_tags": [normalize_token(x) for x in matched_tags if x],
    }


def build_heuristic_payload(
    candidates: List[Place],
    top_k: int,
    reason: str,
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    picks = candidates[:top_k]
    likes = []
    travel_style = ""
    if profile:
        likes = [str(x) for x in profile.get("likes", []) if x]
        travel_style = str(profile.get("travel_style") or "").strip()

    recs = []
    for i, p in enumerate(picks):
        # Build a personalized short_answer from profile info + place name
        if likes and travel_style:
            short_answer = (
                f"Based on your {travel_style} travel style and love of "
                f"{', '.join(likes[:3])}, {p.name} is a great match for you."
            )
        elif likes:
            short_answer = (
                f"Given your interest in {', '.join(likes[:3])}, "
                f"{p.name} is a recommended choice."
            )
        elif travel_style:
            short_answer = (
                f"With your {travel_style} travel style, "
                f"{p.name} is well suited for you."
            )
        else:
            short_answer = f"{p.name} is a recommended place in Pula."

        # Build a place-specific reason
        place_parts = []
        if p.rating is not None:
            place_parts.append(f"rated {p.rating:.1f}")
        if p.price:
            place_parts.append(f"{p.price} price range")
        if p.tags:
            place_parts.append(f"tags: {', '.join(p.tags[:4])}")
        place_reason = (
            f"{p.name} was selected because it is {'; '.join(place_parts)}." if place_parts
            else f"{p.name} was selected as a top heuristic candidate."
        )

        recs.append({
            "name": p.name,
            "type": p.place_type,
            "score": max(50, 85 - i),
            "short_answer": short_answer,
            "reason": place_reason,
            "matched_tags": [],
        })
    return {"recommendations": recs}


def validate_recommendations(
    parsed: Dict[str, Any],
    place_lookup: Dict[str, Place],
    profile: Dict[str, Any],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    recs = parsed.get("recommendations")
    if not isinstance(recs, list):
        recs = []

    likes = {normalize_token(x) for x in profile.get("likes", []) if x}
    pref_price = normalize_token(profile.get("price_preference", ""))
    if pref_price not in PRICE_ORDER:
        pref_price = ""

    min_rating = profile.get("min_rating")
    try:
        min_rating = float(min_rating)
    except (TypeError, ValueError):
        min_rating = None

    validated: List[Dict[str, Any]] = []
    validity_count = 0
    tag_overlap_total = 0.0
    rating_ok = 0
    price_ok = 0

    for item in recs[:top_k]:
        norm = normalize_recommendation_item(item)
        key = normalize_token(norm["name"])
        place = place_lookup.get(key)

        exists = place is not None
        if exists:
            validity_count += 1
            place_tags = set(place.tags)
            overlap = likes.intersection(place_tags)
            tag_overlap_total += len(overlap)
            if min_rating is not None:
                if place.rating is not None and place.rating >= min_rating:
                    rating_ok += 1
            else:
                rating_ok += 1

            if pref_price:
                if place.price == pref_price:
                    price_ok += 1
            else:
                price_ok += 1

            if not norm["type"]:
                norm["type"] = place.place_type

        norm["exists_in_dataset"] = exists
        validated.append(norm)

    rec_count = max(1, len(validated))
    metrics = {
        "recommendation_count": len(validated),
        "recommendation_validity_rate": validity_count / rec_count,
        "avg_tag_overlap": tag_overlap_total / rec_count,
        "rating_compliance_rate": rating_ok / rec_count,
        "price_compliance_rate": price_ok / rec_count,
        "type_diversity": len({x.get("type", "") for x in validated if x.get("type")}) / rec_count,
    }
    return validated, metrics


def precision_at_k(pred: List[str], truth: set, k: int) -> float:
    if k == 0:
        return 0.0
    pred_k = pred[:k]
    hits = sum(1 for x in pred_k if x in truth)
    return hits / k


def recall_at_k(pred: List[str], truth: set, k: int) -> float:
    if not truth:
        return 0.0
    pred_k = pred[:k]
    hits = sum(1 for x in pred_k if x in truth)
    return hits / len(truth)


def ndcg_at_k(pred: List[str], truth: set, k: int) -> float:
    pred_k = pred[:k]
    dcg = 0.0
    for i, item in enumerate(pred_k, start=1):
        rel = 1.0 if item in truth else 0.0
        dcg += rel / math.log2(i + 1)

    ideal_hits = min(k, len(truth))
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def build_final_response(query: str, recommendations: List[Dict]) -> str:
    lines = [f'Based on your query "{query}", here are my top recommendations in Pula:\n']
    for i, rec in enumerate(recommendations, 1):
        name = rec.get("name", "Unknown")
        rtype = rec.get("type", "place")
        desc = rec.get("short_answer") or rec.get("reason", "")
        lines.append(f"{i}. {name} ({rtype}) — {desc}")
    return "\n".join(lines)


def maybe_login_hf() -> None:
    try:
        from huggingface_hub import login

        if os.getenv("HF_TOKEN"):
            login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def run(args: argparse.Namespace) -> None:
    maybe_login_hf()

    places = load_places(args.places_file)
    place_lookup = build_place_lookup(places)
    profiles = load_profiles(args.profiles_file)
    labels = load_gold_labels(args.labels_file)

    if args.max_profiles is not None:
        profiles = profiles[: args.max_profiles]

    if not profiles:
        raise ValueError("No profiles to process.")

    os.makedirs(args.output_dir, exist_ok=True)
    sanitized_model_name = re.sub(r"[/:.]", "_", args.model_id)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    jsonl_path = os.path.join(
        args.output_dir,
        f"tourism_eval_raw_{sanitized_model_name}_{timestamp}.jsonl",
    )
    raw_only_path = os.path.join(
        args.output_dir,
        f"tourism_eval_raw_unparsed_{sanitized_model_name}_{timestamp}.jsonl",
    )
    csv_path = os.path.join(
        args.output_dir,
        f"tourism_eval_summary_{sanitized_model_name}_{timestamp}.csv",
    )
    metadata_path = os.path.join(
        args.output_dir,
        f"tourism_eval_metadata_{sanitized_model_name}_{timestamp}.json",
    )
    performance_path = os.path.join(
        args.output_dir,
        f"tourism_eval_performance_{sanitized_model_name}_{timestamp}.jsonl",
    )

    print(f"Loaded {len(places)} places and {len(profiles)} profiles.")
    print(f"Output JSONL: {jsonl_path}")
    print(f"Output Raw JSONL: {raw_only_path}")
    print(f"Output CSV: {csv_path}")
    print(f"Output Performance: {performance_path}")

    model = None
    tokenizer = None
    active_backend = args.backend
    if not args.no_model:
        if active_backend == "vllm":
            print(f"Loading vLLM model: {args.model_id}")
            try:
                model = load_vllm_model(
                    args.model_id,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_model_len=args.vllm_max_model_len,
                )
            except Exception as e:
                print(f"vLLM initialization failed: {e}")
                print("Falling back to transformers backend for compatibility.")
                active_backend = "transformers"
                model, tokenizer = load_model_and_tokenizer(
                    args.model_id,
                    offload_folder=args.offload_folder,
                    gpu_memory_gib=args.gpu_memory_gib,
                )
        else:
            print(f"Loading model (transformers): {args.model_id}")
            model, tokenizer = load_model_and_tokenizer(
                args.model_id,
                offload_folder=args.offload_folder,
                gpu_memory_gib=args.gpu_memory_gib,
            )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(top_k=args.top_k)

    # ── Phase 1: prepare all inputs ──────────────────────────────────────────
    all_inputs: List[Tuple[Dict[str, Any], List[Place], str]] = []
    for profile in profiles:
        scored = filter_and_score_candidates(profile, places, strict_price=args.strict_price)
        candidate_places = [x[0] for x in scored[: args.candidate_limit]]
        user_prompt = (
            build_user_prompt(profile, candidate_places, args.top_k)
            if not args.no_model
            else ""
        )
        all_inputs.append((profile, candidate_places, user_prompt))

    # ── Phase 2: batch generation (vLLM) ─────────────────────────────────────
    all_raw_responses: Optional[List[str]] = None
    if active_backend == "vllm" and not args.no_model:
        print(f"Batch-generating {len(all_inputs)} profiles with vLLM …")
        all_raw_responses = batch_generate_vllm(
            model,
            system_prompt,
            [inp[2] for inp in all_inputs],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_model_len=args.vllm_max_model_len,
        )
        print("Generation complete.")
    elif active_backend == "transformers" and not args.no_model:
        print(
            f"Batch-generating {len(all_inputs)} profiles with transformers "
            f"(batch_size={args.transformers_batch_size}) …"
        )
        all_raw_responses = []
        all_user_prompts = [inp[2] for inp in all_inputs]
        for start in range(0, len(all_user_prompts), args.transformers_batch_size):
            end = min(start + args.transformers_batch_size, len(all_user_prompts))
            batch_outputs = batch_generate_transformers(
                model,
                tokenizer,
                system_prompt,
                all_user_prompts[start:end],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )
            all_raw_responses.extend(batch_outputs)
            if end % max(10, args.progress_every) == 0 or end == len(all_user_prompts):
                print(f"Transformers generation progress: {end}/{len(all_user_prompts)}")
        print("Generation complete.")

    summary_rows: List[Dict[str, Any]] = []
    aggregate_metrics = {
        "valid_json_count": 0,
        "parse_error_rows": 0,
        "fallback_rows": 0,
        "rows": 0,
        "recommendation_validity_rate_sum": 0.0,
        "avg_tag_overlap_sum": 0.0,
        "rating_compliance_rate_sum": 0.0,
        "price_compliance_rate_sum": 0.0,
        "type_diversity_sum": 0.0,
        "p_at_k_sum": 0.0,
        "r_at_k_sum": 0.0,
        "ndcg_at_k_sum": 0.0,
        "gold_rows": 0,
    }

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_f, open(raw_only_path, "w", encoding="utf-8") as raw_f, open(performance_path, "w", encoding="utf-8") as perf_f:
        pbar = tqdm(profiles, desc="Evaluating", unit="profile", dynamic_ncols=True)
        for idx, profile in enumerate(pbar):
            current = idx + 1

            if args.no_model:
                parsed = build_heuristic_payload(
                    candidate_places,
                    args.top_k,
                    reason="Heuristic fallback recommendation.",
                    profile=profile,
                )
                raw_response = json.dumps(parsed, ensure_ascii=False)
                parse_error = ""
            elif all_raw_responses is not None:
                # vLLM pre-generated the full batch
                raw_response = all_raw_responses[idx]
                parsed, parse_error = extract_json_payload(raw_response)
                if not parsed:
                    parsed = build_heuristic_payload(
                        candidate_places,
                        args.top_k,
                        reason="Fallback used because model response was not valid JSON.",
                        profile=profile,
                    )
                    parse_error = f"{parse_error}|heuristic_fallback"
            else:
                # Sequential transformers generation
                raw_response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )
                parsed, parse_error = extract_json_payload(raw_response)
                if not parsed:
                    # Keep the run productive even when the model returns non-JSON text.
                    parsed = build_heuristic_payload(
                        candidate_places,
                        args.top_k,
                        reason="Fallback used because model response was not valid JSON.",
                        profile=profile,
                    )
                    parse_error = f"{parse_error}|heuristic_fallback"

            if parse_error:
                aggregate_metrics["parse_error_rows"] += 1
            if "heuristic_fallback" in parse_error:
                aggregate_metrics["fallback_rows"] += 1

            valid_json = int(bool(parsed))
            aggregate_metrics["valid_json_count"] += valid_json

            recommendations, row_metrics = validate_recommendations(
                parsed=parsed,
                place_lookup=place_lookup,
                profile=profile,
                top_k=args.top_k,
            )

            aggregate_metrics["rows"] += 1
            aggregate_metrics["recommendation_validity_rate_sum"] += row_metrics["recommendation_validity_rate"]
            aggregate_metrics["avg_tag_overlap_sum"] += row_metrics["avg_tag_overlap"]
            aggregate_metrics["rating_compliance_rate_sum"] += row_metrics["rating_compliance_rate"]
            aggregate_metrics["price_compliance_rate_sum"] += row_metrics["price_compliance_rate"]
            aggregate_metrics["type_diversity_sum"] += row_metrics["type_diversity"]

            user_id = profile.get("id")
            pred_names = [normalize_token(x.get("name", "")) for x in recommendations if x.get("name")]
            p_at_k = None
            r_at_k = None
            ndcg_k = None

            if isinstance(user_id, int) and user_id in labels:
                truth = labels[user_id]
                p_at_k = precision_at_k(pred_names, truth, args.top_k)
                r_at_k = recall_at_k(pred_names, truth, args.top_k)
                ndcg_k = ndcg_at_k(pred_names, truth, args.top_k)
                aggregate_metrics["gold_rows"] += 1
                aggregate_metrics["p_at_k_sum"] += p_at_k
                aggregate_metrics["r_at_k_sum"] += r_at_k
                aggregate_metrics["ndcg_at_k_sum"] += ndcg_k

            recs = parsed.get("recommendations", []) if parsed else []
            final_response = build_final_response(profile.get("input", ""), recs)

            raw_record = {
                "user_id": user_id,
                "query": profile.get("input"),
                "profile": {
                    "persona": profile.get("persona"),
                    "travel_style": profile.get("travel_style"),
                    "likes": profile.get("likes", []),
                    "dislikes": profile.get("dislikes", []),
                    "price_preference": profile.get("price_preference"),
                    "min_rating": profile.get("min_rating"),
                },
                "candidate_count": len(candidate_places),
                "recommendations": recs,
                "final_response": final_response,
            }
            jsonl_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            jsonl_f.flush()

            perf_record = {
                "user_id": user_id,
                "query": profile.get("input"),
                "persona": profile.get("persona"),
                **row_metrics,
                "p_at_k": p_at_k,
                "r_at_k": r_at_k,
                "ndcg_at_k": ndcg_k,
            }
            perf_f.write(json.dumps(perf_record, ensure_ascii=False) + "\n")
            perf_f.flush()

            raw_only_record = {
                "user_id": user_id,
                "query": profile.get("input"),
                "candidate_count": len(candidate_places),
                "parse_error": parse_error,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw_response,
            }
            raw_f.write(json.dumps(raw_only_record, ensure_ascii=False) + "\n")
            raw_f.flush()

            summary_row: Dict[str, Any] = {
                "user_id": user_id,
                "query": profile.get("input"),
                "persona": profile.get("persona"),
                "travel_style": profile.get("travel_style"),
                "candidate_count": len(candidate_places),
                "recommendation_count": row_metrics["recommendation_count"],
                "recommendation_validity_rate": row_metrics["recommendation_validity_rate"],
                "avg_tag_overlap": row_metrics["avg_tag_overlap"],
                "rating_compliance_rate": row_metrics["rating_compliance_rate"],
                "price_compliance_rate": row_metrics["price_compliance_rate"],
                "type_diversity": row_metrics["type_diversity"],
                "p_at_k": p_at_k,
                "r_at_k": r_at_k,
                "ndcg_at_k": ndcg_k,
            }

            for k in range(args.top_k):
                col_name = f"top_{k + 1}_name"
                col_type = f"top_{k + 1}_type"
                col_score = f"top_{k + 1}_score"
                col_short_answer = f"top_{k + 1}_short_answer"
                if k < len(recommendations):
                    summary_row[col_name] = recommendations[k].get("name")
                    summary_row[col_type] = recommendations[k].get("type")
                    summary_row[col_score] = recommendations[k].get("score")
                    summary_row[col_short_answer] = recommendations[k].get("short_answer")
                else:
                    summary_row[col_name] = ""
                    summary_row[col_type] = ""
                    summary_row[col_score] = ""
                    summary_row[col_short_answer] = ""

            summary_rows.append(summary_row)

            if (idx + 1) % args.save_every == 0:
                write_summary_csv(summary_rows, csv_path)

            rows_so_far = max(1, current)
            pbar.set_postfix({
                "validity": f"{aggregate_metrics['recommendation_validity_rate_sum'] / rows_so_far:.2f}",
                "tag_overlap": f"{aggregate_metrics['avg_tag_overlap_sum'] / rows_so_far:.2f}",
                "rating_ok": f"{aggregate_metrics['rating_compliance_rate_sum'] / rows_so_far:.2f}",
                "price_ok": f"{aggregate_metrics['price_compliance_rate_sum'] / rows_so_far:.2f}",
            })

    write_summary_csv(summary_rows, csv_path)

    rows = max(1, aggregate_metrics["rows"])
    gold_rows = aggregate_metrics["gold_rows"]

    aggregate_performance = {
        "_type": "aggregate",
        "rows_processed": aggregate_metrics["rows"],
        "avg_recommendation_validity_rate": aggregate_metrics["recommendation_validity_rate_sum"] / rows,
        "avg_tag_overlap": aggregate_metrics["avg_tag_overlap_sum"] / rows,
        "avg_rating_compliance_rate": aggregate_metrics["rating_compliance_rate_sum"] / rows,
        "avg_price_compliance_rate": aggregate_metrics["price_compliance_rate_sum"] / rows,
        "avg_type_diversity": aggregate_metrics["type_diversity_sum"] / rows,
        "gold_rows": gold_rows,
        "precision_at_k": (aggregate_metrics["p_at_k_sum"] / gold_rows) if gold_rows else None,
        "recall_at_k": (aggregate_metrics["r_at_k_sum"] / gold_rows) if gold_rows else None,
        "ndcg_at_k": (aggregate_metrics["ndcg_at_k_sum"] / gold_rows) if gold_rows else None,
    }
    with open(performance_path, "a", encoding="utf-8") as perf_f:
        perf_f.write(json.dumps(aggregate_performance, ensure_ascii=False) + "\n")

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "no_model": args.no_model,
        "places_file": args.places_file,
        "profiles_file": args.profiles_file,
        "labels_file": args.labels_file,
        "raw_unparsed_jsonl": raw_only_path,
        "performance_jsonl": performance_path,
        "rows_processed": aggregate_metrics["rows"],
        "top_k": args.top_k,
        "candidate_limit": args.candidate_limit,
        "strict_price": args.strict_price,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "backend_requested": args.backend,
        "backend_used": active_backend,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nRun complete")
    print(json.dumps(metadata, indent=2))
    print(f"Metadata: {metadata_path}")

    if model is not None:
        if active_backend == "vllm":
            del model
        else:
            del model, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tourism prompt evaluation on Pula datasets")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--places-file", type=str, default="pula_merged.json")
    parser.add_argument("--profiles-file", type=str, default="user_profiles.json")
    parser.add_argument("--labels-file", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./prompt_results/tourism_new")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional output token cap. By default, generation can use the remaining model context.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--strict-price", action="store_true")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--offload-folder", type=str, default="./offload")
    parser.add_argument("--gpu-memory-gib", type=int, default=36)
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers"],
        help="Inference backend. 'vllm' uses tensor parallelism across all GPUs and batch generation. "
             "'transformers' uses pipeline-parallel HF model generation (slower, single-sample).",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory vLLM may use (default 0.92).",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=16384,
        help="Maximum sequence length (input + output tokens) for vLLM. "
             "Increase if prompts with 60 candidates exceed 16k tokens.",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model inference and use deterministic heuristic fallback for smoke testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
