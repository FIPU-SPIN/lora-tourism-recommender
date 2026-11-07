import argparse
import ast
import gc
import json
import os
import re
import resource
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import IA3Config
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

PRICE_ORDER = {"low": 0, "medium": 1, "high": 2}

SYSTEM_PROMPT = (
    "You are an intelligent tourism assistant. You recommend personalized activities to tourists based on:\n"
    "1. The user's interests and preferences (as defined in their profile).\n"
    "2. Their current intent or question (described in the user input).\n"
    "3. Relevant local information retrieved via semantic search (provided as bullet points).\n"
    "Your response should be personalized, context-aware, and practical.\n"
    "If there are strong user preferences, make sure to respect them.\n"
    "Respond in a friendly and informative tone. Your output should be a short and clear activity recommendation, "
    "followed by a brief explanation if needed."
)

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


def load_faiss_index(index_path: str, chunks_path: str):
    import faiss
    import numpy as np

    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True).tolist()
    return index, chunks


NOMIC_EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def load_embedding_model(model_name: str):
    """Load the query embedding model.

    The FAISS index was built with nomic-embed-text v1.5 (through Ollama), so the
    queries must be embedded with the same model. Use "ollama:<model>" (for example
    "ollama:nomic-embed-text:latest") to embed through a running Ollama server,
    exactly as when the index was built; otherwise the Hugging Face version of the
    model is loaded with sentence-transformers.
    """
    if model_name.startswith("ollama:"):
        return ("ollama", model_name.split(":", 1)[1])

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, trust_remote_code=True)


def embed_query(embed_model, query: str):
    """Return the query embedding as a float32 array of shape (1, dim)."""
    import numpy as np

    if isinstance(embed_model, tuple) and embed_model[0] == "ollama":
        import ollama

        response = ollama.embed(model=embed_model[1], input=query)
        return np.array(response["embeddings"], dtype="float32").reshape(1, -1)

    return embed_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")


def retrieve_chunks(
    query: str,
    index,
    chunks: List[str],
    embed_model,
    top_k: int,
) -> List[str]:
    embedding = embed_query(embed_model, query)
    _, indices = index.search(embedding, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(str(chunks[idx]))
    return results


def _parse_listish(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return [x.strip() for x in re.split(r"[;,]", value) if x.strip()]


def _row_to_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("User_ID"),
        "name": row.get("User_Name"),
        "travel_style": row.get("User_Travel_Style"),
        "persona": row.get("User_Persona"),
        "likes": _parse_listish(row.get("User_Likes")),
        "dislikes": _parse_listish(row.get("User_Dislikes")),
        "price_preference": row.get("User_Price_Preference"),
        "min_rating": row.get("User_Min_Rating"),
        "input": row.get("User_Input") or "",
    }


def build_user_prompt_with_chunks(profile: Dict[str, Any], retrieved_chunks: List[str]) -> str:
    profile_summary = (
        f"Name: {profile.get('name', 'Unknown')}\n"
        f"Travel style: {profile.get('travel_style', 'N/A')}\n"
        f"Persona: {profile.get('persona', 'N/A')}\n"
        f"Likes: {', '.join(profile.get('likes', [])) or 'none specified'}\n"
        f"Dislikes: {', '.join(profile.get('dislikes', [])) or 'none specified'}\n"
        f"Price preference: {profile.get('price_preference', 'N/A')}\n"
        f"Minimum rating: {profile.get('min_rating', 'N/A')}"
    )

    bullets = "\n".join(f"• {c}" for c in retrieved_chunks) if retrieved_chunks else "• No relevant information found."
    query = profile.get("input", "What do you recommend?")

    return (
        f"User profile:\n{profile_summary}\n\n"
        f"User question: {query}\n\n"
        f"Retrieved local information:\n{bullets}"
    )


def load_csv_examples_with_faiss(
    csv_path: str,
    faiss_index,
    chunks: List[str],
    embed_model,
    retrieval_top_k: int,
    max_rows: Optional[int],
) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    examples: List[Dict[str, str]] = []
    for row in df.to_dict(orient="records"):
        profile = _row_to_profile(row)
        query = str(profile.get("input") or "").strip()
        if not query:
            continue

        assistant_response = str(row.get("LLM_Response") or "").strip()
        if not assistant_response:
            continue

        retrieved = retrieve_chunks(query, faiss_index, chunks, embed_model, retrieval_top_k)
        user_prompt = build_user_prompt_with_chunks(profile, retrieved)
        examples.append(
            {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "assistant_json": assistant_response,
            }
        )

    return examples


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


def _parse_module_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def resolve_ia3_target_modules(model, requested_modules: str) -> List[str]:
    requested = _parse_module_list(requested_modules)
    if requested:
        return requested

    preferred = ["k_proj", "v_proj", "down_proj"]
    available = set()
    for module_name, _ in model.named_modules():
        suffix = module_name.split(".")[-1]
        if suffix in preferred:
            available.add(suffix)

    if available:
        return [m for m in preferred if m in available]

    return ["k_proj", "v_proj"]


def resolve_ia3_feedforward_modules(target_modules: List[str], requested_modules: str) -> List[str]:
    requested = _parse_module_list(requested_modules)
    if requested:
        return requested

    preferred_ff = ["down_proj", "fc2", "c_proj", "dense_4h_to_h"]
    return [m for m in preferred_ff if m in target_modules]


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


def _bytes_to_mib(value: float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 2)


def _get_current_rss_mib() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # VmRSS is reported in kilobytes on Linux.
                        return round(float(parts[1]) / 1024.0, 2)
    except Exception:
        return None
    return None


def _get_rusage_snapshot() -> Dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux ru_maxrss is in kilobytes.
    return {
        "current_rss_mib": _get_current_rss_mib(),
        "max_rss_mib": round(float(usage.ru_maxrss) / 1024.0, 2),
        "user_cpu_seconds": round(float(usage.ru_utime), 4),
        "system_cpu_seconds": round(float(usage.ru_stime), 4),
    }


def _get_cuda_memory_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "devices": []}

    devices = []
    for dev_idx in range(torch.cuda.device_count()):
        devices.append(
            {
                "index": dev_idx,
                "name": torch.cuda.get_device_name(dev_idx),
                "memory_allocated_mib": _bytes_to_mib(torch.cuda.memory_allocated(dev_idx)),
                "memory_reserved_mib": _bytes_to_mib(torch.cuda.memory_reserved(dev_idx)),
                "max_memory_allocated_mib": _bytes_to_mib(torch.cuda.max_memory_allocated(dev_idx)),
                "max_memory_reserved_mib": _bytes_to_mib(torch.cuda.max_memory_reserved(dev_idx)),
            }
        )
    return {"cuda_available": True, "devices": devices}


def _get_nvidia_smi_snapshot() -> Dict[str, Any]:
    query = (
        "index,name,utilization.gpu,utilization.memory,memory.total,memory.used,"
        "memory.free,temperature.gpu,power.draw"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "gpu_utilization_percent": float(parts[2]),
                "memory_utilization_percent": float(parts[3]),
                "memory_total_mib": float(parts[4]),
                "memory_used_mib": float(parts[5]),
                "memory_free_mib": float(parts[6]),
                "temperature_c": float(parts[7]),
                "power_draw_w": float(parts[8]),
            }
        )
    return {"available": True, "gpus": gpus}


def build_response_collator(tokenizer):
    """Data collator that computes the loss only on the assistant response.

    The prompt (system prompt, user profile, question and retrieved records) is
    still part of the input, but its tokens are ignored in the loss, so the adapter
    learns to write the response rather than to reproduce the prompt.
    The text that starts the assistant turn is taken from the model's own chat
    template, so the same code works for Llama and Qwen.
    """
    probe = [{"role": "user", "content": "x"}]
    without_prompt = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    response_template = with_prompt[len(without_prompt):].rstrip()
    if not response_template:
        raise ValueError("Could not find the assistant header in the chat template.")
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    print(f"Loss is computed only after the response template: {response_template!r}")
    return DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


def train_single_model(
    args: argparse.Namespace,
    model_id: str,
    train_examples: Sequence[Dict[str, str]],
    eval_examples: Sequence[Dict[str, str]],
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_short = extract_model_short_name(model_id)
    adapter_type = "ia3"
    output_dir = os.path.join(args.output_root, f"{adapter_type}-{model_short}-checkpoints")
    adapter_dir = os.path.join(args.adapters_dir, f"{adapter_type}-{model_short}")

    print("=" * 70)
    print(f"Training model: {model_id}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting training text with model chat template...")
    train_text_rows = [
        {"text": make_chat_text(tokenizer, ex["system_prompt"], ex["user_prompt"], ex["assistant_json"])}
        for ex in train_examples
    ]
    eval_text_rows = [
        {"text": make_chat_text(tokenizer, ex["system_prompt"], ex["user_prompt"], ex["assistant_json"])}
        for ex in eval_examples
    ]
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_text_rows))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_text_rows))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map={"": local_rank},
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    ia3_target_modules = resolve_ia3_target_modules(model, args.ia3_target_modules)
    ia3_feedforward_modules = resolve_ia3_feedforward_modules(
        ia3_target_modules,
        args.ia3_feedforward_modules,
    )

    ia3_kwargs = {
        "task_type": "CAUSAL_LM",
        "target_modules": ia3_target_modules,
    }
    if ia3_feedforward_modules:
        ia3_kwargs["feedforward_modules"] = ia3_feedforward_modules
    peft_config = IA3Config(**ia3_kwargs)

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_length,
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
        tokenizer=tokenizer,
        args=sft_config,
        # Previously no data collator was given, so the loss was computed over the
        # whole sequence (prompt and response). Now only the response counts.
        data_collator=build_response_collator(tokenizer),
    )

    is_main_process = trainer.is_world_process_zero()
    if is_main_process:
        print(f"Train/Eval samples: {len(train_dataset)}/{len(eval_dataset)}")
        trainer.model.print_trainable_parameters()

    if torch.cuda.is_available():
        for dev_idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(dev_idx)

    rusage_start = _get_rusage_snapshot()
    cuda_start = _get_cuda_memory_snapshot()
    nvidia_smi_start = _get_nvidia_smi_snapshot()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    rusage_end = _get_rusage_snapshot()
    cuda_end = _get_cuda_memory_snapshot()
    nvidia_smi_end = _get_nvidia_smi_snapshot()

    os.makedirs(args.adapters_dir, exist_ok=True)
    if is_main_process:
        os.makedirs(adapter_dir, exist_ok=True)
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
            "ia3_target_modules": ia3_target_modules,
            "ia3_feedforward_modules": ia3_feedforward_modules,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "max_length": args.max_length,
        }
        metrics_path = os.path.join(adapter_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        performance_metrics = {
            "model_id": model_id,
            "adapter_dir": adapter_dir,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "wall_time_seconds": round(wall_elapsed, 4),
            "cpu_time_seconds": round(cpu_elapsed, 4),
            "rusage_start": rusage_start,
            "rusage_end": rusage_end,
            "cuda_start": cuda_start,
            "cuda_end": cuda_end,
            "nvidia_smi_start": nvidia_smi_start,
            "nvidia_smi_end": nvidia_smi_end,
        }
        performance_path = os.path.join(adapter_dir, "system_performance_metrics.json")
        with open(performance_path, "w", encoding="utf-8") as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=2)

        print(f"Saved adapter: {adapter_dir}")
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved system performance metrics: {performance_path}")

    del trainer, model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def check_model_compatibility(model_id: str) -> Tuple[bool, Optional[str]]:
    try:
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IA3 adapters for tourism recommendations")

    parser.add_argument(
        "--model-ids",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct,meta-llama/Llama-3.2-3B-Instruct,Qwen/Qwen3-4B",
    )
    parser.add_argument("--places-file", type=str, default="pula_merged.json")
    parser.add_argument("--profiles-file", type=str, default="user_profiles.json")
    parser.add_argument("--train-dataset-file", type=str, default="split_1_both1_2609.csv")
    parser.add_argument("--eval-dataset-file", type=str, default="split_2_both1_870.csv")
    parser.add_argument("--faiss-index", type=str, default="faiss_index.bin")
    parser.add_argument("--chunks-file", type=str, default="chunks.npy")
    parser.add_argument("--embed-model", type=str, default=NOMIC_EMBED_MODEL)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument(
        "--teaching-dataset-file",
        type=str,
        default="",
        help="Optional JSON file with gold recommendations per profile. If set, this is used instead of heuristic labels.",
    )
    parser.add_argument("--output-root", type=str, default="./adapters_results/ollama/checkpoints")
    parser.add_argument("--adapters-dir", type=str, default="./adapters_results/ollama/adapters")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=60)
    parser.add_argument("--strict-price", action="store_true")
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # parser.add_argument("--max-length", type=int, default=512)
    # 512 was used for the results in the paper. It cut almost every training example
    # (about 1790 tokens on average with 5 retrieved records, up to about 3400), so the
    # reference response was usually not part of the training sequence.
    # 6144 covers the longest examples with room for longer ones in the future.
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
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

    parser.add_argument(
        "--ia3-target-modules",
        type=str,
        default="",
        help="Comma-separated IA3 target modules. Empty means auto-detect (default: k_proj,v_proj,down_proj when available).",
    )
    parser.add_argument(
        "--ia3-feedforward-modules",
        type=str,
        default="",
        help="Comma-separated IA3 feedforward modules (must be subset of target modules). Empty means auto-detect.",
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="sdpa")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    maybe_hf_login()

    model_ids = [x.strip() for x in args.model_ids.split(",") if x.strip()]
    if not model_ids:
        raise ValueError("--model-ids must contain at least one model")

    print(f"Loading FAISS index from {args.faiss_index} and chunks from {args.chunks_file}")
    faiss_index, chunks = load_faiss_index(args.faiss_index, args.chunks_file)
    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading embedding model: {args.embed_model}")
    embed_model = load_embedding_model(args.embed_model)

    print(f"Building train examples from: {args.train_dataset_file}")
    train_examples = load_csv_examples_with_faiss(
        csv_path=args.train_dataset_file,
        faiss_index=faiss_index,
        chunks=chunks,
        embed_model=embed_model,
        retrieval_top_k=args.retrieval_top_k,
        max_rows=args.max_profiles,
    )
    print(f"Building eval examples from: {args.eval_dataset_file}")
    eval_examples = load_csv_examples_with_faiss(
        csv_path=args.eval_dataset_file,
        faiss_index=faiss_index,
        chunks=chunks,
        embed_model=embed_model,
        retrieval_top_k=args.retrieval_top_k,
        max_rows=args.max_profiles,
    )

    if not train_examples:
        raise ValueError("No training examples generated from train CSV")
    if not eval_examples:
        raise ValueError("No validation examples generated from eval CSV")

    print(f"Prepared recommendation examples train/eval: {len(train_examples)}/{len(eval_examples)}")
    failed_models: List[str] = []
    skipped_models: List[str] = []
    for model_id in model_ids:
        is_compatible, compatibility_error = check_model_compatibility(model_id)
        if not is_compatible:
            skipped_models.append(model_id)
            print("=" * 70)
            print(f"Skipping unsupported model: {model_id}")
            print(f"Reason: {compatibility_error}")
            print("Continuing with next model...")
            print("=" * 70)
            continue
        try:
            train_single_model(args, model_id, train_examples, eval_examples)
        except Exception as exc:
            failed_models.append(model_id)
            print("=" * 70)
            print(f"Training failed for model: {model_id}")
            print(f"Error: {exc}")
            traceback.print_exc()
            print("Continuing with next model...")
            print("=" * 70)

    if failed_models:
        print("Run completed with failures in the following model(s):")
        for model_id in failed_models:
            print(f"  - {model_id}")
    if skipped_models:
        print("Run skipped unsupported model(s):")
        for model_id in skipped_models:
            print(f"  - {model_id}")
    if not failed_models and not skipped_models:
        print("All requested models finished successfully.")


if __name__ == "__main__":
    main()
