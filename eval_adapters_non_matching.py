import argparse
import ast
import csv
import gc
import json
import os
import re
import resource
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an intelligent tourism assistant. You recommend personalized activities to tourists based on:\n"
    "1. The user's interests and preferences (as defined in their profile).\n"
    "2. Their current intent or question (described in the user input).\n"
    "3. Relevant local information retrieved v   semantic search (provided as bullet points).\n"
    "Your response should be personalized, context-aware, and practical.\n"
    "If there are strong user preferences, make sure to respect them.\n"
    "Respond in a friendly and informative tone. Your output should be a short and clear activity recommendation, "
    "followed by a brief explanation if needed."
)


def normalize_token(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def strip_think_sections(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_listish(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    return [x.strip() for x in re.split(r"[;,]", text) if x.strip()]


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9_]+", normalize_token(text)))


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def _bytes_to_mib(value: float) -> float:
    return round(float(value) / (1024.0 * 1024.0), 2)


def _get_current_rss_mib() -> Optional[float]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(float(parts[1]) / 1024.0, 2)
    except Exception:
        return None
    return None


def _get_rusage_snapshot() -> Dict[str, float]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
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


def _reset_cuda_peak_stats() -> None:
    if not torch.cuda.is_available():
        return
    for dev_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(dev_idx)


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

    def _safe_float(val: str) -> Optional[float]:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_utilization_percent": _safe_float(parts[2]),
                    "memory_utilization_percent": _safe_float(parts[3]),
                    "memory_total_mib": _safe_float(parts[4]),
                    "memory_used_mib": _safe_float(parts[5]),
                    "memory_free_mib": _safe_float(parts[6]),
                    "temperature_c": _safe_float(parts[7]),
                    "power_draw_w": _safe_float(parts[8]),
                }
            )
        except Exception:
            continue
    return {"available": True, "gpus": gpus}


def load_eval_rows(csv_path: str, max_rows: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profile = {
                "id": row.get("User_ID"),
                "name": row.get("User_Name"),
                "age": row.get("User_Age"),
                "gender": row.get("User_Gender"),
                "nationality": row.get("User_Nationality"),
                "travel_style": row.get("User_Travel_Style"),
                "persona": row.get("User_Persona"),
                "likes": _parse_listish(row.get("User_Likes")),
                "dislikes": _parse_listish(row.get("User_Dislikes")),
                "price_preference": row.get("User_Price_Preference"),
                "min_rating": row.get("User_Min_Rating"),
                "input": row.get("User_Input") or "",
            }
            rows.append(
                {
                    "profile": profile,
                    "query": row.get("User_Input") or "",
                    "relevant_information": row.get("Relevant_Information") or "",
                    "reference_response": row.get("LLM_Response") or "",
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def build_user_prompt(profile: Dict[str, Any], relevant_information: str) -> str:
    profile_summary = (
        f"Name: {profile.get('name', 'Unknown')}\n"
        f"Travel style: {profile.get('travel_style', 'N/A')}\n"
        f"Persona: {profile.get('persona', 'N/A')}\n"
        f"Likes: {', '.join(profile.get('likes', [])) or 'none specified'}\n"
        f"Dislikes: {', '.join(profile.get('dislikes', [])) or 'none specified'}\n"
        f"Price preference: {profile.get('price_preference', 'N/A')}\n"
        f"Minimum rating: {profile.get('min_rating', 'N/A')}"
    )

    retrieved_block = relevant_information.strip() or "No relevant information found."
    if not retrieved_block.startswith("1.") and not retrieved_block.startswith("•"):
        retrieved_block = "• " + retrieved_block

    query = profile.get("input", "What do you recommend?")
    return (
        f"User profile:\n{profile_summary}\n\n"
        f"User question: {query}\n\n"
        f"Retrieved local information:\n{retrieved_block}"
    )


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_base_model_id(adapter_dir: str) -> str:
    metrics_path = os.path.join(adapter_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        metrics = read_json(metrics_path)
        model_id = str(metrics.get("model_id") or "").strip()
        if model_id:
            return model_id

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        cfg = read_json(config_path)
        model_id = str(cfg.get("base_model_name_or_path") or "").strip()
        if model_id:
            return model_id

    raise ValueError(f"Could not resolve base model for adapter: {adapter_dir}")


def load_model_and_tokenizer(base_model_id: str, adapter_dir: str, gpu_memory_gib: int):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    load_kwargs = {
        "quantization_config": bnb_config,
        "trust_remote_code": True,
        "dtype": torch.float16,
    }

    if n_gpus >= 2:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": 0}
        load_kwargs["max_memory"] = {0: f"{gpu_memory_gib}GiB"}

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    # IA3 scaling vectors (stored in ia3_l ParameterDict) load as float32 by
    # default. The bitsandbytes layers output float16 (bnb_4bit_compute_dtype),
    # so multiplying by a float32 scaling vector upcasts key/value to float32
    # while query stays float16, causing the attention dtype mismatch.
    # Cast them directly by name rather than iterating all parameters (which
    # would conflict with Params4bit internals).
    for module in model.modules():
        if hasattr(module, 'ia3_l'):
            for adapter_param in module.ia3_l.values():
                adapter_param.data = adapter_param.data.to(torch.float16)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    do_sample: bool,
    temperature: float,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        prompt = f"{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
        inputs = tokenizer(prompt, return_tensors="pt")

    input_len = inputs["input_ids"].shape[-1]
    inputs = inputs.to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_length": model.config.max_position_embeddings,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_row_metrics(response: str, reference: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    response_norm = normalize_token(response)
    word_count = len(response.split())

    likes = [normalize_token(x) for x in profile.get("likes", []) if x]
    dislikes = [normalize_token(x) for x in profile.get("dislikes", []) if x]

    likes_mentioned = sum(1 for tag in likes if tag in response_norm)
    dislikes_mentioned = sum(1 for tag in dislikes if tag in response_norm)

    response_tokens = _token_set(response)
    ref_tokens = _token_set(reference)
    token_overlap = _safe_div(len(response_tokens.intersection(ref_tokens)), len(ref_tokens)) if ref_tokens else None

    return {
        "word_count": word_count,
        "is_empty": word_count == 0,
        "likes_mentioned": likes_mentioned,
        "total_likes": len(likes),
        "preference_mention_rate": _safe_div(likes_mentioned, len(likes)) if likes else None,
        "dislikes_mentioned": dislikes_mentioned,
        "total_dislikes": len(dislikes),
        "dislike_avoid_rate": _safe_div(len(dislikes) - dislikes_mentioned, len(dislikes)) if dislikes else None,
        "reference_token_overlap": token_overlap,
    }


def compute_similarity_scores(
    embed_model: SentenceTransformer,
    generated: List[str],
    references: List[str],
) -> List[Optional[float]]:
    if not generated or not references:
        return []

    gen_emb = embed_model.encode(generated, convert_to_numpy=True, normalize_embeddings=True)
    ref_emb = embed_model.encode(references, convert_to_numpy=True, normalize_embeddings=True)
    scores = np.sum(gen_emb * ref_emb, axis=1)
    return [float(x) for x in scores]


def write_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "rows_processed": 0,
            "empty_response_rate": 0.0,
            "avg_word_count": 0.0,
            "avg_preference_mention_rate": 0.0,
            "avg_dislike_avoid_rate": 0.0,
            "avg_reference_token_overlap": 0.0,
            "avg_reference_similarity": 0.0,
            "combined_score": 0.0,
        }

    empty = sum(1 for r in rows if r.get("is_empty"))
    avg_words = sum(float(r.get("word_count", 0)) for r in rows) / total

    pref_vals = [r.get("preference_mention_rate") for r in rows if r.get("preference_mention_rate") is not None]
    dis_vals = [r.get("dislike_avoid_rate") for r in rows if r.get("dislike_avoid_rate") is not None]
    tok_vals = [r.get("reference_token_overlap") for r in rows if r.get("reference_token_overlap") is not None]
    sim_vals = [r.get("reference_similarity") for r in rows if r.get("reference_similarity") is not None]

    avg_pref = float(sum(pref_vals) / len(pref_vals)) if pref_vals else 0.0
    avg_dis = float(sum(dis_vals) / len(dis_vals)) if dis_vals else 0.0
    avg_tok = float(sum(tok_vals) / len(tok_vals)) if tok_vals else 0.0
    avg_sim = float(sum(sim_vals) / len(sim_vals)) if sim_vals else 0.0

    non_empty_rate = 1.0 - (empty / total)
    combined_score = (
        0.20 * non_empty_rate
        + 0.25 * avg_pref
        + 0.25 * avg_dis
        + 0.15 * avg_tok
        + 0.15 * max(0.0, min(1.0, (avg_sim + 1.0) / 2.0))
    )

    return {
        "rows_processed": total,
        "empty_response_rate": empty / total,
        "avg_word_count": avg_words,
        "avg_preference_mention_rate": avg_pref,
        "avg_dislike_avoid_rate": avg_dis,
        "avg_reference_token_overlap": avg_tok,
        "avg_reference_similarity": avg_sim,
        "combined_score": combined_score,
    }


def discover_adapter_dirs(adapters_root: str, adapter_filter: str) -> List[str]:
    if not os.path.isdir(adapters_root):
        raise FileNotFoundError(f"Adapters directory not found: {adapters_root}")

    names = sorted([x for x in os.listdir(adapters_root) if os.path.isdir(os.path.join(adapters_root, x))])
    selected = set(x.strip() for x in adapter_filter.split(",") if x.strip())
    if selected:
        names = [x for x in names if x in selected]

    dirs = [os.path.join(adapters_root, name) for name in names]
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "adapter_model.safetensors"))]
    return dirs


def evaluate_adapter(
    adapter_dir: str,
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    sim_model: Optional[SentenceTransformer],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    adapter_name = os.path.basename(adapter_dir)
    base_model_id = resolve_base_model_id(adapter_dir)

    print("=" * 80)
    print(f"Evaluating adapter: {adapter_name}")
    print(f"Base model      : {base_model_id}")
    print("=" * 80)

    _reset_cuda_peak_stats()
    rusage_start = _get_rusage_snapshot()
    cuda_start = _get_cuda_memory_snapshot()
    nvidia_smi_start = _get_nvidia_smi_snapshot()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    model, tokenizer = load_model_and_tokenizer(base_model_id, adapter_dir, args.gpu_memory_gib)

    out_rows: List[Dict[str, Any]] = []
    generated_texts: List[str] = []
    reference_texts: List[str] = []

    pbar = tqdm(rows, desc=f"{adapter_name[:28]}", unit="row", dynamic_ncols=True)
    for row in pbar:
        profile = row["profile"]
        user_prompt = build_user_prompt(profile, row.get("relevant_information", ""))
        raw = generate_response(
            model=model,
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            do_sample=args.do_sample,
            temperature=args.temperature,
        )
        response = strip_think_sections(raw)

        reference = row.get("reference_response", "")
        metrics = compute_row_metrics(response, reference, profile)

        output_row = {
            "adapter_name": adapter_name,
            "base_model_id": base_model_id,
            "user_id": profile.get("id"),
            "user_name": profile.get("name"),
            "query": row.get("query", ""),
            "reference_response": reference,
            "generated_response": response,
            **metrics,
        }
        out_rows.append(output_row)
        generated_texts.append(response)
        reference_texts.append(reference)

    if sim_model is not None and generated_texts:
        sim_scores = compute_similarity_scores(sim_model, generated_texts, reference_texts)
        for idx, score in enumerate(sim_scores):
            out_rows[idx]["reference_similarity"] = score
    else:
        for row in out_rows:
            row["reference_similarity"] = None

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    rusage_end = _get_rusage_snapshot()
    cuda_end = _get_cuda_memory_snapshot()
    nvidia_smi_end = _get_nvidia_smi_snapshot()

    summary = summarize_rows(out_rows)
    summary.update(
        {
            "adapter_name": adapter_name,
            "adapter_dir": adapter_dir,
            "base_model_id": base_model_id,
            "evaluation_time_seconds": round(wall_elapsed, 2),
        }
    )

    performance_payload = {
        "adapter_name": adapter_name,
        "adapter_dir": adapter_dir,
        "base_model_id": base_model_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_metrics": summary,
        "per_profile_metrics": out_rows,
        "system_performance": {
            "rows_processed": len(out_rows),
            "wall_time_seconds": round(wall_elapsed, 4),
            "cpu_time_seconds": round(cpu_elapsed, 4),
            "rusage_start": rusage_start,
            "rusage_end": rusage_end,
            "cuda_start": cuda_start,
            "cuda_end": cuda_end,
            "nvidia_smi_start": nvidia_smi_start,
            "nvidia_smi_end": nvidia_smi_end,
        },
    }

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return out_rows, summary, performance_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated adapters on split_3_non_matching.csv")

    parser.add_argument("--eval-csv", type=str, default="split_3_non_matching.csv")
    parser.add_argument("--adapters-dir", type=str, default="./adapters_results/ollama/adapters")
    parser.add_argument("--output-dir", type=str, default="./adapters_results/ollama/eval_non_matching")
    parser.add_argument(
        "--adapter-filter",
        type=str,
        default="ia3-qwen3-4b,lora-llama3.1,lora-llama3.2,lora-qwen3-4b",
        help=(
            "Comma-separated adapter directory names to evaluate. Defaults to the adapters "
            "without a completed non-matching evaluation."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--gpu-memory-gib", type=int, default=36)

    parser.add_argument("--similarity-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--skip-similarity", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_eval_rows(args.eval_csv, args.max_rows)
    if not rows:
        raise ValueError(f"No rows found in evaluation CSV: {args.eval_csv}")

    adapter_dirs = discover_adapter_dirs(args.adapters_dir, args.adapter_filter)
    if not adapter_dirs:
        raise ValueError(f"No adapter directories found in: {args.adapters_dir}")

    sim_model = None
    if not args.skip_similarity:
        print(f"Loading similarity model: {args.similarity_model}")
        sim_model = SentenceTransformer(args.similarity_model)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    leaderboard: List[Dict[str, Any]] = []

    for adapter_dir in adapter_dirs:
        try:
            row_results, summary, performance_payload = evaluate_adapter(adapter_dir, rows, args, sim_model)
        except Exception as exc:
            print(f"Evaluation failed for {adapter_dir}: {exc}")
            leaderboard.append(
                {
                    "adapter_name": os.path.basename(adapter_dir),
                    "adapter_dir": adapter_dir,
                    "error": str(exc),
                }
            )
            # Attempt CUDA state recovery so subsequent adapters can still run.
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
            continue

        adapter_name = summary["adapter_name"]
        rows_json_path = os.path.join(args.output_dir, f"adapter_eval_rows_{adapter_name}_{timestamp}.json")
        rows_csv_path = os.path.join(args.output_dir, f"adapter_eval_rows_{adapter_name}_{timestamp}.csv")
        summary_path = os.path.join(args.output_dir, f"adapter_eval_summary_{adapter_name}_{timestamp}.json")
        performance_path = os.path.join(args.output_dir, f"adapter_eval_performance_{adapter_name}_{timestamp}.json")

        write_json(row_results, rows_json_path)
        write_csv(row_results, rows_csv_path)
        write_json(summary, summary_path)
        write_json(performance_payload, performance_path)
        leaderboard.append(summary)

        print(f"Saved rows: {rows_csv_path}")
        print(f"Saved summary: {summary_path}")
        print(f"Saved performance: {performance_path}")

    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda x: float(x.get("combined_score", -1.0)) if "combined_score" in x else -1.0,
        reverse=True,
    )

    leaderboard_json = os.path.join(args.output_dir, f"adapter_eval_leaderboard_{timestamp}.json")
    leaderboard_csv = os.path.join(args.output_dir, f"adapter_eval_leaderboard_{timestamp}.csv")
    write_json(leaderboard_sorted, leaderboard_json)
    write_csv(leaderboard_sorted, leaderboard_csv)

    print("=" * 80)
    print("Evaluation complete")
    print(f"Leaderboard JSON: {leaderboard_json}")
    print(f"Leaderboard CSV : {leaderboard_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
