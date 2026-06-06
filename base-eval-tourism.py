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
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


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


RERANK_OUTPUT_FIELDS = [
    "User_ID",
    "User_Name",
    "User_Age",
    "User_Gender",
    "User_Nationality",
    "User_Travel_Style",
    "User_Persona",
    "User_Likes",
    "User_Dislikes",
    "User_Price_Preference",
    "User_Min_Rating",
    "User_Input",
    "Relevant_Information",
    "LLM_Response",
]


def normalize_token(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def strip_think_sections(text: str) -> str:
    """Remove Qwen-style <think>...</think> blocks from a response."""
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


def _profile_from_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
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


def load_profiles(file_path: str) -> List[Dict[str, Any]]:
    if file_path.lower().endswith(".csv"):
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [_profile_from_csv_row(row) for row in reader]

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    try:
        import torch
    except Exception:
        return {"cuda_available": False, "devices": [], "error": "torch-not-available"}

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
    try:
        import torch

        if torch.cuda.is_available():
            for dev_idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev_idx)
    except Exception:
        pass


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


def load_faiss_index(index_path: str, chunks_path: str):
    import faiss
    import numpy as np

    index = faiss.read_index(index_path)
    chunks = np.load(chunks_path, allow_pickle=True).tolist()
    return index, chunks


def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_embedding_dimension(embed_model) -> int:
    dim = None
    if hasattr(embed_model, "get_sentence_embedding_dimension"):
        dim = embed_model.get_sentence_embedding_dimension()
    if dim is None:
        sample = embed_model.encode(["dimension probe"], convert_to_numpy=True)
        dim = int(sample.shape[1])
    return int(dim)


def retrieve_chunks(
    query: str,
    index,
    chunks: List[str],
    embed_model,
    top_k: int,
) -> List[str]:
    import numpy as np

    embedding = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    _, indices = index.search(embedding, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(str(chunks[idx]))
    return results


def build_user_prompt(profile: Dict[str, Any], retrieved_chunks: List[str]) -> str:
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


def format_retrieved_chunks_for_output(results: List[str]) -> str:
    if not results:
        return "No relevant information found."
    return "\n".join(f"{i}. {chunk}" for i, chunk in enumerate(results, start=1))


def build_rerank_style_result_row(
    profile: Dict[str, Any],
    query: str,
    retrieved_chunks: List[str],
    final_response: str,
) -> Dict[str, Any]:
    return {
        "User_ID": profile.get("id", "Unknown"),
        "User_Name": profile.get("name", "Unknown"),
        "User_Age": profile.get("age", "Unknown"),
        "User_Gender": profile.get("gender", "Unknown"),
        "User_Nationality": profile.get("nationality", "Unknown"),
        "User_Travel_Style": profile.get("travel_style", "Unknown"),
        "User_Persona": profile.get("persona", "Unknown"),
        "User_Likes": str(profile.get("likes", [])),
        "User_Dislikes": str(profile.get("dislikes", [])),
        "User_Price_Preference": profile.get("price_preference", "Unknown"),
        "User_Min_Rating": profile.get("min_rating", "Unknown"),
        "User_Input": query,
        "Relevant_Information": format_retrieved_chunks_for_output(retrieved_chunks),
        "LLM_Response": final_response,
    }


def write_results_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RERANK_OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RERANK_OUTPUT_FIELDS})


def write_results_json(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


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


def load_model_and_tokenizer(model_id: str, gpu_memory_gib: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if n_gpus >= 2:
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        load_label = f"multi-GPU 4-bit ({n_gpus} GPUs)"
    else:
        load_kwargs = {
            "quantization_config": bnb_config,
            "device_map": {"": 0},
            "max_memory": {0: f"{gpu_memory_gib}GiB"},
            "trust_remote_code": True,
        }
        load_label = "single-GPU 4-bit"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except (ValueError, RuntimeError) as e:
        raise RuntimeError(
            f"Model loading failed on {load_label}. "
            "Try increasing requested GPUs, reducing --gpu-memory-gib, or using a smaller model. "
            f"Original error: {e}"
        ) from e

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    import torch

    inputs = build_inputs(system_prompt, user_prompt, tokenizer)
    input_len = inputs["input_ids"].shape[-1]
    inputs = inputs.to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_response_metrics(
    response: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    response_lower = response.lower()
    word_count = len(response.split())

    likes = [normalize_token(x) for x in profile.get("likes", []) if x]
    dislikes = [normalize_token(x) for x in profile.get("dislikes", []) if x]

    likes_mentioned = sum(1 for tag in likes if tag in response_lower)
    dislikes_mentioned = sum(1 for tag in dislikes if tag in response_lower)

    preference_mention_rate = likes_mentioned / len(likes) if likes else None
    dislike_avoid_rate = (
        (len(dislikes) - dislikes_mentioned) / len(dislikes) if dislikes else None
    )

    return {
        "word_count": word_count,
        "is_empty": word_count == 0,
        "likes_mentioned": likes_mentioned,
        "total_likes": len(likes),
        "preference_mention_rate": preference_mention_rate,
        "dislikes_mentioned": dislikes_mentioned,
        "total_dislikes": len(dislikes),
        "dislike_avoid_rate": dislike_avoid_rate,
    }


def write_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: List[str] = []
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_login_hf() -> None:
    try:
        from huggingface_hub import login

        if os.getenv("HF_TOKEN"):
            login(token=os.getenv("HF_TOKEN"), add_to_git_credential=False)
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")


def run_single_model(
    model_id: str,
    args: argparse.Namespace,
    profiles: List[Dict[str, Any]],
    faiss_index,
    chunks: List[str],
    embed_model,
) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    sanitized_model_name = re.sub(r"[/:.]", "_", model_id)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    results_json_path = os.path.join(args.output_dir, f"base_eval_results_{sanitized_model_name}_{timestamp}.json")
    results_csv_path = os.path.join(args.output_dir, f"base_eval_results_{sanitized_model_name}_{timestamp}.csv")
    performance_json_path = os.path.join(
        args.output_dir,
        f"base_eval_performance_{sanitized_model_name}_{timestamp}.json",
    )

    print(f"\n{'='*60}")
    print(f"Model        : {model_id}")
    print(f"Output JSON  : {results_json_path}")
    print(f"Output CSV   : {results_csv_path}")
    print(f"Performance  : {performance_json_path}")
    print(f"{'='*60}")

    _reset_cuda_peak_stats()
    rusage_start = _get_rusage_snapshot()
    cuda_start = _get_cuda_memory_snapshot()
    nvidia_smi_start = _get_nvidia_smi_snapshot()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    model = None
    tokenizer = None
    if not args.no_model:
        print(f"Loading model: {model_id}")
        model, tokenizer = load_model_and_tokenizer(model_id, args.gpu_memory_gib)

    aggregate: Dict[str, Any] = {
        "rows": 0,
        "empty_responses": 0,
        "word_count_sum": 0,
        "preference_mention_rate_sum": 0.0,
        "preference_mention_rate_count": 0,
        "dislike_avoid_rate_sum": 0.0,
        "dislike_avoid_rate_count": 0,
    }

    results_rows: List[Dict[str, Any]] = []
    per_profile_performance_rows: List[Dict[str, Any]] = []

    pbar = tqdm(profiles, desc=f"Evaluating {sanitized_model_name[:30]}", unit="profile", dynamic_ncols=True)
    for profile in pbar:
            query = profile.get("input", "What do you recommend?")

            retrieved = retrieve_chunks(query, faiss_index, chunks, embed_model, args.retrieval_top_k)
            user_prompt = build_user_prompt(profile, retrieved)

            if args.no_model:
                final_response = f"[no-model mode] Recommendation for {profile.get('name', 'user')} based on: {query}"
            else:
                final_response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                )

            final_response = strip_think_sections(final_response)

            metrics = compute_response_metrics(final_response, profile)

            aggregate["rows"] += 1
            if metrics["is_empty"]:
                aggregate["empty_responses"] += 1
            aggregate["word_count_sum"] += metrics["word_count"]
            if metrics["preference_mention_rate"] is not None:
                aggregate["preference_mention_rate_sum"] += metrics["preference_mention_rate"]
                aggregate["preference_mention_rate_count"] += 1
            if metrics["dislike_avoid_rate"] is not None:
                aggregate["dislike_avoid_rate_sum"] += metrics["dislike_avoid_rate"]
                aggregate["dislike_avoid_rate_count"] += 1

            results_row = build_rerank_style_result_row(
                profile=profile,
                query=query,
                retrieved_chunks=retrieved,
                final_response=final_response,
            )
            results_rows.append(results_row)

            perf_record = {
                "user_id": profile.get("id"),
                "query": query,
                "persona": profile.get("persona"),
                **metrics,
            }
            per_profile_performance_rows.append(perf_record)

            if len(results_rows) % args.save_every == 0:
                write_results_csv(results_rows, results_csv_path)
                write_results_json(results_rows, results_json_path)

            rows_so_far = max(1, aggregate["rows"])
            pmr = aggregate["preference_mention_rate_sum"] / max(1, aggregate["preference_mention_rate_count"])
            dar = aggregate["dislike_avoid_rate_sum"] / max(1, aggregate["dislike_avoid_rate_count"])
            pbar.set_postfix({
                "avg_words": f"{aggregate['word_count_sum'] / rows_so_far:.0f}",
                "pref_mention": f"{pmr:.2f}",
                "dislike_avoid": f"{dar:.2f}",
                "empty": aggregate["empty_responses"],
            })

    write_results_csv(results_rows, results_csv_path)
    write_results_json(results_rows, results_json_path)

    rows = max(1, aggregate["rows"])
    pmr_count = max(1, aggregate["preference_mention_rate_count"])
    dar_count = max(1, aggregate["dislike_avoid_rate_count"])

    aggregate_performance = {
        "rows_processed": aggregate["rows"],
        "empty_response_rate": aggregate["empty_responses"] / rows,
        "avg_word_count": aggregate["word_count_sum"] / rows,
        "avg_preference_mention_rate": aggregate["preference_mention_rate_sum"] / pmr_count,
        "avg_dislike_avoid_rate": aggregate["dislike_avoid_rate_sum"] / dar_count,
    }

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    rusage_end = _get_rusage_snapshot()
    cuda_end = _get_cuda_memory_snapshot()
    nvidia_smi_end = _get_nvidia_smi_snapshot()

    performance_payload = {
        "model_id": model_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "aggregate_metrics": aggregate_performance,
        "per_profile_metrics": per_profile_performance_rows,
        "system_performance": {
            "rows_processed": aggregate["rows"],
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
    with open(performance_json_path, "w", encoding="utf-8") as f:
        json.dump(performance_payload, f, ensure_ascii=False, indent=2)

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "no_model": args.no_model,
        "profiles_file": args.profiles_file,
        "faiss_index": args.faiss_index,
        "chunks_file": args.chunks_file,
        "embed_model": args.embed_model,
        "retrieval_top_k": args.retrieval_top_k,
        "results_json": results_json_path,
        "results_csv": results_csv_path,
        "performance_json": performance_json_path,
        "rows_processed": aggregate["rows"],
        "avg_word_count": aggregate_performance["avg_word_count"],
        "avg_preference_mention_rate": aggregate_performance["avg_preference_mention_rate"],
        "avg_dislike_avoid_rate": aggregate_performance["avg_dislike_avoid_rate"],
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    print(f"\nModel {model_id} complete.")
    print(json.dumps(metadata, indent=2))
    print(f"Results JSON: {results_json_path}")
    print(f"Results CSV: {results_csv_path}")
    print(f"Performance JSON: {performance_json_path}")

    if model is not None:
        del model, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    return metadata


def run(args: argparse.Namespace) -> None:
    maybe_login_hf()

    profiles = load_profiles(args.profiles_file)
    if args.max_profiles is not None:
        profiles = profiles[: args.max_profiles]
    if not profiles:
        raise ValueError("No profiles to process.")

    print(f"Loading FAISS index from {args.faiss_index} and chunks from {args.chunks_file}")
    faiss_index, chunks = load_faiss_index(args.faiss_index, args.chunks_file)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model: {args.embed_model}")
    embed_model = load_embedding_model(args.embed_model)
    embed_dim = get_embedding_dimension(embed_model)
    print(f"Embedding dimension: {embed_dim} | FAISS index dimension: {faiss_index.d}")
    if embed_dim != faiss_index.d:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"index expects {faiss_index.d}, but model '{args.embed_model}' outputs {embed_dim}. "
            "Use an embedding model with matching dimension (for a 768-d index, "
            "try 'sentence-transformers/all-mpnet-base-v2')."
        )

    print(f"\nLoaded {len(profiles)} profiles. Running {len(args.model_ids)} model(s).")

    all_metadata: List[Dict[str, Any]] = []
    for i, model_id in enumerate(args.model_ids, start=1):
        print(f"\n[{i}/{len(args.model_ids)}] Starting evaluation for: {model_id}")
        metadata = run_single_model(model_id, args, profiles, faiss_index, chunks, embed_model)
        all_metadata.append(metadata)

    print("\n" + "=" * 60)
    print("All models complete. Summary:")
    for m in all_metadata:
        print(
            f"  {m['model_id']:<50} "
            f"avg_words={m['avg_word_count']:.1f}  "
            f"pref_mention={m['avg_preference_mention_rate']:.3f}  "
            f"dislike_avoid={m['avg_dislike_avoid_rate']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base model RAG evaluation for tourism recommendations")
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=[
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen3-4B",
        ],
        help="One or more HuggingFace model IDs to evaluate sequentially.",
    )
    parser.add_argument("--profiles-file", type=str, default="split_3_non_matching.csv")
    parser.add_argument("--faiss-index", type=str, default="faiss_index.bin")
    parser.add_argument("--chunks-file", type=str, default="chunks.npy")
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-transformers model used to embed queries for FAISS retrieval.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="Number of chunks to retrieve per query.")
    parser.add_argument("--output-dir", type=str, default="./base_results")
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--gpu-memory-gib", type=int, default=36)
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model inference for smoke testing the pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
