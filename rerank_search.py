import time
import ollama
import os
import json
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
import faiss
import numpy as np
from datetime import datetime
import logging

load_dotenv() 
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding(text: str):
    if not text.strip():
        return None
    
    try:
        response = ollama.embed(
                model='nomic-embed-text:latest',
                input=text)
        emb = response["embeddings"][0]

        return np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def load_faiss_index_and_chunks():
    """
    Load FAISS index and chunks from files.
    Returns:
        index: FAISS index object
        chunks: List of text chunks
    """
    try:
        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True)
        # print(f"Loaded FAISS index with {len(chunks)} chunks.")
        return index, chunks
    except Exception as e:
        logger.error(f"Error loading FAISS index or chunks: {e}")
        return None, None

def search_with_faiss(query: str, index, chunks, initial_k: int = 20, final_k: int = 5):
    """
    FAISS-only semantic search.
    """
    
    # Load FAISS index and chunks
    # try:
    #     index = faiss.read_index(INDEX_FILE)
    #     chunks = np.load(CHUNKS_FILE, allow_pickle=True)
    #     print(f"Loaded FAISS index with {len(chunks)} chunks.")
    # except Exception as e:
    #     print(f"Error loading FAISS index: {e}")
    #     return []
    
    # Get query embedding
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Failed to get embedding for query.")
        return []
    
    # FAISS search
    query_vector = query_embedding.reshape(1, -1)
    faiss_scores, faiss_indices = index.search(query_vector, initial_k)
    
    # Extract retrieved chunks
    retrieved_chunks = []
    for i, idx in enumerate(faiss_indices[0]):
        if idx != -1:  # Valid index
            chunk_text = chunks[idx]
            faiss_score = float(faiss_scores[0][i])
            retrieved_chunks.append({
                'text': chunk_text,
                'faiss_score': faiss_score,
                'index': idx
            })
    
    if not retrieved_chunks:
        print("FAISS search returned no results.")
        return []
    
    return retrieved_chunks[:final_k]

def format_retrieved_chunks(results) -> str:
    """Format retrieved chunks for the prompt"""
    if not results:
        return "No relevant information found."
    
    formatted_chunks = []
    for i, result in enumerate(results, 1):
        formatted_chunks.append(f"{i}. {result['text']}")
    
    return "\n".join(formatted_chunks)

def get_response(message, model_name):
    system_message = {

        "role": "system",
        "content": "You are an intelligent tourism assistant. You recommend personalized activities to tourists based on:\n\
                    1. The user's interests and preferences (as defined in their profile).\n\
                    2. Their current intent or question (described in the user input).\n\
                    3. Relevant local information retrieved via semantic search (provided as bullet points).\n\
                    Your response should be personalized, context-aware, and practical.\n\
                    If there are strong user preferences, make sure to respect them.\n\
                    Respond in a friendly and informative tone. Your output should be a short and clear activity recommendation, followed by a brief explanation if needed."

    }

    if not message or message[0]["role"] != "system":
        message.insert(0, system_message)
    
    try:
        response = completion(
            model=model_name,
            messages= message,
            api_base="http://localhost:11434",
        )

        # Extract content from response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None
                
    except Exception as e:
        print(f"Error making API call: {e}")
        return None
    
def load_user_profiles(json_file_path: str):
    """Load user profiles from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading user profiles: {e}")
        return {}
    
def save_results_to_csv(results, model_name, output_dir="csv_outputs"):
    """Save results to CSV file for each model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name.replace(':', '_').replace('/', '_')}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Add model name column
    df['Model_Name'] = model_name
    
    # Reorder columns to have Model_Name first
    columns = ['Model_Name'] + [col for col in df.columns if col != 'Model_Name']
    df = df[columns]
    
    # Save to CSV
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Results saved to: {filepath}")
    return filepath

def append_result_to_csv(result, model_name, output_dir="csv_outputs"):
    """Append a single result row to a model-specific checkpoint CSV."""
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_filename = f"{model_name.replace(':', '_').replace('/', '_')}_checkpoint.csv"
    checkpoint_filepath = os.path.join(output_dir, checkpoint_filename)

    row = dict(result)
    row['Model_Name'] = model_name
    df = pd.DataFrame([row])

    file_exists = os.path.exists(checkpoint_filepath)
    df.to_csv(
        checkpoint_filepath,
        mode='a',
        header=not file_exists,
        index=False,
        encoding='utf-8'
    )

    return checkpoint_filepath

def build_profile_key(user_id, user_input):
    """Create a stable key for resume checks."""
    return f"{str(user_id).strip()}||{str(user_input).strip()}"

def load_completed_profile_keys(model_name, output_dir="csv_outputs"):
    """Load already processed profile keys from checkpoint CSV."""
    checkpoint_filename = f"{model_name.replace(':', '_').replace('/', '_')}_checkpoint.csv"
    checkpoint_filepath = os.path.join(output_dir, checkpoint_filename)

    if not os.path.exists(checkpoint_filepath):
        return set(), checkpoint_filepath

    try:
        df = pd.read_csv(checkpoint_filepath, encoding='utf-8')
    except Exception as e:
        print(f"Warning: Could not read checkpoint file {checkpoint_filepath}: {e}")
        return set(), checkpoint_filepath

    required_columns = {"User_ID", "User_Input"}
    if not required_columns.issubset(df.columns):
        print(
            f"Warning: Checkpoint {checkpoint_filepath} missing required columns {required_columns}. "
            "Resume will start from beginning."
        )
        return set(), checkpoint_filepath

    completed = set()
    for _, row in df.iterrows():
        completed.add(build_profile_key(row.get("User_ID", "Unknown"), row.get("User_Input", "")))

    return completed, checkpoint_filepath

def main():
    print("Starting FAISS semantic search...")
    if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
        print(f"FAISS files not found. Please run your FAISS building script first.")
        return
    
    user_profiles = load_user_profiles("user_profiles.json")
    if not user_profiles:
        print("No user profiles loaded. Exiting.")
        return

    index, chunks = load_faiss_index_and_chunks()
    if index is None or chunks is None:
        print("Failed to load FAISS index or chunks.")
        return
    
    

    model_names = ["ollama/qwen3.6:35b"]
    total_prompts_per_model = len(user_profiles)
    for model_name in model_names:
        print(f"\nProcessing with LLM model: {model_name}")
        results = []
        completed_profile_keys, checkpoint_path = load_completed_profile_keys(model_name)
        if completed_profile_keys:
            print(f"Resume mode: found {len(completed_profile_keys)} completed profiles in {checkpoint_path}")

        current_prompt_count = 0
        start_time = datetime.now()
        for user_profile in user_profiles:
            user_id = user_profile.get("id", "Unknown")
            user_input = user_profile.get("input", "")
            profile_key = build_profile_key(user_id, user_input)

            if profile_key in completed_profile_keys:
                print(f"Skipping already completed profile: {user_id}")
                continue

            print(f"Processing user profile: {user_id}")

            current_prompt_count += 1
            remaining = total_prompts_per_model - current_prompt_count
            print(f"📊 {model_name}: {current_prompt_count}/{total_prompts_per_model} Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}, (⏳ {remaining} left)")

            user_name = user_profile.get("name", "Unknown")
            user_age = user_profile.get("age", "Unknown")
            user_gender = user_profile.get("gender", "Unknown")
            user_nationality = user_profile.get("nationality", "Unknown")
            user_travel_style = user_profile.get("travel_style", "Unknown")
            user_persona = user_profile.get("persona", "Unknown")
            user_likes = user_profile.get("likes", [])
            user_dislikes = user_profile.get("dislikes", [])
            user_price_preference = user_profile.get("price_preference", "Unknown")
            user_min_rating = user_profile.get("min_rating", "Unknown")

            faiss_results = search_with_faiss(user_input, index, chunks, initial_k=30, final_k=5)
            if not faiss_results:
                print("No results found from FAISS search.")
                continue

            retrieved_chunks = format_retrieved_chunks(faiss_results)

            message = [
                {
                    "role": "user",
                    "content": f"""
                                    User age: 
                                    {user_age}

                                    User name:
                                    {user_name}
                                    
                                    User gender: 
                                    {user_gender}

                                    User nationality:
                                    {user_nationality}

                                    User travel style:
                                    {user_travel_style}

                                    User persona:
                                    {user_persona}
                                    
                                    User likes: 
                                    {user_likes}
                                    
                                    User dislikes: 
                                    {user_dislikes}
                                    
                                    User price preference:
                                    {user_price_preference}

                                    User minimum rating:
                                    {user_min_rating}

                                    User Input:
                                    {user_input}

                                    Relevant Information:
                                    {retrieved_chunks}

                                    Please provide a personalized activity recommendation that considers the user’s preferences, current input, and background information.
                                    """
                }
            ]
            

            # response = get_response(message, model_name = "llama3.1:latest")
            response = get_response(message, model_name)
            time.sleep(6)
            
            if response is None:
                response = "Error: No response from model"
                print(f"Warning: No response from model for user {user_id}")
            
            # Create result record
            result = {
                "User_ID": user_id,
                "User_Name": user_name,
                "User_Age": user_age,
                "User_Gender": user_gender,
                "User_Nationality": user_nationality,
                "User_Travel_Style": user_travel_style,
                "User_Persona": user_persona,
                "User_Likes": str(user_likes),
                "User_Dislikes": str(user_dislikes),
                "User_Price_Preference": user_price_preference,
                "User_Min_Rating": user_min_rating,
                "User_Input": user_input,
                "Relevant_Information": retrieved_chunks,
                "LLM_Response": response
            }
            
            results.append(result)
            checkpoint_csv_path = append_result_to_csv(result, model_name)
            completed_profile_keys.add(profile_key)
            print(f"Checkpoint saved: {checkpoint_csv_path}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60

            print(f"✓ Processed successfully!!! Duration: {duration:.2f} minutes")
            
        if results:
            csv_path = save_results_to_csv(results, model_name)
            print(f"Completed processing {model_name}. Results saved to: {csv_path}")
            print(f"Total records generated: {len(results)}")
        else:
            print(f"No new results generated for model: {model_name}")

if __name__ == "__main__":
    print("Script is being run")
    main()