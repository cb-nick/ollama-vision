import signal
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
import pickle

MODEL_SERVER_URL = "http://localhost:8000"

def on_exit(sig, frame):
    print("Stopping the model_server...")
    # Send a request to stop the model server gracefully
    os.system('pkill -f "uvicorn model_server:app" || true')
    sys.exit(0)

def load_pickle_files():
    """Load all necessary pickle files for the application."""
    base_dir = os.path.dirname(__file__)  # Get the directory of the current script
    
    with open(os.path.join(base_dir, "pc_to_item.pkl"), "rb") as f:
        pc_to_item = pickle.load(f)
    
    with open(os.path.join(base_dir, "alias_to_pc.pkl"), "rb") as f:
        alias_to_pc = pickle.load(f)
    
    with open(os.path.join(base_dir, "aliases.pkl"), "rb") as f:
        aliases = pickle.load(f)
    
    with open(os.path.join(base_dir, "alias_embeddings.pkl"), "rb") as f:
        embeddings = pickle.load(f)

    return pc_to_item, alias_to_pc, aliases, embeddings

def get_type_embedding(item_type):
    """Get the embedding for a given item type using the model server."""
    payload = {"type": item_type}
    try:
        response = requests.post(f"{MODEL_SERVER_URL}/encode", json=payload)
        response.raise_for_status()
        return response.json().get("embeddings", [])
    except requests.exceptions.RequestException as e:
        print(e)
        return []


def find_closest_matches(item_type):
    item_embedding = get_type_embedding(item_type)

    # Compute cosine similarities
    similarities = cosine_similarity(item_embedding, ALIAS_EMBEDDINGS)[0]
    
    # Get indices of the top 5 most similar items
    top_indices = np.argsort(similarities)[::-1][:5]
    results = []
    for idx in top_indices:
        closest_alias = ALIASES[idx]
        closest_pc = ALIAS_TO_PC.get(closest_alias, "")
        closest_cb_item = PC_TO_ITEM.get(closest_pc, "")
        score = float(similarities[idx])
        results.append({
            "closest_alias": closest_alias,
            "closest_type": closest_cb_item,
            "closest_pc": closest_pc,
            "score": score
        })

    # Return the list of top 5 matches
    return results

def print_results(results):
    # Prepare data for column formatting
    headers = ["Closest Alias", "Closest Chargerback Item", "Product Code", "Score"]
    rows = [
        [match["closest_alias"], match["closest_type"], match["closest_pc"], f"{match['score']:.4f}"]
        for match in results
    ]
    # Include headers for width calculation
    all_rows = [headers] + rows
    col_widths = [max(len(str(row[i])) for row in all_rows) for i in range(len(headers))]

    # Print header
    header_row = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_row)
    print("-" * len(header_row))

    # Print each result row
    for row in rows:
        print(" | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(len(row))))
        
    print("-" * len(header_row) + "\n")


signal.signal(signal.SIGINT, on_exit)

if __name__ == "__main__":
    # Load the model and data
    PC_TO_ITEM, ALIAS_TO_PC, ALIASES, ALIAS_EMBEDDINGS = load_pickle_files()
    
    # Main loop to accept user input
    while True:
        item_type = input("Enter an item type to search for (or 'q' to quit): ")
        if item_type.lower() == 'q':
            os.system('pkill -f "uvicorn model_server:app" || true')
            break
        if not item_type.strip():
            print("Please enter a valid item type.")
            continue
        
        print(f"\nSearching for closest matches to '{item_type}'...\n")
        results = find_closest_matches(item_type)

        print_results(results)
