import pickle
import streamlit as st
import time

class DataLoader:
    """Handles loading and caching of pickle data files."""
    
    @st.cache_resource
    def load_all_data(_self):
        """Load all necessary pickle files for the application."""
        start_time = time.time()
        print("Loading pickle file data...", flush=True)
        
        data = {}
        
        with open("data/pc_to_item.pkl", "rb") as f:
            data["PC_TO_ITEM"] = pickle.load(f)
        
        with open("data/alias_to_pc.pkl", "rb") as f:
            data["ALIAS_TO_PC"] = pickle.load(f)
        
        with open("data/aliases.pkl", "rb") as f:
            data["ALIASES"] = pickle.load(f)
        
        with open("data/alias_embeddings.pkl", "rb") as f:
            data["ALIAS_EMBEDDINGS"] = pickle.load(f)
        
        end_time = time.time()
        print(f"Data loaded in {end_time - start_time:.2f} seconds", flush=True)
        
        return data
