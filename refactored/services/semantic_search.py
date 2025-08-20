import requests
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from config import AppConfig
from typing import Tuple

class SemanticSearchService:
    """Handles semantic search for item type matching."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def get_type_embedding(self, item_type: str) -> np.ndarray:
        """Get embedding for an item type using the model server."""
        payload = {"type": item_type}
        try:
            response = requests.post(f"{self.config.MODEL_SERVER_URL}/encode", json=payload)
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            return np.array(embeddings)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to model server: {e}")
            return np.array([])
    
    def find_closest_match(self, item_type: str) -> Tuple[str, str]:
        """Find the closest matching item type in the database."""
        # Initialize saved types if not exists
        if "saved_types" not in st.session_state:
            st.session_state.saved_types = {}
        
        # Return cached result if available
        if item_type in st.session_state.saved_types:
            cached = st.session_state.saved_types[item_type]
            return cached["cb_type"], cached["product_code"]
        
        # Get embedding and find closest match
        item_embedding = self.get_type_embedding(item_type)
        if item_embedding.size == 0:
            return "", ""
        
        similarities = cosine_similarity(
            item_embedding, 
            st.session_state.ALIAS_EMBEDDINGS
        )[0]
        
        closest_index = np.argmax(similarities)
        closest_alias = st.session_state.ALIASES[closest_index]
        closest_pc = st.session_state.ALIAS_TO_PC.get(closest_alias, "")
        closest_cb_item = st.session_state.PC_TO_ITEM.get(closest_pc, "")
        
        # Cache the result
        st.session_state.saved_types[item_type] = {
            "cb_type": closest_cb_item,
            "product_code": closest_pc
        }
        
        return closest_cb_item, closest_pc
