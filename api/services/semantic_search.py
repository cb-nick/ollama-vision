import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import AppConfig
from typing import Tuple, Dict

class SemanticSearchService:
    """Handles semantic search for item type matching."""
    
    def __init__(self, config: AppConfig, data: Dict):
        self.config = config
        self.saved_types = {}  # Cache for saved types
        self.aliases = data["ALIASES"]
        self.alias_embeddings = data["ALIAS_EMBEDDINGS"]
        self.alias_to_pc = data["ALIAS_TO_PC"]
        self.pc_to_item = data["PC_TO_ITEM"]
    
    def get_type_embedding(self, item_type: str) -> np.ndarray:
        """Get embedding for an item type using the model server."""
        payload = {"type": item_type}
        try:
            response = requests.post(f"{self.config.MODEL_SERVER_URL}/encode", json=payload)
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            return np.array(embeddings)
        except requests.exceptions.RequestException as e:
            return np.array([])
    
    def find_closest_match(self, item_type: str) -> Tuple[str, str]:
        """Find the closest matching item type in the database."""
        
        # Return cached result if available
        if item_type in self.saved_types: 
            cached = self.saved_types[item_type]
            return cached["cb_type"], cached["product_code"]
        
        # Get embedding and find closest match
        item_embedding = self.get_type_embedding(item_type)
        if item_embedding.size == 0:
            return "", ""
        
        similarities = cosine_similarity(
            item_embedding, 
            self.alias_embeddings
        )[0]

        closest_index = np.argmax(similarities)
        closest_score = similarities[closest_index]
        print(closest_score, flush=True)
        
        if closest_score > 0.6:  # Adjust threshold as needed
            closest_alias = self.aliases[closest_index]
            closest_pc = self.alias_to_pc.get(closest_alias, "")
            closest_cb_item = self.pc_to_item.get(closest_pc, "")

            # Cache the result
            self.saved_types[item_type] = {
                "cb_type": closest_cb_item,
                "product_code": closest_pc
            }
        else:
            closest_cb_item = "Not Listed"
            closest_pc = "217"

        return closest_cb_item, closest_pc
