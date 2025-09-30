import requests
import json
import base64
from config import AppConfig

class VisionService:
    """Handles vision analysis requests."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def analyze_image(self, image_bytes: bytes) -> dict:
        """Analyze an image using the vision model."""
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        payload = {
            "model": self.config.MODEL,
            "keep_alive": -1,
            "prompt": self.config.SYSTEM_PROMPT,
            "images": [image_b64],
            "options": {
                "temperature": self.config.TEMPERATURE,
                "repeat_penalty": self.config.REPEAT_PENALTY,
                "num_predict": 1024,
            },
            "format": "json",
            "stream": False
        }
        
        response = requests.post(f"{self.config.OLLAMA_HOST}/api/generate", json=payload)
        response.raise_for_status()
        
        raw_output = response.json().get("response", "").strip()
        return self._parse_json_output(raw_output)
    
    def _parse_json_output(self, text: str) -> dict:
        """Parse JSON output from the model."""
        print(text, flush=True)  # Debug logging
        
        try:
            # Sanitize markdown wrapping
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            elif text.startswith("```"):
                text = text.replace("```", "").strip()
            
            parsed_output = json.loads(text)
        except json.JSONDecodeError as e:
            return {"error": "Failed to parse JSON"}
        
        # Clean empty/unknown attributes
        if parsed_output:
            parsed_output = self._clean_item(parsed_output)

        return parsed_output
    
    def _clean_item(self, item: dict) -> dict:
        """Remove empty, None, or 'unknown' attributes from an item."""
        return {k: v for k, v in item.items() if v not in ("", None, "unknown", [], {})}
