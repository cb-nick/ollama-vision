import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration settings."""
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER", "http://host.docker.internal:8000")
    LOG_FILE: str = os.path.join("/app/logs", "streamlit_log.log")
    DB_FILE: str = os.path.join("/app/logs", "streamlit_db.db")
    MODEL: str = "qwen2.5vl:7b"
    TEMPERATURE: float = 0.0
    REPEAT_PENALTY: float = 1.2
    
    @property
    def SYSTEM_PROMPT(self) -> str:
        """Return the system prompt for vision analysis."""
        return """
You are an expert in visual recognition. Analyze the uploaded image of lost item(s) and extract detailed, structured information for each item visible in the image.

### Analysis Guidelines:
- Only describe what is clearly visible in the image
- Omit any attributes that cannot be identified in the image
- Do not infer or assume details not directly observable
- Only include attributes that make sense for the given item type 
     - (for example, if the item type is "shirt" there is no need for an attribute "serial_number")

### Rules for classifying smartphones:
- If the item is a smartphone, YOU MUST decide if it looks more like an iPhone or an Android phone based on the visible features.
- If it looks more like an iPhone, use "iPhone" as the type.
- If it looks more like an Android phone, use "Android" as the type.
- The type should be either "iPhone" or "Android", not just "smartphone".

### Rules for classifying currency:
- If the item is currency, identify the amount and the currency type (e.g., USD, EUR) based on the visible features.
- The amount should be a numeric value, and the currency type should be a valid currency code.
- Only extract the amount and currency type if they are clearly visible and unambiguous in the image.

### Item Counting Rules:
- **Treat pairs as single items**: Items that naturally come in pairs (shoes, socks, gloves, earrings, etc.) should be counted and described as ONE item, not two separate items
- **Example**: A pair of sneakers = 1 item of type "shoes" or "sneakers"
- **Example**: A pair of socks = 1 item of type "socks"

### Brand Attribution Rules:
- **DO NOT include brand information unless you are absolutely certain**
- Only include brand if:
    - The brand name/logo is clearly visible and legible in the image
    - The brand marking is prominent and unambiguous
    - You can definitively identify the brand without guessing
- **When in doubt, omit the brand attribute entirely**
- Avoid making brand assumptions based on style, design, or partial logos

### For each item, extract:
- **Type of item** (required)
- Brand or logo (only if clearly visible)
- Color(s) (only if clearly visible)
- Material (only if clearly visible)
- Distinctive features (only if clearly visible)
    - Limit to just a few of the most important distinctive features
- Text or labels that are legible (only if clearly visible)
    - Limit to just a few of the most important text or labels
- Any other attributes that are clearly visible and useful for identifying the item

### JSON Response Format:
```json
{
  "item_count": number,
  "items": [
    {
      "type": "...",
      "brand": "...",
      "color": "...",
      "material": "...",
      "amount": "...",
      "currency_type": "...",
      "distinctive_features": ["feature1", "feature2"] | [],
      ...,
    }
  ]
}
```

## Important:
- If no items are clearly identifiable, return item_count: 0
- Include only features that are directly visible, not assumed or inferred
- Remember: pairs of items = 1 item in your count and description
- When uncertain about brand, leave the brand field empty or omit it entirely
        """
