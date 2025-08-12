import base64
import requests
import streamlit as st
import os
import json
import time
import pickle
import logging
import uuid
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration Variables ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER", "http://host.docker.internal:8000")
LOG_FILE = os.path.join("/app/logs", "streamlit_log.log")
MODEL = "qwen2.5vl:7b"  # Use the 7B model for better performance
TEMPERATURE = 0.0
REPEAT_PENALTY = 1.2
SYSTEM_PROMPT = """
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

# --- Logging Configuration ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# --- Load data from Pickle Files ---
@st.cache_resource
def load_pickle_files():
    """Load all necessary pickle files for the application."""
    
    with open("pc_to_item.pkl", "rb") as f:
        pc_to_item = pickle.load(f)
    
    with open("alias_to_pc.pkl", "rb") as f:
        alias_to_pc = pickle.load(f)
    
    with open("aliases.pkl", "rb") as f:
        aliases = pickle.load(f)
    
    with open("alias_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    return pc_to_item, alias_to_pc, aliases, embeddings

if "data_loaded" not in st.session_state:
    start_load_time = time.time()
    print("Loading pickle file data...", flush=True)
    
    PC_TO_ITEM, ALIAS_TO_PC, ALIASES, ALIAS_EMBEDDINGS = load_pickle_files()
    st.session_state.data_loaded = True  # Ensure data is loaded only once
    
    end_load_time = time.time()
    print(f"Data loaded in {end_load_time - start_load_time:.2f} seconds", flush=True)

# --- SQLite Database Configuration ---
def get_db_connection():
    conn = sqlite3.connect(os.path.join("/app/logs", "streamlit_db.db"))
    return conn

with get_db_connection() as conn: # Create the logs table if it doesn't exist
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS logs (request_id TEXT PRIMARY KEY, image BLOB, response TEXT, response_time REAL, feedback TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
    conn.commit()

# --- Logging Helper Functions ---
def log_response():
    state = st.session_state
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM logs WHERE request_id = ?", (state.request_id,))
        already_logged = cursor.fetchone() is not None

        if already_logged:
            return  # Skip logging if this request_id already exists

        cursor.execute(
            '''
            INSERT INTO logs (request_id, image, response, response_time)
            VALUES (?, ?, ?, ?)
            ''',
            (state.request_id, state.image, json.dumps(state.response), state.response_time)
        )
        conn.commit()
    
def log_feedback():
    if "feedback" not in st.session_state:
        st.session_state.feedback = None  # Ensure feedback is initialized

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE logs SET feedback = ? WHERE request_id = ?",
            (st.session_state.feedback, st.session_state.request_id)
        )
        conn.commit()

def log_types():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE logs SET types = ? WHERE request_id = ?",
            (json.dumps(st.session_state.saved_types), st.session_state.request_id)
        )
        conn.commit()


# --- Semantic Search for Item Types ---
def get_type_embedding(item_type: str) -> np.ndarray:
    """Get the embedding for a given item type using the model server."""
    payload = {"type": item_type}
    try:
        response = requests.post(f"{MODEL_SERVER_URL}/encode", json=payload)
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        np_embeddings = np.array(embeddings)
        return np_embeddings  # Return as a flat list
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to model server: {e}")
        return np.array([])

def find_closest_match(item_type):
    # Check if the item type is already saved in session state
    if "saved_types" not in st.session_state:
        st.session_state.saved_types = {}

    if item_type in st.session_state.saved_types:
        # If already saved, return the saved item
        saved_item = st.session_state.saved_types[item_type]
        return saved_item["cb_type"], saved_item["product_code"]
    
    item_embedding = get_type_embedding(item_type)

    # Compute cosine similarities
    similarities = cosine_similarity(item_embedding, ALIAS_EMBEDDINGS)[0]

    # Find the index of the most similar item
    closest_index = np.argmax(similarities)

    closest_alias = ALIASES[closest_index]
    closest_pc = ALIAS_TO_PC.get(closest_alias, "")
    closest_cb_item = PC_TO_ITEM.get(closest_pc, "")

    # Save the closest known item to session state
    st.session_state.saved_types[item_type] = {"cb_type": closest_cb_item, "product_code": closest_pc}
    # Return the closest known item
    return closest_cb_item, closest_pc

# --- Helper Function to Parse JSON Safely ---
def parse_json_output(text):
    print(text, flush=True)  # Debugging: print the raw output to logs
    try:
        # Sanitize markdown wrapping if present
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
        parsed_output = json.loads(text)
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Failed to parse JSON: {e}")
        return None
    
    # Remove any attribute from parsed_output that is empty, None, or "unknown"
    def clean_item(item):
        return {k: v for k, v in item.items() if v not in ("", None, "unknown", [], {})}

    if parsed_output and "items" in parsed_output:
        parsed_output["items"] = [clean_item(item) for item in parsed_output["items"]]
    return parsed_output

# --- Function to format JSON data as a bulleted string ---
def format_json_as_bullets(data, indent=0):
    """Recursively format JSON data as a bulleted string."""
    formatted = ""
    prefix = "" if indent == 0 else "  " * indent + "- "
    
    if isinstance(data, dict):
        for key, value in data.items():
            formatted += f"{prefix}{str(key).replace('_', ' ').title()}:\n"
            formatted += format_json_as_bullets(value, indent + 1)
    elif isinstance(data, list):
        for item in data:
            # formatted += format_json_as_bullets(item, indent + 1)
            formatted += format_json_as_bullets(item, indent)
    else:
        data_str = str(data)
        data_formatted = data_str.replace("_", " ")
        if data_formatted.lower().startswith("ip"):
            data_formatted = data_formatted.replace("p", "P")
        else:
            data_formatted = data_formatted[0].upper() + data_formatted[1:] if data_formatted else ""
            
        formatted += f"{prefix}{data_formatted}\n"

    return formatted

# --- Streamlit UI ---
st.set_page_config(page_title="Chargerback Vision Demo", layout="centered")
st.image("logo.png", width=250)
st.title("Chargerback® AI — Vision Demo")
st.text("Upload an image, and our AI-powered vision model will instantly analyze it, returning a detailed JSON description of the item(s) in the frame. This is a demonstration tool to showcase our capabilities.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an image of the lost item", type=["jpg", "jpeg", "png"])

# --- Inference Execution ---
if uploaded_file:
    # --- Reset session state if a new file is uploaded ---
    if "last_uploaded_filename" not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.clear()
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.request_id = str(uuid.uuid4())  # Generate a new request ID for this session

    image_bytes = uploaded_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    st.session_state.image = image_b64  # Save the base64 image in session state
    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

    # -- Initialize session state variables if not already set ---
    if "response" not in st.session_state:
        st.session_state.response = None
    if "response_time" not in st.session_state:
        st.session_state.response_time = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    # --- Handle button click for analysis ---
    if st.button("🔍 Analyze"):
        loading_placeholder = st.empty()
        loading_placeholder.info("⏳ Analyzing image...")

        payload = {
            "model": MODEL,
            "prompt": SYSTEM_PROMPT,
            "images": [image_b64],
            "options": {
                "temperature": TEMPERATURE,
                "repeat_penalty": REPEAT_PENALTY,
                "num_predict": 1024,
            },
            "format": "json",
            "stream": False
        }

        try:
            start_time = time.time()
            response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            response.raise_for_status()
            loading_placeholder.empty()
            end_time = time.time()

            # Parse the response and save results
            raw_output = response.json().get("response", "").strip()
            parsed_output = parse_json_output(raw_output)

            st.session_state.response = parsed_output
            st.session_state.response_time = round(end_time - start_time, 2)
            st.session_state.feedback = None  # Reset feedback on new analysis
            
            logging.info(f"Request ID: {st.session_state.request_id}, Response time: {st.session_state.response_time} seconds")

        except requests.exceptions.RequestException as req_err:
            st.error(f"Request failed: {req_err}")
            logging.error(f"Request failed: {req_err}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            logging.error(f"Unexpected error: {e}")

    # --- Display response and feedback section ---
    if st.session_state.response:
        st.success(f"Response received in {st.session_state.response_time} seconds.")
        st.subheader("Results:")

        if "items" in st.session_state.response:
            for item in st.session_state.response["items"]:
                item["cb_type"], item["product_code"] = find_closest_match(item.get("type", "unknown"))

                # Pretty print results as a bulleted outline
                col1, col2 = st.columns([0.4, 0.6])
                cb_type = item["cb_type"]
                if cb_type.lower().startswith("ip"):
                    cb_type_formatted = cb_type.replace("p", "P", 1)
                else:
                    cb_type_formatted = cb_type[:1].upper() + cb_type[1:] if cb_type else ""
                col1.markdown("##### Chargerback Type: :green[" + (cb_type_formatted if cb_type_formatted else "Unknown") + "]")
                col1.markdown("##### Product Code: :green[" + (str(item["product_code"]) if item["product_code"] else "Unknown") + "]")
                
                # Highlight known CB Attributes in the left column
                cb_attributes = {"brand", "amount", "currency_type", "color", "material", "case_color"}
                for attr in cb_attributes:
                    if attr in item:
                        value = item[attr]
                        
                        if value == 0 or value is None or value == "":
                            continue
                        
                        if isinstance(value, list):
                            value = ", ".join(value)
                        
                        if not isinstance(value, (int, float, np.number)):
                            value = value[0].upper() + value[1:] if value else "Unknown"

                        col1.markdown(f"##### {attr.replace('_', ' ').title()}: :green[{value}]")

                with col2:
                    item_no_cb = {k: v for k, v in item.items() if k not in ("cb_type", "product_code", "type") and k not in cb_attributes}
                    formatted_item = format_json_as_bullets(item_no_cb)
                    st.markdown(f"```\n{formatted_item}\n```")
                st.divider()
                
            log_response()

        # Feedback section (allows changing/removing feedback)
        sentiment_mapping = ["negative", "positive"]
        feedback_selected = st.feedback("thumbs", key="feedback_widget")
        if feedback_selected is not None:
            sentiment = sentiment_mapping[feedback_selected]
            st.session_state.feedback = sentiment 
        elif feedback_selected is None and "feedback" in st.session_state:
            st.session_state.feedback = None # Reset feedback if no thumbs selected
        
        log_feedback() # Update database record for this request_id with feedback

