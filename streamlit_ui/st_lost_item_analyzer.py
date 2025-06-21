import base64
import requests
import streamlit as st
import os
import json
import time

# --- Configuration ---
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
VISION_MODELS = ["llava", "bakllava", "llava-phi3", "qwen2.5vl:3b", "minicpm-v:8b", "llama3.2-vision:11b"]

# --- Streamlit UI ---
st.set_page_config(page_title="Lost Item Image Analyzer", layout="centered")
st.title("🔍 Lost Item Image Analyzer")

# --- Sidebar Controls ---
with st.sidebar:
    model = st.selectbox("Choose Vision Model", VISION_MODELS, index=0)
    prompt = st.text_area("Prompt", """
You are an expert in visual recognition. Analyze the uploaded image of a lost item and extract detailed, structured information including:

- Type of item
- Brand or logo (if visible)
- Color(s)
- Material (e.g. leather, plastic, fabric)
- Distinctive features (e.g. scratches, patterns, accessories)
- Estimated size or dimensions
- Any text or labels visible

Respond in structured JSON format with fields:
{
  "type": "...",
  "brand": "...",
  "color": "...",
  "material": "...",
  "features": "...",
  "size": "...",
  "text": "..."
}
""".strip(), height=300)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload an image of the lost item", type=["jpg", "jpeg", "png"])

# --- Helper Function to Parse JSON Safely ---
def parse_json_output(text):
    try:
        # Sanitize markdown wrapping if present
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Failed to parse JSON: {e}")
        return None

# --- Inference Execution ---
if uploaded_file and st.button("🔍 Analyze"):
    image_bytes = uploaded_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }

    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
    st.write("⏳ Analyzing image...")

    try:
        start_time = time.time()
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        response.raise_for_status()
        end_time = time.time()

        raw_output = response.json().get("response", "").strip()
        parsed_output = parse_json_output(raw_output)

        st.success(f"Response received in {round(end_time - start_time, 2)} seconds.")
        st.subheader("Model Output")

        if parsed_output:
            st.json(parsed_output)
        else:
            st.code(raw_output, language="json")

    except requests.exceptions.RequestException as req_err:
        st.error(f"Request failed: {req_err}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
