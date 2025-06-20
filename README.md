# 🔍 Lost Item Image Analyzer

This project is a **Streamlit-based visual recognition tool** that uses local vision models (via [Ollama](https://ollama.com)) to analyze uploaded images of lost items and extract detailed, structured descriptions in JSON format.

---

## 🚀 Features

- Upload image files (`.jpg`, `.jpeg`, `.png`)
- Select from multiple vision models (e.g. `llava`, `bakllava`, `llava-phi3`, `llama3.2-vision:11b`)
- Uses a structured prompt to extract key visual attributes:
  - Item type
  - Brand or logo
  - Color(s)
  - Material
  - Distinctive features
  - Estimated size
  - Visible text or labels
- Displays response in both JSON and raw format (if parsing fails)
- Measures response time for each request

---

## 🧠 Example Output

```json
{
  "type": "Backpack",
  "brand": "Nike",
  "color": "Black with white logo",
  "material": "Nylon",
  "features": "Mesh side pockets, padded straps",
  "size": "Medium",
  "text": "NIKE"
}
```

---

## 🛠 Requirements

- Python 3.8+
- Ollama installed and running with a compatible vision model
- Docker (optional, for containerized setup)

---

## 📦 Installation (Local)

1. Clone the repository:

```bash
git clone https://github.com/your-username/lost-item-image-analyzer.git
cd lost-item-image-analyzer
```

2. Set up a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. Start Ollama and load a vision model:

```bash
ollama serve
ollama run llava
```

4. Run the app:

```bash
streamlit run app.py
```

---

## 🐳 Running in Docker

**Build and run the container** (assuming Ollama runs on host):

```bash
docker build -t lost-item-app .
docker run -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 lost-item-app
```

For Linux users, use:

```bash
docker run --network host -e OLLAMA_HOST=http://localhost:11434 lost-item-app
```

---

## 📁 Project Structure

```
lost-item-image-analyzer/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── Dockerfile           # (Optional) container build config
└── README.md            # This file
```

---

## 🧪 Vision Models Supported

- `llava`
- `bakllava`
- `llava-phi3`
- `llama3.2-vision:11b`

Ensure the model is pulled and available in Ollama before use.

---

## 📄 License

MIT License © 2025 [Your Name]

---

## 🙌 Acknowledgements

- [Ollama](https://ollama.com) for running local multimodal models
- [Streamlit](https://streamlit.io) for UI simplicity