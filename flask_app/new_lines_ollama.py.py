from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import time

app = Flask(__name__)
CORS(app)

# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
# OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")


@app.route('/parse', methods=['POST'])
def parse_report():
    data = request.get_json()
    report = data.get("report", "")
    model = data.get("model", "llava")
    lines = report.strip().splitlines()
    results = []
    start = time.time()

    for line in lines:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": line, "stream": False}
        )
        result = resp.json()
        results.append({"input": line, "parsed": result.get("response", "")})

    end = time.time()
    return jsonify({
        "results_by_line": results,
        "processing_time_seconds": round(end - start, 2)
    })
#llama3.2-vision:11b
@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({"models": ["llava", "bakllava", "llava-phi3", "qwen2.5vl:3b", "minicpm-v:8b", "llama3.2-vision:11b"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
