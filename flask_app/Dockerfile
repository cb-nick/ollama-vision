FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY new_lines_ollama.py.py .
EXPOSE 5002
CMD ["python", "new_lines_ollama.py.py"]
