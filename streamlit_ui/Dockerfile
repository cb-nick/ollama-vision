#FROM python:3.11-slim
FROM python:3.10-slim
RUN pip install streamlit
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY st_lost_item_analyzer.py .
EXPOSE 8502
CMD ["streamlit", "run", "st_lost_item_analyzer.py", "--server.port=8501", "--server.headless=true"]
