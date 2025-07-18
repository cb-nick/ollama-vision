#FROM python:3.11-slim
FROM python:3.10-slim
RUN pip install streamlit
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY st_lost_item_analyzer.py .
COPY logo.png .
COPY *.pkl .
RUN mkdir -p /app/logs
VOLUME /app/logs
EXPOSE 8502
# Enables docker logs to show output from python print statements
ENV PYTHONBUFFERED=1
CMD ["streamlit", "run", "st_lost_item_analyzer.py", "--server.port=8501", "--server.headless=true"]
