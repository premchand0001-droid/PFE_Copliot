FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pfe_copilot.py app.py ./
COPY data ./data
ENV PFE_OUT_DIR=/app/outputs
EXPOSE 8501
# pass LLM_BASE_URL / LLM_API_KEY / LLM_MODEL at runtime:  docker run -p 8501:8501 -e LLM_API_KEY=... -e LLM_BASE_URL=... pfe-copilot
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
