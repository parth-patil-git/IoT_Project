FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY  data_collector.py .
CMD ["pyhton", "-u", "data_collector.py"]