FROM python:3.11-slim

# Set working directory
RUN mkdir -p /app/
WORKDIR /app/
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt 

# Expose the application port
EXPOSE 8000

# Run the application
CMD python /app/scripts/loadData.py && \
    uvicorn main:app --host 0.0.0.0 --port 8000
