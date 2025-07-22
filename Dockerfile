 # Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && \
    apt-get install -y python3-pip ffmpeg && \
    apt-get clean

# Copy code
COPY . /app

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install --retries 10 --timeout 60 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Expose port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

