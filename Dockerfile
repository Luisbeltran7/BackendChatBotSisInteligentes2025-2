FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps (if needed) and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
	&& rm -rf /var/lib/apt/lists/* \
	&& python -m pip install --upgrade pip setuptools wheel

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY ./src /app/src


EXPOSE 8000

# Default command - expects an ASGI app at src.main:app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
