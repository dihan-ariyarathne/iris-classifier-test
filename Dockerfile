# Use Python slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Train model (creates iris_model.pkl)
RUN python model.py

# Expose port
EXPOSE 8501

# Correct CMD with runtime variable expansion
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0