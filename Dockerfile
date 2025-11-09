# Dockerfile for Housing Regression MLE backend

# Use slim python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first (better caching)
# Copy the pyproject.toml and uv.lock files and save them in the working directory
COPY pyproject.toml uv.lock* ./ 

# Install uv (dependency manager)
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy the rest of the application code/project files
COPY . .

# Expose FastAPI deafult port 8000
EXPOSE 8000

# Command to run the FastAPI app with uvicorn
CMD [ "uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
