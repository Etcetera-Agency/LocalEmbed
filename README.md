# LocalEmbed

A fast and simple local API for generating text embeddings.

## Features
- Generate text embeddings locally without external API dependencies.
- List available embedding models.
- Containerized with Docker for easy deployment.

## Getting Started

### Prerequisites
- Python 3.11+ (for local development)
- Docker (optional, for containerized deployment)

### Running Locally

1. Install the dependencies:
   ```bash
   uv sync # or your preferred method based on pyproject.toml
   ```

2. Run the FastAPI development server:
   ```bash
   fastapi dev app/main.py
   ```
   The API will be available at `http://localhost:8000`.

### Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t localembed .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 localembed
   ```

## API Endpoints

- `GET /v1/health` — Health check
- `POST /v1/embeddings` — Generate text embeddings using local models (OpenAI API compatible)
- `GET /v1/models` — List supported and ready-to-use embedding models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.