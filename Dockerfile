FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY config.yaml .

# Mount your own corpus and index at runtime, e.g.:
#   docker run --rm -e ANTHROPIC_API_KEY -v $(pwd)/data:/app/data rag-engine ingest --input data/raw
VOLUME ["/app/data"]

ENTRYPOINT ["rag"]
CMD ["--help"]
