FROM python:3.12-trixie

# Copy UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies and packages
# Set uv to:
#   - just install the lock file, and don't check against pyproject.toml
#   - not install dev dependencies
WORKDIR /app
ENV UV_NO_DEV=1 UV_FROZEN=1

COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

COPY src ./src
COPY api ./api
COPY models ./models
COPY README.md ./README.md
RUN uv sync

# Startup instructions
CMD ["uv", "run", "uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
