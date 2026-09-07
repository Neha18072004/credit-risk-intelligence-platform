# ===========================================================================
# Credit Risk Intelligence Platform -- application image
#
# Python 3.11 rather than a newer release: LightGBM, CatBoost and SHAP all
# publish mature wheels for 3.11, so the build needs no compiler toolchain and
# is reproducible across architectures.
# ===========================================================================
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="Credit Risk Intelligence Platform" \
      org.opencontainers.image.description="Explainable credit-risk scoring with talk-to-data" \
      org.opencontainers.image.source="https://github.com/"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RUNNING_IN_DOCKER=true \
    MPLBACKEND=Agg

# libgomp1 is LightGBM's OpenMP runtime -- the wheel links against it and fails
# at import without it. curl is used by the container healthcheck.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are copied and installed first, so editing source code does not
# invalidate the (slow) dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY sql/ ./sql/
COPY notebooks/ ./notebooks/
COPY docker/ ./docker/
COPY data/sample/ ./data/sample/
# The pre-trained model and the generated reports ship with the image, so the
# container serves the real 307k-row model immediately instead of spending
# minutes retraining on synthetic fixtures at first start.
COPY models/ ./models/
COPY reports/ ./reports/

# Writable output directories. Created explicitly so they exist even when no
# volume is mounted over them.
RUN mkdir -p /app/models /app/reports/figures /app/data \
    && chmod +x /app/docker/entrypoint.sh \
    # Strip bytecode and test suites from site-packages: they are never
    # executed in a container and cost a couple of hundred megabytes.
    && find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d \
         -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.11/site-packages -name "tests" -type d \
         -maxdepth 2 -exec rm -rf {} + 2>/dev/null || true

# Run as a non-root user. The app never needs write access outside these paths.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
