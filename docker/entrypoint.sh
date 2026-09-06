#!/usr/bin/env bash
# ===========================================================================
# Container start-up: bring the platform to a usable state, then serve it.
#
# Every step is idempotent and skipped when its output already exists, so the
# first run does the work (a few minutes) and later runs start in seconds.
# The whole point is that `docker-compose up` needs no follow-up commands.
# ===========================================================================
set -euo pipefail

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

cd /app

# ---------------------------------------------------------------------------
# 1. Wait for PostgreSQL. The compose healthcheck already gates this, but the
#    app is also expected to run against an external database.
# ---------------------------------------------------------------------------
if [ "${USE_SQLITE_FALLBACK:-false}" != "true" ]; then
    log "Waiting for PostgreSQL at ${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}"
    for attempt in $(seq 1 60); do
        if python -c "
import socket, sys
host, port = '${POSTGRES_HOST:-db}', int('${POSTGRES_PORT:-5432}')
sys.exit(0 if socket.create_connection((host, port), timeout=2) else 1)
" 2>/dev/null; then
            echo "    database reachable after ${attempt}s"
            break
        fi
        if [ "$attempt" -eq 60 ]; then
            warn "database did not become reachable; continuing anyway"
        fi
        sleep 1
    done
fi

# ---------------------------------------------------------------------------
# 2. Fixtures. Only needed in sample mode, and only if they are absent -- the
#    image ships them, so this is a safety net for a mounted-over data volume.
# ---------------------------------------------------------------------------
if [ "${DATA_MODE:-sample}" = "sample" ] && [ ! -f /app/data/sample/application_train.csv ]; then
    log "Generating synthetic sample fixtures"
    python -m src.data.generate_sample
fi

# ---------------------------------------------------------------------------
# 3. EDA artifacts -- the figures and insight index the Overview tab reads.
# ---------------------------------------------------------------------------
if [ ! -f /app/reports/eda_insights.json ]; then
    log "Running exploratory data analysis"
    python -m notebooks.eda > /dev/null
fi

# ---------------------------------------------------------------------------
# 4. Model. The bake-off is the slow step, so it runs only when no artifacts
#    exist. Mount ./models to keep the trained model across rebuilds.
# ---------------------------------------------------------------------------
if [ ! -f /app/models/model.joblib ]; then
    log "Training the model (three-way bake-off; this takes a few minutes)"
    python -m src.ml.train
    log "Evaluating"
    python -m src.ml.evaluate > /dev/null
    log "Deriving credit-policy rules"
    python -m src.xai.rules > /dev/null
else
    echo "    model artifacts present; skipping training"
fi

# ---------------------------------------------------------------------------
# 5. Analytics database for the talk-to-data feature. Reloaded on every start
#    so the tables always match the current model's predictions.
# ---------------------------------------------------------------------------
log "Loading the analytics database"
if ! python -m src.data.db_loader; then
    warn "database load failed -- the Chat tab will be unavailable, everything else works"
fi

# ---------------------------------------------------------------------------
# 6. Serve.
# ---------------------------------------------------------------------------
log "Starting Streamlit on port ${STREAMLIT_SERVER_PORT:-8501}"
exec streamlit run src/ui/app.py \
    --server.port="${STREAMLIT_SERVER_PORT:-8501}" \
    --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
    --server.headless=true \
    --browser.gatherUsageStats=false
