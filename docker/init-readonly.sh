#!/bin/bash
# ===========================================================================
# PostgreSQL first-start initialisation: create the read-only role used by the
# talk-to-data feature.
#
# Runs once, from /docker-entrypoint-initdb.d/, before the application has
# created any tables. That ordering is why ALTER DEFAULT PRIVILEGES matters:
# a plain GRANT ON ALL TABLES would grant nothing, because no tables exist yet.
# The FOR ROLE clause is essential too -- default privileges apply to objects
# created by a specific role, and the app creates its tables as POSTGRES_USER,
# not as the superuser running this script.
#
# This is the second of three defences. The SQL validator refuses unsafe
# statements, this role cannot execute them even if one slipped through, and a
# server-side statement timeout bounds anything that is valid but pathological.
# ===========================================================================
set -euo pipefail

READONLY_USER="${POSTGRES_READONLY_USER:-credit_readonly}"
READONLY_PASSWORD="${POSTGRES_READONLY_PASSWORD:-readonly_pass_change_me}"

echo "==> Creating read-only role '${READONLY_USER}'"

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${READONLY_USER}') THEN
        CREATE ROLE ${READONLY_USER} LOGIN PASSWORD '${READONLY_PASSWORD}';
    END IF;
END
\$\$;

GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO ${READONLY_USER};
GRANT USAGE ON SCHEMA public TO ${READONLY_USER};

-- Anything that already exists...
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${READONLY_USER};

-- ...and anything ${POSTGRES_USER} creates later, which is every table, since
-- the application loads them after this script has finished.
ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
    GRANT SELECT ON TABLES TO ${READONLY_USER};

-- Withhold everything that writes, explicitly rather than by omission.
REVOKE CREATE ON SCHEMA public FROM ${READONLY_USER};
ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
    REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON TABLES FROM ${READONLY_USER};

-- A statement timeout on the role itself, so a pathological generated query
-- cannot hold a connection open indefinitely.
ALTER ROLE ${READONLY_USER} SET statement_timeout = '${SQL_TIMEOUT_SECONDS:-15}s';
SQL

echo "==> Read-only role ready"
