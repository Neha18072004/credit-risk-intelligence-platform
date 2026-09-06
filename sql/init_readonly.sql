-- ===========================================================================
-- Read-only role for the talk-to-data feature.
--
-- Defence in depth. src/talk_to_data/sql_validator.py is the first line: it
-- parses every generated statement and rejects anything that is not a single
-- read-only SELECT. This role is the second: even if a statement somehow got
-- past the validator, the database itself has no permission to execute it.
--
-- Run as a superuser after the schema is created. docker-compose does this
-- automatically on first start.
-- ===========================================================================

-- The password is substituted from POSTGRES_READONLY_PASSWORD at container init.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = :'readonly_user') THEN
        EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L', :'readonly_user', :'readonly_password');
    END IF;
END
$$;

GRANT CONNECT ON DATABASE :"db_name" TO :"readonly_user";
GRANT USAGE ON SCHEMA public TO :"readonly_user";

-- SELECT and nothing else, on everything that exists now...
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"readonly_user";
-- ...and on anything created later.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO :"readonly_user";

-- Explicitly withhold everything that writes.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
    ON ALL TABLES IN SCHEMA public FROM :"readonly_user";
REVOKE CREATE ON SCHEMA public FROM :"readonly_user";
