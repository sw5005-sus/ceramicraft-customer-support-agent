-- Ceramicraft Customer Support Agent - Database Initialization
-- This script runs automatically on first PostgreSQL container startup
-- (via /docker-entrypoint-initdb.d mount in docker-compose.yml).
-- It creates the cs_agent_db database for LangGraph conversation checkpoints.
-- Table schema is managed by AsyncPostgresSaver.setup() at application startup.

SELECT 'CREATE DATABASE cs_agent_db'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'cs_agent_db'
)\gexec
