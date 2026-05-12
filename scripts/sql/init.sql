-- Database initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create airflow DB
CREATE DATABASE airflow;

-- Create indexes for common queries (tables created by Alembic)
