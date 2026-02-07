#!/bin/bash
set -e

# Run migrations before starting app
echo "Running database migrations..."
flask db upgrade

echo "Starting application..."
exec "$@"