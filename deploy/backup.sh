#!/usr/bin/env bash
# ==============================================================================
# Legal RAG System - Database Backup Script
# Usage: ./deploy/backup.sh
# Cron:  0 3 * * * cd /path/to/project && ./deploy/backup.sh >> /var/log/legalrag-backup.log 2>&1
# ==============================================================================
set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

if [ -z "${POSTGRES_URL:-}" ]; then
    echo "$(date -Iseconds) ERROR: POSTGRES_URL not set"
    exit 1
fi

# Configuration
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_FILE="$BACKUP_DIR/legalrag_${TIMESTAMP}.sql.gz"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "$(date -Iseconds) Starting backup..."

# Run pg_dump and compress
pg_dump "$POSTGRES_URL" \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    | gzip > "$BACKUP_FILE"

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "$(date -Iseconds) Backup created: $BACKUP_FILE ($BACKUP_SIZE)"

# Clean old backups
DELETED=$(find "$BACKUP_DIR" -name "legalrag_*.sql.gz" -mtime "+$RETENTION_DAYS" -delete -print | wc -l)
if [ "$DELETED" -gt 0 ]; then
    echo "$(date -Iseconds) Cleaned $DELETED backups older than $RETENTION_DAYS days"
fi

# Optional: Upload to S3-compatible storage
if [ -n "${S3_BACKUP_BUCKET:-}" ]; then
    if command -v aws &> /dev/null; then
        aws s3 cp "$BACKUP_FILE" "s3://$S3_BACKUP_BUCKET/legalrag/$(basename "$BACKUP_FILE")"
        echo "$(date -Iseconds) Uploaded to s3://$S3_BACKUP_BUCKET"
    else
        echo "$(date -Iseconds) WARN: aws CLI not found, skipping S3 upload"
    fi
fi

echo "$(date -Iseconds) Backup complete"
