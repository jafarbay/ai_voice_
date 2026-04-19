#!/usr/bin/env bash
# Smoke-тест для запущенного сервера:
#   ./scripts/quick_test.sh [server_url] [audio_file]
# По умолчанию: http://127.0.0.1:8000 и tests/fixtures/sample.wav
set -e

SERVER=${1:-http://127.0.0.1:8000}
FILE=${2:-tests/fixtures/sample.wav}

if [[ ! -f "$FILE" ]]; then
  echo "File not found: $FILE" >&2
  exit 1
fi

echo "POST $FILE -> $SERVER/jobs"
JOB=$(curl -s -X POST -F "file=@$FILE" "$SERVER/jobs" \
  | python -c "import sys,json;print(json.load(sys.stdin)['job_id'])")
echo "  job_id=$JOB"

echo "Polling status..."
while :; do
  STATUS=$(curl -s "$SERVER/jobs/$JOB" \
    | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "  $STATUS"
  if [[ "$STATUS" == "done" || "$STATUS" == "failed" ]]; then
    break
  fi
  sleep 1
done

echo "--- transcript redacted ---"
curl -s "$SERVER/jobs/$JOB/transcript?version=redacted" | python -m json.tool

echo "--- events ---"
curl -s "$SERVER/jobs/$JOB/events" | python -m json.tool

echo "--- audio ---"
OUT=${TMPDIR:-/tmp}/redacted_${JOB}.wav
curl -s -o "$OUT" "$SERVER/jobs/$JOB/audio?version=redacted"
ls -la "$OUT"

echo "Done. job_id=$JOB"
