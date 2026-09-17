#!/bin/bash
# 加 METRICS_INTERNAL_TOKEN 并重启 backend,然后拉取 /metrics
cd ~/projectdocker
if ! grep -q '^METRICS_INTERNAL_TOKEN=' .env; then
  echo "METRICS_INTERNAL_TOKEN=rt-metrics-50x5read" >> .env
fi
docker compose up -d backend >/dev/null 2>&1
sleep 8
for i in 1 2 3 4 5 6; do
  H=$(curl -s --noproxy '*' -m 3 http://127.0.0.1:8001/health)
  [ -n "$H" ] && break
  sleep 3
done
echo "health: $H"
curl -s --noproxy '*' http://127.0.0.1:8001/metrics -H 'X-Internal-Token: rt-metrics-50x5read' > /tmp/metrics.json
python3 - <<'PYEOF'
import json
d = json.load(open('/tmp/metrics.json'))
print(json.dumps(d, ensure_ascii=False, indent=1)[:4000])
PYEOF
