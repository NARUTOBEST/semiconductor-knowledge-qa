#!/bin/bash
# 隧道 -L 改绑 0.0.0.0,容器经 host.docker.internal 才能访问 8002
pkill -f start_tunnel.sh 2>/dev/null
sleep 1
sed -i 's#-L 8002:127.0.0.1:8002#-L 0.0.0.0:8002:127.0.0.1:8002#' /tmp/start_tunnel.sh
grep -- '-L' /tmp/start_tunnel.sh
nohup bash /tmp/start_tunnel.sh >/dev/null 2>&1 &
sleep 5
echo "== listening:"
netstat -tln 2>/dev/null | grep 8002 || ss -tln | grep 8002
echo "== container -> retrieval health:"
docker exec semi-backend python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:8002/health',timeout=5).read()[:80])"
