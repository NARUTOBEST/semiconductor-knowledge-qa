#!/bin/bash
# cloudflared 看护:进程退出即重启,URL 写入 /c/project3/eval/_tunnel_url.txt
while true; do
  "/c/Users/33423/Downloads/cloudflared.exe" tunnel --url http://192.168.88.138:3000 --protocol http2 --no-autoupdate > /c/project3/eval/_tunnel.log 2>&1
  echo "$(date +%T) cloudflared exited, restarting..." >> /c/project3/eval/_tunnel_watchdog.log
  sleep 5
done
