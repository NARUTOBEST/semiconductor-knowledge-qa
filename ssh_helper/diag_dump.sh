#!/bin/bash
# 诊断:每 30s 向 backend 全部 uvicorn worker(spawn_main,宿主 pid)发 SIGUSR1,
# faulthandler 全线程栈 -> 容器 stderr(docker logs)。宿主侧直接 kill,免容器内引号地狱。
PW=123
for i in $(seq 1 100); do
  echo "== DUMP $i $(date +%T)" >> /tmp/diag_dump.log
  echo $PW | sudo -S docker top semi-backend -o pid,args 2>/dev/null \
    | awk '/spawn_main/ && !/awk/ {print $1}' > /tmp/worker_pids.txt
  while read -r p; do
    [ -n "$p" ] && sudo kill -USR1 "$p"
  done < /tmp/worker_pids.txt
  sleep 30
done
