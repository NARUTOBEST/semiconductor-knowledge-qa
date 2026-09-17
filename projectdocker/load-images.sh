#!/bin/bash
# 把离线镜像 tar 导入本机 Docker(目标 Ubuntu 机执行,无需联网)。
set -e
cd "$(dirname "$0")/images"
for f in *.tar; do
  echo "[load] $f"
  docker load -i "$f"
done
echo "全部镜像导入完成。docker images 应可见 semi-backend/semi-web/qdrant/redis-stack/pgvector-postgres。"
