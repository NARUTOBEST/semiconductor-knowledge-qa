#!/bin/bash
# ============================================================
# 恢复 Qdrant 知识库快照(ald_text 40 万块 + ald_image 14.9 万块)。
# 前提: docker compose up -d 且 qdrant 已启动;首次部署执行一次即可(幂等跳过已有集合)。
# ============================================================
set -e
cd "$(dirname "$0")"
Q="http://127.0.0.1:6333"

for name in ald_text ald_image; do
  file=$(ls snapshots/${name}-*.snapshot 2>/dev/null | head -1)
  if [ -z "$file" ]; then echo "未找到 $name 快照文件,跳过"; continue; fi
  if curl -s "$Q/collections/$name" | grep -q '"status":"ok"\|"result"'; then
    echo "[$name] 集合已存在,跳过恢复"
    continue
  fi
  echo "[$name] 上传并恢复 $file ..."
  curl -s -X POST "$Q/collections/$name/snapshots/upload?priority=snapshot" \
       -F "snapshot=@$file" >/dev/null
  echo "[$name] 恢复完成"
done

echo
echo "当前集合点数:"
for name in ald_text ald_image; do
  curl -s "$Q/collections/$name" | grep -o '"points_count":[0-9]*' || echo "$name: 不存在"
done
