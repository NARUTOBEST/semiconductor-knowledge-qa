#!/bin/bash
# 全链路复测:登录 eval1 → 提问 1416 报警题 → 统计 sources 事件
TOKEN=$(curl -s --noproxy '*' -X POST http://127.0.0.1:3000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"eval1","password":"Eval#pass1"}' | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))')
if [ -z "$TOKEN" ]; then
  TOKEN=$(curl -s --noproxy '*' -X POST http://127.0.0.1:3000/api/auth/register \
    -H 'Content-Type: application/json' \
    -d '{"username":"eval1","password":"Eval#pass1"}' | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))')
fi
echo "token: ${TOKEN:0:12}***"
curl -s --noproxy '*' -N -m 120 -X POST http://127.0.0.1:3000/api/chat \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"BESI Datacon 2200 固晶机报 1416 报警是什么意思","history":[]}' > /tmp/sse_out.txt
echo "--- 事件统计:"
grep -c '"type":"sources"' /tmp/sse_out.txt
echo "--- tier:"
grep -o '"tier":"[a-z]*"' /tmp/sse_out.txt | head -2
echo "--- sources 条数:"
python3 - <<'EOF'
import json
srcs = []
for line in open('/tmp/sse_out.txt', encoding='utf-8'):
    if line.startswith('data: '):
        try: ev = json.loads(line[6:])
        except Exception: continue
        if ev.get('type') == 'sources': srcs.extend(ev.get('items', []))
        if ev.get('type') == 'assistant_message': ans = ev.get('content','')
print("sources:", len(srcs))
print("answer 前200字:", ans[:200])
EOF
