#!/bin/bash
# 直连 HTTP /search_text 20 并发压测(绕过 MCP 层,定位串行点)
set -a; source /root/autodl-tmp/app/ret.env; set +a
Q=("ALD 工艺中 TMA 前驱体的饱和脉冲参数" "wafer chuck 温度控制原理" "plasma pretreatment 等离子预处理流程" "腔体压力控制与节气门调节" "removal chamber 清洗腔的维护周期" "RF 射频电源功率设置范围" "recipe 建立与编辑步骤" "prefix 配气系统的组成" "泄漏率检漏方法与验收标准" "氧气等离子清洗的工艺窗口" "远程等离子体源 RPS 的工作原理" "基片传输机器人手臂校准流程" "ALD 沉积速率的影响因素" "尾气处理与尾气洗涤器维护" "MFC 质量流量计校准步骤" "设备联锁 interlock 逻辑说明" "生长温度对薄膜均匀性的影响" "脉冲时间与饱和曲线的关系" "日常点检项目清单" "报警代码 E001-E050 含义")
t0=$(date +%s.%N)
for i in $(seq 0 59); do
  q="${Q[$((i % 20))]}"
  (
    s=$(date +%s.%N)
    curl -s -m 300 -X POST http://127.0.0.1:8002/search_text \
      -H "Content-Type: application/json" -H "X-Internal-Token: $RETRIEVAL_INTERNAL_TOKEN" \
      -d "{\"query\": \"$q\", \"k\": 4}" -o /dev/null
    e=$(date +%s.%N)
    echo "$e $s" | awk '{printf "%.1f\n", $1-$2}'
  ) &
done
wait
t1=$(date +%s.%N)
echo "wall: $(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')"
