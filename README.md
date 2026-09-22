# 半导体设备知识库智能问答系统

> **关于仓库历史的说明**：本项目因个人原因疏忽，忘记在项目刚开始开发时建立仓库，也忘记在后续开发过程中于 GitHub 上 commit 每一次修改，因此提交历史集中且不能反映真实开发节奏。**但本项目以人格担保是从 0 到 1 独立开发的**，全部设计、编码、部署与评测均由本人完成。评测脚本与题集已随仓库开源，所有指标可复现（见[评测](#-评测体系与指标)）。

面向半导体产线工程师的领域知识问答助手：覆盖设备手册、SEMI 标准、行业研报、CAD 图纸/演示视频等约 **2.4 万份**异构资料（**55 万**知识块），解决工程师定位报警码与工艺参数耗时长、通用大模型编造参数导致现场误用的问题。答案带出处页码、逐句可核查。

## ✨ 核心特性

- **三级范式路由**：simple / raglite / react 按问题复杂度自动分流 + 质检门低置信自动升级重答，用户无感
- **防幻觉闸门**：判定"有支撑"必须给出可字符串验证的原文引文，裁判造假即失败；低置信整段不出后端，降级为翻阅指引
- **混合检索**：BGE-m3 稠密 + 稀疏 + RRF 融合，章节路径注入重排（重排池保底 32，经 1000 题 × 10 配置免 LLM 扫描调优），低召回自动改写重查；报警码精确召回兜底、稀疏保底救表格块
- **图文双模态**：2.1 万 CAD 零件/装配自动渲染入库，跨模态检索
- **记忆系统**：后台异步记忆图 + 升迁门，会话记忆不阻塞回答
- **高并发流式链路**：SSE 流式输出、看门狗断流恢复、空答案跨模型救援、双层限流
- **全链路故障韧性**：熔断降级 + 本地 WAL/spool 兜底，恢复后幂等回填；崩溃安全的上下文压缩（先落盘再删除，重跑幂等）

## 🏗️ 架构

```
浏览器 ── cloudflared 公网隧道 ──> VM (Docker Compose 五容器)
                                    ├── web        (Next.js :3000)
                                    ├── backend    (FastAPI :8001)
                                    ├── qdrant / redis / postgres
                                    └── SSH 隧道 ──> AutoDL GPU 服务器
                                                      ├── vLLM 推理 :8000/:8001（主模型 Qwen3-32B-FP8 + 副模型 Qwen3-14B-FP8，均稠密）──> LiteLLM 网关 :4000（main/light 别名，云端 API 仅兜底）
                                                      └── 检索微服务 :8002（BGE-m3 / Reranker / CLIP）
```

- 三机分工：LLM 与向量检索在 GPU，业务后端与存储在 VM，公网入口经隧道
- 检索微服务与业务后端解耦，通过 MCP 协议 + 内部令牌鉴权互访

## 📊 评测体系与指标

自建 RAGAS 口径评测线束：20 并发 × 50 题真实 HTTP（SSE）全量回归、题级断点续跑、GPU 本地 LLM-as-Judge（Qwen3-32B 裁判，零云成本、无限流）+ 确定性检索硬指标（MRR / NDCG@K / Rfact@K / P_ret / R_pool），与基线自动对比，可一键复跑：

```bash
eval\pipeline\pipeline_50x5\regression_50x5.bat          # 全量 20 用户 × 50 题（1000 条记录）
eval\pipeline\pipeline_50x5\fetch_pools.py               # 离线补查 1000 题重排前候选池（R_pool 分母）
eval\pipeline\pipeline_50x5\concurrent_metrics.py        # 检索硬指标计算（--pools 池内召回）
eval\pipeline\pipeline_50x5\judge_50x5.py                # GPU LLM-as-Judge（忠实度/幻觉率/上下文召回/切题/答案准确）
eval\pipeline\pipeline_50x5\sweep_retrieval.py           # 免 LLM 参数扫描（/debug_cfg 热调参，1000 题 × 10 配置）
eval\pipeline\pipeline_full100\run_eval.py               # 全量 100 题白盒评测
```

**检索调优方法**：直调检索服务 `/search_text`（不经 LLM）扫 RERANK_POOL_MIN{8,16,24,32,48} × score_ratio{0.4,0} × 截断模式（τ+断崖 / 固定 K），真值分母统一取 48 候选全集，选定 **RERANK_POOL_MIN=32**（MRR +2.5pt、NDCG@5 +4.6pt、R@10 +5.8pt，P_ret 仅 -1.7pt）；扫描还证明 τ/MIN_K/断崖/固定 K 变体收益均 <1pt，池深是唯一显著杠杆。

**最新一轮实测（2026-09-21，20 并发 × 50 题 = 1000 条记录全量完成，内网全链路：VM Docker → SSH 隧道 → GPU vLLM/检索微服务）**：

| 类别 | 指标 | 数值（对比 09-20 基线） |
|---|---|---|
| 运行 | 完成率 / 错误率 | **100% / 0%**（1000/1000，基线 98.3% / 1.7%） |
| 检索 | MRR / NDCG@3 / NDCG@5 | **0.7452 / 0.7112 / 0.7584**（基线 0.7289 / 0.6973 / 0.7361） |
| 检索 | Rfact@5 / Rfact@10（要点级召回） | **0.7919 / 0.7919**（基线同口径 0.7519，**+4.0pt**） |
| 质量 | 忠实度 Faithfulness（LLM 裁判） | **0.9849**（基线 0.9821） |
| 质量 | 幻觉率 | **1.99%**（17/854 例，逐题可复核；基线 2.56%） |
| 质量 | 上下文召回 Context Recall | **0.9567** |
| 质量 | 答案切题度 Answer Relevancy | **0.9604** |
| 质量 | 答案准确度 Answer Accuracy | 0.8554 |
| 路由 | tier 路由准确率 | **93.4%**（基线 90.3%） |

> 口径说明：**Rfact@K** = 金标要点的证据块进入返回前 K 名的要点占比（分母=要点数，跨版本可比，是召回能力的真实口径）；块级 R@K 的分母为金标文档在候选全集中的全部命中切块（均值 ~14 块/题），受返回容量（MAX_K=8）结构性封顶（上限 ≈ 8/14 ≈ 0.56），故不作为召回主指标。Rfact@K 本轮为评测记录内 sources 文本口径；上一轮表中 0.8481 为 Qdrant 全文口径，同口径基线为 0.7519。

完整产物见 [`eval/runs/50x5_20260921_1000_v2/`](eval/runs/50x5_20260921_1000_v2/)（本轮）与 [`eval/runs/50x5_20260920_poolmin32/`](eval/runs/50x5_20260920_poolmin32/)（上轮，REPORT.md 含 10 配置扫描全表）。

## 🚀 从 0 到 1 保姆级搭建教程

> 所有命令均与仓库脚本一一对应,可逐条执行。整体分 6 个阶段:
>
> | 阶段 | 内容 | 机器 | 花费 |
> |---|---|---|---|
> | 0 | 前置准备(账号/本地环境/密钥文件) | Windows 本机 | 0 |
> | 1 | **无卡实例下载模型权重 → autodl-fs 网盘** | AutoDL 无卡机 | ¥0.1/h |
> | 2 | GPU 实例开机,起 vLLM×2 + LiteLLM + 检索微服务 | AutoDL GPU 机 | ¥6.98/h(RTX PRO 6000 96G) |
> | 3 | VM Docker 五容器(业务前后端 + 存储 + 知识库) | 内网 Ubuntu VM | 0 |
> | 4 | SSH 三向隧道(VM ↔ GPU) | VM | 0 |
> | 5 | 端到端验证 | 浏览器 | 0 |
> | 6 | 评测复跑(20×50 全量) | Windows + GPU | GPU 时长 |
>
> 三机分工:**LLM 与向量检索在 GPU 机,业务后端与存储在 VM,Windows 只做开发/评测客户端**。

### 阶段 0:前置准备

**0.1 账号**

- **AutoDL**(autodl.com):充值(余额紧张时按需充,RTX PRO 6000 约 ¥6.98/h、无卡模式 ¥0.1/h;**余额耗尽会自动关机**,压测中途会大面积 Connection error,务必留足)。
- **火山方舟**(可选,云端兜底):拿一个 OpenAI 兼容 key + 两个推理接入点(主答 + 兜底),配到 `env/env.env` 的 `OPENAI_API_KEY/OPENAI_TEXT_MODEL/OPENAI_FALLBACK_MODEL`,以及视觉描述模型 `OPENAI_VISION_*`(Qwen3 无视觉,入库图片描述走它)。
- **阿里云 OSS**(可选,仅当用 OSS 中转权重):见 1.5 备选方案。

**0.2 Windows 本机环境**

- Git Bash(所有部署脚本在项目根用 `bash` 执行)。
- Python 3.10+ 且装了 `paramiko`(`pip install paramiko`,ssh_push/ssh_run/push_big 都依赖)。
- 若本机有系统代理(如 Clash 7897):**访问内网/本机地址必须设 NO_PROXY**,否则 httpx 会被注册表系统代理劫持,连 127.0.0.1 都出网:

```bash
export NO_PROXY="192.168.88.138,127.0.0.1,localhost"
```

**0.3 本地密钥文件(全部被 .gitignore 排除,不推送)**

在项目根创建以下 5 个文件(字段模板,值自己填;**任何密钥不进 git**):

```
ssh_helper/gpu_ssh.env      # AutoDL GPU 实例 SSH:HOST=connect.westd.seetacloud.com / PORT=xxxxx / USER=root / PASS=xxx
ssh_helper/vm_ssh.env       # 内网 VM SSH:HOST=192.168.88.138 / PORT=22 / USER=lly / PASS=xxx
ssh_helper/gpu.env          # LITELLM_MASTER_KEY=sk-xxxx(自造一串,与 env/env.env 同串) / CLOUD_API_KEY=火山方舟key
ssh_helper/ret.env          # QDRANT_URL=http://127.0.0.1:6333 / RETRIEVAL_INTERNAL_TOKEN=rt-xxxx /
                            # EMBED_DEVICE=cuda / RETRIEVAL_HOST=127.0.0.1 / HF_HUB_OFFLINE=1 / RETRIEVAL_DEBUG_CFG=1
env/env.env                 # OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_TEXT_MODEL / OPENAI_FALLBACK_MODEL /
                            # OPENAI_VISION_* / TOS_*(可选)/ RETRIEVAL_INTERNAL_TOKEN(与 ret.env 同串) /
                            # LITELLM_MASTER_KEY(与 gpu.env 同串)
```

三个"同串"对齐关系(配错就是 401/403):
- `gpu.env 的 LITELLM_MASTER_KEY` = `env/env.env 的 LITELLM_MASTER_KEY`(业务调网关的 key);
- `ret.env 的 RETRIEVAL_INTERNAL_TOKEN` = `env/env.env 的 RETRIEVAL_INTERNAL_TOKEN`(backend 调检索的令牌);
- `gpu.env 的 CLOUD_API_KEY` = 火山方舟 key(LiteLLM 云端兜底用)。

### 阶段 1:AutoDL 无卡机下载权重 → autodl-fs(重点)

**为什么用无卡机**:模型权重合计约 50GB,用 GPU 机(¥6.98/h)下载纯烧钱;无卡模式 ¥0.1/h,有 CPU/内存/网络,没有 GPU,完全够做下载机。**权重必须放 `/root/autodl-fs/`(网盘)**——`/root/autodl-tmp/`(数据盘)在实例释放后被清空,而 autodl-fs 换同地区实例自动挂载,一次下载处处可用。

**1.1 租实例并开无卡模式**

1. AutoDL 控制台 → 算力市场 → 选 **西北B区** 的实例(如 RTX PRO 6000 96G)。**地区一旦选定,后续 GPU 实例、autodl-fs 网盘都必须同地区**,否则网盘不挂载。
2. 创建实例时镜像先随意(如 PyTorch 基础镜像),或直接选社区镜像 `rag-env-vllm029-cu130-qwen3`(阶段 2 可少装环境)。
3. 实例列表 → 该实例「更多」→ **「无卡模式开机」**(¥0.1/h)。无卡模式下不能跑 vLLM/检索,只能下载传输。
4. 复制控制台的 **SSH 登录指令**(形如 `ssh -p 12345 root@connect.westd.seetacloud.com`),密码在控制台重置/复制。把 HOST/PORT/USER/PASS 填进 `ssh_helper/gpu_ssh.env`。

**1.2 登录无卡机**

控制台打开 **JupyterLab → Terminal**,或本地 Git Bash 直接 `ssh -p <PORT> root@<HOST>`。

**1.3 ModelScope 下载 5 套权重(推荐,国内源快)**

在无卡机终端执行(**目录名约定很重要**,阶段 2 的 start_gpu.sh 按通配符自动链接这些目录):

```bash
mkdir -p /root/autodl-fs/models
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

FS=/root/autodl-fs/models

# ---- LLM 两个(vLLM 直读 FP8,无需转换)----
modelscope download --model Qwen/Qwen3-32B-FP8 --local_dir $FS/Qwen3-32B-FP8
modelscope download --model Qwen/Qwen3-14B-FP8 --local_dir $FS/Qwen3-14B-FP8

# ---- 检索三件套(按 RAG/embed.py 期望的 <owner>__<repo> 命名)----
modelscope download --model BAAI/bge-m3 --local_dir $FS/BAAI__bge-m3 \
  --exclude "*.onnx" "onnx/*" "*.ot" "*.msgpack" "*.DS_Store" "imgs/*"
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir $FS/BAAI__bge-reranker-v2-m3
modelscope download --model sentence-transformers/clip-ViT-B-32 --local_dir $FS/sentence-transformers__clip-ViT-B-32
# CLIP 若官方镜像缺失,用 fork 兜底:
# modelscope download --model AI-ModelScope/clip-ViT-B-32 --local_dir $FS/sentence-transformers__clip-ViT-B-32
```

要点:
- **断点续传**:下载中断直接重跑同一条命令,已下分片自动跳过。
- 长任务建议 nohup 挂后台防 SSH 断线:`nohup bash -c 'modelscope download ...' > dl.log 2>&1 &`,进度看 `tail -f dl.log`。
- 下载完每套目录放一个 `.download_ok` 空文件做标记(可选):`touch $FS/Qwen3-32B-FP8/.download_ok`。

**1.4 校验**

```bash
du -sh /root/autodl-fs/models/*/*
ls /root/autodl-fs/models/Qwen3-32B-FP8/*.safetensors | wc -l
```

预期:Qwen3-32B-FP8 = **7 个分片,约 32GiB**;Qwen3-14B-FP8 = **4 个分片,约 15.2GiB**;bge-m3 ≈ 2.3G;reranker-v2-m3 ≈ 2.3G;clip-ViT-B-32 ≈ 0.6G。

**1.5 备选传输方案(本地/第三方已有权重时)**

- **AutoDL 网盘中转**(小批量):控制台「文件存储」网页上传到网盘 → 实例内出现在 `/root/autodl-fs/`。大文件上传慢,不如实例内直下。
- **阿里云 OSS 中转**(量大):本地 ossutil 上传到自己的 bucket,实例内用 `ssh_helper/weights_oss_dl.sh` 的 ossutil 模式拉取(`ossutil cp -r -f -u --parallel 8 --job 4 --checkpoint-dir ...` 支持断点续传)。
- **scp 直传**(应急):`scp -P <PORT> -r <本地目录> root@<HOST>:/root/autodl-fs/models/`,家宽上行慢,不推荐超过 10G。

**1.6 下载完:无卡机关机**

控制台「关机」(不是释放!)。权重在 autodl-fs,随时可正常开机换 GPU。

> ⚠️ 实测注意:**释放实例前先确认网盘数据的保留策略**(以控制台提示为准);重要权重建议在 OSS/本地另留一份备份。

### 阶段 2:GPU 实例开机与全栈启动

**2.1 开机**

控制台把无卡实例 **关机** → **正常开机**(选 GPU,RTX PRO 6000 96G)。同一实例关机再开,系统盘/数据盘保留;如果是**全新实例**,镜像选 `rag-env-vllm029-cu130-qwen3`(内置 `/root/envs-gateway` 环境:vllm + litellm + FlagEmbedding + sentence-transformers + fastapi/qdrant-client/mcp,共享 vllm 的 torch,不用装系统 CUDA)。

从零自建环境的备选(脚本见 `ssh_helper/setup_gpu.sh`,约 20 分钟):实例内克隆本项目后执行 `bash setup_gpu.sh`,它会配清华源、`conda create -p /root/autodl-tmp/envs/gateway python=3.12` 并装 vllm/litellm/检索依赖。

**2.2 更新连接信息**

新实例的 SSH 地址/端口可能变了:把控制台新登录指令填回 `ssh_helper/gpu_ssh.env`。

**2.3 一键部署(项目根 Git Bash)**

```bash
bash ssh_helper/deploy_gpu.sh all
```

`all` 模式一次做完(幂等,可反复跑):
1. 推 `litellm_config_alias.yaml` → GPU `/root/autodl-tmp/gw/litellm_config.yaml`;
2. 推 `start_gpu.sh` 与 `gpu.env`(密钥只落 GPU 机)→ `/root/autodl-tmp/gw/`;
3. 推运行面代码 `mcp_servers/ RAG/ config/ env/` → `/root/autodl-tmp/app/`,推 `ret.env` → `/root/autodl-tmp/app/ret.env`;
4. **自动生成本地 `ssh_helper/tunnel_target.env`**(记录本次 GPU HOST/PORT,供阶段 4 隧道用);
5. 幂等启动全栈(start_gpu.sh),再重启检索服务。

**2.4 start_gpu.sh 做了什么(理解它才能排障)**

- **权重软链**:把 autodl-fs 里的权重按通配符(`*Qwen3-32B*FP8*` 等)链接到规范路径——LLM 两个 → `/root/autodl-tmp/models/{llm-main,llm-light}`(vLLM 直读);检索三件套 → `~/.cache/huggingface/hub_local/{BAAI__bge-m3, BAAI__bge-reranker-v2-m3, sentence-transformers__clip-ViT-B-32}`(`RAG/embed.py` 离线加载硬编码该路径,`HF_HUB_OFFLINE=1`)。**实例重启后重跑本脚本即可重建软链**。
- **vLLM 主模型 :8000**:`Qwen3-32B-FP8`,`--served-model-name vllm-main vllm-light`(双别名)、`--max-model-len 16384`、`--gpu-memory-utilization 0.66`、`--enable-auto-tool-choice --tool-call-parser hermes`、`--reasoning-parser qwen3`。
- **等主模型健康后才起副模型**(避免并发抢显存):vLLM 副模型 :8001,`Qwen3-14B-FP8`,`--max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.26`。
- **LiteLLM :4000**:幂等修复镜像坑(入口脚本 shebang 指向旧路径 `/root/autodl-tmp/envs/gateway` 时自动 `sed` 改为新路径),从 `gpu.env` 读 `LITELLM_MASTER_KEY/CLOUD_API_KEY` 起网关(main/light 别名 + 云端兜底 fallback)。
- **检索微服务 :8002**:`python -m mcp_servers.retrieval.service`,读 `/root/autodl-tmp/app/ret.env`,EMBED_DEVICE=cuda,连 qdrant 走本机 6333(由 VM 的 -R 反向隧道提供,见阶段 4)。

显存布局(单卡 96G):main 0.66(≈63G = FP8 权重 33G + KV ~30G,够 20 并发 ×11K 上下文)+ light 0.26(≈25.5G)+ 剩余 ≈7G 给检索三件套。**light 低于 0.19 会因 KV 归零起不来**。双卡/四卡换机时按脚本头注释传 `MAIN_GPUS/MAIN_TP/MAIN_GPU_FRAC/LIGHT_GPUS/RETRIEVAL_GPUS`。

降级开关:`LIGHT_ALIAS_MAIN=1 bash start_gpu.sh` = 不单独起 14B,light 别名复用主模型(副模型起不来时的应急方案)。

**2.5 观察启动与健康检查**

首次启动 vLLM 要编译 CUDA graph,**主模型约 5–10 分钟**:

```bash
ssh -p <PORT> root@<HOST>
tail -f /root/autodl-tmp/logs/vllm_main.log    # 等 "Uvicorn running on ..."
tail -f /root/autodl-tmp/logs/start_gpu.out    # 总启动日志
```

四件套健康(实例内):

```bash
curl -s -m2 http://127.0.0.1:8000/health   # vLLM main
curl -s -m2 http://127.0.0.1:8001/health   # vLLM light
curl -s -m2 http://127.0.0.1:4000/health/liveliness   # LiteLLM
curl -s -m2 http://127.0.0.1:8002/health   # 检索微服务
```

测 LiteLLM 对话**必须带真 master key**(`gpu.env` 里的那串),用别的 key 会报误导性的 `No connected db.`:

```bash
source /root/autodl-tmp/gw/gpu.env
curl -s http://127.0.0.1:4000/v1/chat/completions -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"light","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

> ⚠️ 本地直接 `bash ssh_helper/start_gpu.sh` 或 ssh_run 启动时,本地命令可能 2 分钟超时(exit 143),**但远端往往已把整个流程跑完**——先查健康端点再决定是否重跑,脚本本身幂等,重复触发也只是多一次重启。

### 阶段 3:VM Docker 五容器

**3.1 前提**

内网 Ubuntu VM(本项目 192.168.88.138),装好 Docker Engine + docker compose 插件,内存 ≥8G(本机不跑模型)。

**3.2 传部署包(镜像 3.9G + 快照 3.4G)**

项目根 Git Bash(流式 tar 推送,恒定低内存、带进度):

```bash
eval $(grep -E '^(HOST|PORT|USER|PASS)=' ssh_helper/vm_ssh.env | sed 's/^/VM_/')

# 大文件:镜像 tar 与 qdrant 快照
python ssh_helper/push_big.py --host $VM_HOST --port $VM_PORT --user $VM_USER \
  --password "$VM_PASS" --local projectdocker/images    --remote projectdocker/images
python ssh_helper/push_big.py --host $VM_HOST --port $VM_PORT --user $VM_USER \
  --password "$VM_PASS" --local projectdocker/snapshots --remote projectdocker/snapshots

# 小文件:compose/脚本/env
for f in docker-compose.yml load-images.sh setup-env.sh restore-snapshots.sh; do
  python ssh_helper/ssh_push.py --host $VM_HOST --port $VM_PORT --user $VM_USER \
    --password "$VM_PASS" --local projectdocker/$f --remote projectdocker/$f
done
python ssh_helper/ssh_push.py --host $VM_HOST --port $VM_PORT --user $VM_USER \
  --password "$VM_PASS" --local env/env.env --remote env/env.env   # setup-env.sh 按相对路径读 ../env/env.env
```

(`pkg.tar` 是 5 个镜像的 docker save 合包:semi-backend / semi-web / qdrant / redis-stack / pgvector-postgres,不含任何模型。)

**3.3 VM 上执行部署**

```bash
ssh <USER>@<VM_HOST>
cd ~/projectdocker

bash load-images.sh          # 1) docker load 导入 5 个镜像(离线,不构建)
bash setup-env.sh            # 2) 一键生成 .env:JWT/Redis/Postgres 密码 openssl 随机;
                             #    LLM 网关地址/检索地址/token 全部内置对齐;
                             #    读 ../env/env.env 校验 OPENAI_API_KEY/RETRIEVAL_INTERNAL_TOKEN/LITELLM_MASTER_KEY
docker compose up -d         # 3) 起 5 容器(qdrant/redis/postgres/backend/web)
docker compose exec backend python -m memories.db.init_all   # 4) 建长期记忆表(幂等,首次一次)
bash restore-snapshots.sh    # 5) 恢复知识库快照:ald_text 40万块 + ald_image 14.9万块(幂等,已有集合跳过)
```

关键设计(排障前先懂):
- backend 容器访问 GPU 统一走 `host.docker.internal:4000 / :8002`(compose 已配 `host-gateway`),所以隧道 -L 必须**绑 0.0.0.0** 而不是默认 127.0.0.1;
- redis 必须是 `redis-stack-server`(RediSearch):LangGraph RedisSaver 的 checkpoint 依赖 `FT.*` 命令,纯 redis 会降级内存 checkpoint;
- qdrant/backend 只绑 127.0.0.1,对外只开放 **3000**(web,host 网络模式,镜像内 Next.js rewrite 硬编码 127.0.0.1:8001)。

**3.4 日常代码迭代同步(开发用)**

```bash
bash ssh_helper/deploy_vm.sh restart      # 同步 backend_src + bind mount + 重启 backend 容器
bash ssh_helper/deploy_vm.sh norestart    # 只同步不重启
```

**3.5 账号初始化**

浏览器打开 `http://<VM_IP>:3000` 注册用户。**第一个注册用户**若在请求头带 `X-Bootstrap-Token: <.env 里的 ADMIN_BOOTSTRAP_TOKEN>` 则授予 admin。评测演示账号:`eval1 / Eval#pass1`。

### 阶段 4:SSH 三向隧道(VM → GPU)

一条 SSH 连接做双向三转发(脚本 `ssh_helper/start_tunnel.sh`):

| 转发 | 方向 | 用途 |
|---|---|---|
| `-L 0.0.0.0:8002 → GPU:8002` | VM 收 → GPU 检索 | backend 容器调检索微服务 |
| `-L 0.0.0.0:4000 → GPU:4000` | VM 收 → GPU LiteLLM | backend 调 LLM 网关 |
| `-R 6333 → VM:6333` | GPU 收 → VM qdrant | GPU 检索服务回查知识库 |

**4.1 启动**

**前提**:阶段 2.3 的 `deploy_gpu.sh` 已生成 `ssh_helper/tunnel_target.env` 且 `deploy_vm.sh` 已把它推到 VM 家目录(顺序:先 deploy_gpu 再 deploy_vm)。

```bash
ssh <USER>@<VM_HOST>
cd ~/projectdocker
nohup bash start_tunnel.sh > tunnel.log 2>&1 &
tail -f tunnel.log    # 脚本是 while-true 自愈循环,断线 5 秒后自动重连
```

> ⚠️ `tunnel_target.env` 在 VM 上有**两处副本**(家目录 `~/tunnel_target.env` 与 `~/projectdocker/tunnel_target.env`),脚本**同目录副本优先**。换 GPU 实例后:重跑 `deploy_gpu.sh`(本地重新生成)→ `deploy_vm.sh`(重新推送)→ 重启隧道,不要手改一处漏另一处。

**4.2 验证(VM 上)**

```bash
curl -s -m2 http://127.0.0.1:4000/health/liveliness   # 通了 = LiteLLM 经隧道可达
curl -s -m2 http://127.0.0.1:8002/health              # 检索微服务
curl -s -m2 http://127.0.0.1:6333/collections         # -R 反向:GPU 上的检索能回查 VM qdrant
```

排障:隧道没起 → `pgrep -af "ssh -N"`;端口被占/绑不上 → `ssh_helper/fix_tunnel_bind.sh`;Windows 本机开发不走 VM,直接用 `http://192.168.88.138:4000`(-L 绑了 0.0.0.0,局域网可达)。

### 阶段 5:端到端验证

1. 浏览器打开 `http://<VM_IP>:3000`,登录;
2. 新建会话,问一个领域问题(如某设备报警码的处理参数);
3. 检查:流式出答案、答案带**出处**(文档名+页码)、路由 tier 事件正常(simple/raglite/react);
4. GPU 侧 `tail -f /root/autodl-tmp/logs/litellm.log` 能看到 `POST /v1/chat/completions 200`。

全链路即:浏览器 → VM web:3000 → backend:8001 → (隧道) GPU LiteLLM:4000 → vLLM:8000/8001;backend → (隧道) 检索:8002 → (-R 隧道) VM qdrant:6333。

### 阶段 6:评测复跑(20 并发 × 50 题 = 1000 条)

工具五件套在 `eval/pipeline/pipeline_50x5/`,全部可一键复跑:

```bash
cd eval/pipeline/pipeline_50x5

# 1) 全量回归(真实 HTTP/SSE,题集 qa_sets_20x50,20 用户各 50 题)
regression_50x5.bat
#    断点续跑:删除输出目录里的 error/空答案记录后重跑,已成功题目自动跳过

# 2) 离线补查重排前候选池(R_pool 分母;注意必须用 (user,id) 联合键,裸 id 跨用户撞车)
python fetch_pools.py

# 3) 检索硬指标(MRR/NDCG@k/Rfact@k/P_ret,--pools 池内召回)
python concurrent_metrics.py --pools <pools.jsonl 路径>

# 4) GPU LLM-as-Judge(Qwen3-32B 本地裁判,零云成本;务必显式 --out-dir 指到本次 run 目录)
python judge_50x5.py --out-dir eval/runs/<run目录>

# 5) 免 LLM 参数扫描(直调 /search_text + /debug_cfg 热调参;需 ret.env 开 RETRIEVAL_DEBUG_CFG=1)
python sweep_retrieval.py
```

评测注意事项(全部实测踩过):
- Windows 侧先 `export NO_PROXY="192.168.88.138,127.0.0.1,localhost"`,否则系统代理劫持一切请求;
- **评测中途不要重启检索服务**(会打断在途请求)、**盯住 AutoDL 余额**(耗尽自动关机 → 剩余题全打 Connection error;恢复:控制台开机 → `deploy_gpu.sh all` 幂等重启 → 删 error 记录断点续跑);
- 大批量后台跑用 `python -u`(否则输出缓冲,日志看起来是空的);
- 裁判进度观察:`grep -c 'POST /v1/chat' litellm.log` 增量(约 164 次/分,1000 题约 20 分钟)。

### 附 A:踩坑清单

1. **litellm 入口脚本 shebang 指旧 conda 路径**(镜像复制的坑):新实例首启报 `No such file or directory`;start_gpu.sh 幂等 `sed` 修复。
2. **LiteLLM 必须用真 master key** 测试:设置过 `LITELLM_MASTER_KEY` 时用别的 key 会报误导性的 `No connected db.`。
3. **Windows 系统代理劫持 httpx**:连 127.0.0.1/内网也要显式 `NO_PROXY`。
4. **ssh_run 本地超时 ≠ 远端失败**:nohup 启动类命令本地 2 分钟超时(exit 143),远端可能已跑完,先查健康端点。
5. **`pkill -f` 自杀**:启动命令文本含目标进程名时 pkill 会连自己一起杀;deploy_gpu.sh 已改 pgrep 枚举排除自身。
6. **tunnel_target.env 两处副本**,同目录优先;换实例后按 4.1 流程全量刷新。
7. **题库题 id 只在用户内唯一**:跨用户聚合必须用 `(user,id)` 联合键,裸 id 会把 1000 题去重成 100 题。
8. **余额耗尽自动关机**:压测中途大面积 Connection error;恢复流程见阶段 6。
9. **qdrant 容器只绑 127.0.0.1**:Windows 直连 6333 需 `ssh_helper/forward_6333.py` 转发,或直接走检索服务 8002 的 `/get_chunk`。
10. **light 显存 <0.19 起不来**(`No available memory for cache blocks`):96G 单卡用 0.26;实在不行 `LIGHT_ALIAS_MAIN=1`。
11. **同地区才能挂 autodl-fs**:换实例/租无卡机前先确认地区一致。
12. **judge 必须显式 `--out-dir`**,默认会写进 runs/adhoc。
13. **镜像内 Next.js rewrite 硬编码 127.0.0.1:8001**,所以 web 容器用 host 网络模式——不要改成 bridge。
14. **数据盘 vs 网盘**:`/root/autodl-tmp` 随实例释放清空;权重/长存文件一律放 `/root/autodl-fs`。

### 附 B:日常开关机标准流程

**开机(GPU 实例已存在)**:
```bash
# 1) 控制台开机,核对 SSH 指令(gpu_ssh.env 不用改,若没变)
# 2) 项目根:
bash ssh_helper/deploy_gpu.sh all     # 幂等:软链权重 + 起全栈 + 重启检索
# 3) VM 隧道自愈循环会自动重连,确认:
ssh <VM> 'curl -s -m2 http://127.0.0.1:8002/health'
```

**关机(省钱,¥6.98/h)**:确认无评测在跑 → 控制台关机(保留磁盘);长期不用且权重已备两份时可释放实例。

**换新实例**:填新 gpu_ssh.env → `deploy_gpu.sh all`(自动更新 tunnel_target.env)→ `deploy_vm.sh`(推隧道目标)→ 重启 VM 隧道 → 全链路验证。

## 📁 目录结构

```
├── RAG/                  # 文档解析、分块、嵌入入库（PDF/PPT/Word/CAD 渲染）
├── agent_reasoning/      # 三级范式：simple / raglite（PE）/ react
├── mcp_servers/          # 检索微服务（MCP 协议，GPU 侧部署）
├── context management/   # 崩溃安全的上下文压缩
├── server/               # FastAPI 业务后端（SSE 流式、限流、鉴权）
├── web/                  # Next.js 前端
├── projectdocker/        # Docker Compose 五容器编排
├── eval/                 # 评测：pipeline/（可复用流水线）+ runs/（每次评测产物）
├── tests/                # 单元/集成测试
└── ssh_helper/           # GPU/VM 部署与隧道脚本
```

## 📄 License

仅供学习交流使用。
