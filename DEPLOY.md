# 部署手册（云端 LLM + 检索专用 GPU 版）

> 三台机器角色（2026-10-10 定稿）：**LLM 全走火山推理接入点；GPU 机跑 qdrant 副本 + 检索；
> VM 的 qdrant 仅作写入主库与 judge 读源**。

```
Windows 开发机 (C:\project3)        VM 项目机 (192.168.88.138, lly/123)        GPU 机 (AutoDL, 西北B区)
  ssh_helper/*.sh ──push──▶  ~/projectdocker (docker compose 5 容器) ──SSH隧道(纯正向)──▶  检索:8002
        │                        │  backend LLM 直连火山 Ark(公网)              qdrant 副本:127.0.0.1:6333(本机)
        └────────────────────────┴─▶ Ark: 主=deepseek-v4-flash-压测  副=doubao-seed-2.0-lite-压测
  内网入口 http://192.168.88.138:3000
```

- **LLM 不自建**：主/副模型走火山控制台的两个推理接入点（按量 TPM 配额，100万 TPM 可扛 20 并发），backend 公网直连
- **GPU 机跑检索 + qdrant 副本**：BGE-m3 embed + CLIP + 进程内 FlagLLMReranker（双副本攒批）；qdrant 数据在 GPU 本机 SSD，HNSW 随机读不走隧道也不吃 VM 磁盘
- **隧道纯正向单向**：`start_tunnel.sh` 起 10 条 `-L 0.0.0.0:8002+i → GPU:8002`（GPU_TUNNEL_PORTS，8002~8011），backend 的 MCP 连接池按 RETRIEVAL_SERVICE_URLS 一一绑定分摊 transport；**没有任何 -R 反向通道**（GPU 本机查库，不需要回传）
- 重排为进程内 transformers（vLLM 承载 gemma reranker 已实测不可行，见坑清单 14）

---

## 一、GPU 机部署（换机/新开）

> **实例规格与开机方式（2026-10-09 起）**：检索服务默认用 **西北B区 vGPU-32G** 搭建。
> **开/换实例自己用 Playwright 浏览器上 AutoDL 控制台操作，不问用户**——登录态在
> `playwright-edge-profile`；优先"克隆实例"（系统盘必选，保留 SSH 密码），克隆完成后
> 把新实例 HOST/PORT/PASS 写回 `ssh_helper/gpu_ssh.env`。若目标主机"空闲 GPU 不足"，
> 刷新换主机或就近换同价位规格，不要卡死在一台机器上。

私有镜像 `image-14f96128d0` 已含 `/root/envs-gateway`（检索依赖全套），**不要重装环境**；
权重在 autodl-fs 平铺目录（bge-m3 / bge-reranker-v2-m3 / clip-ViT-B-32 / bge-reranker-v2-gemma），
`/root/.cache/huggingface/hub_local/` 软链指过去（换机后软链若失效需重建），RAG/embed.py 的
`_plain_snapshot_ok()` 兜底识别平铺目录（无 HF refs 元数据）。

### 1) 代码 + 服务

```bash
bash ssh_helper/deploy_gpu.sh restart   # 推 mcp_servers/RAG/config/env/ret.env/tool_circuit.py
                                        # + 起本机 qdrant(:6333) + 重启检索(:8002)
```

restart 块自动做：杀旧检索/qdrant/qdrant_lb 残留 → 写 `/root/autodl-tmp/qdrant_config.yaml`
（storage 指向 `/root/autodl-tmp/qdrant_storage`，绑 127.0.0.1:6333）→ 起数据盘上的
`/root/autodl-tmp/bin/qdrant` → 起检索服务。

### 2) qdrant 数据（换机必做，~20 分钟）

storage 目录在**数据盘** `/root/autodl-tmp/qdrant_storage`，换机/系统盘重置后为空，需从 VM 重传：

```bash
# VM 上(gzip -1 后 ~2.6G,scp 协议会被 AutoDL 掐死,必须 cat|ssh 管道,4 流并行 ~15min):
sudo tar -C /var/lib/docker/volumes/semi_qdrant_storage/_data -cf - . | gzip -1 > /tmp/qs.tar.gz
md5sum /tmp/qs.tar.gz   # 记源 md5
split -b 600M -d /tmp/qs.tar.gz /tmp/qs.part.
# 每片: cat /tmp/qs.part.XX | ssh -p <PORT> root@<HOST> "cat > /root/autodl-tmp/qdrant_in/qs.part.XX"
# GPU 上拼包校验后解压:
cat /root/autodl-tmp/qdrant_in/qs.part.* > /root/autodl-tmp/qs.tar.gz   # md5 对上再解
mkdir -p /root/autodl-tmp/qdrant_storage && tar -xzf /root/autodl-tmp/qs.tar.gz -C /root/autodl-tmp/qdrant_storage
bash ssh_helper/deploy_gpu.sh restart    # 起本机 qdrant
curl 127.0.0.1:6333/collections          # 应有 ald_text(~40万点)/ald_image
```

**同步纪律：VM 主库有写入（重新入库/新增题库）后，必须重传快照同步 GPU 副本**，否则检索命中旧数据。

### 3) GPU 机上验证

```bash
curl 127.0.0.1:6333/collections
curl -X POST 127.0.0.1:8002/search_text -H 'Content-Type: application/json' \
  -d '{"query":"固晶机","top_k":2}'   # 预热检索（首次 ~1min，之后秒回）
```

## 二、VM/backend 侧

```bash
bash ssh_helper/deploy_vm.sh restart          # 同步 backend 源码 + 重建容器
# env/env.env（backend 读这里）：
#   OPENAI_API_KEY = <火山 API Key>
#   LLM base_url   = https://ark.cn-beijing.volces.com/api/v3
#   主模型名 = <deepseek-v4-flash-压测 的 ep-xxx>
#   副模型名 = <doubao-seed-2.0-lite-压测 的 ep-xxx>
#   RETRIEVAL_SERVICE_URLS = http://host.docker.internal:8002,...,8011(十条,对应隧道端口)
# 隧道：bash ssh_helper/start_tunnel.sh(或 deploy_vm.sh 拉起 watchdog)
#       ⚠️ ~/tunnel_target.env 与 ~/projectdocker/tunnel_target.env 两份都要改，后者优先
```

JWT/judge 等本就直连 coding plan，不受影响。

> **接入点信息自取**：需要 ep-ID/配额/模型名时，用本机 Playwright 打开火山方舟控制台
> （在线推理→接入点）自己查、自己复制，不用问用户。已登录态在 `playwright-edge-profile`。

## 三、内网访问与前端

- 浏览器打开 `http://192.168.88.138:3000`，登录即用；文档预览走后端代理 `/api/media/doc`
- web 改动需重建镜像（Next.js standalone，与 backend bind mount 不同）：

```bash
cd web && npm run build && rm -rf .next/standalone/public .next/standalone/.next/static \
  && cp -r public .next/standalone/public && cp -r .next/static .next/standalone/.next/static \
  && tar -czf ../_webbuild.tgz -C .next/standalone .
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"
python ssh_helper/ssh_push.py --host 192.168.88.138 --port 22 --user lly --password 123 --local _webbuild.tgz --remote webbuild.tgz
# VM 上:解压到 ~/webbuild(Dockerfile 同级) → docker build -t semi-web:local ~/webbuild → docker compose up -d web
```

## 四、端到端验收

```bash
python3 /home/lly/e2e_media_check.py   # VM 上,源码 ssh_helper/e2e_media_check.py 先推
```

预期：带引用回答、image_url/video_url 全 200；回答带 [文档名 页码] 引用；视频走应用内浮层播放。

## 五、坑清单（实测踩过）

1. **MSYS 路径改写**：Git Bash 会改写 `/root/...`，手敲 ssh_run/ssh_push 前必须 `export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"`。
2. **CRLF**：推到 Linux 的 .sh/.env/.yaml 必须 `sed -i 's/\r$//'`。
3. **pkill -f 自杀**：远端 compound 命令会匹配杀掉自身 shell；用 pgrep 排除 `$$`/`$PPID`，或按 PID 杀。且 `pkill -f start_tunne[l]` 只杀脚本不杀它派生的 `ssh -N`（cmdline 无脚本名）——清隧道要补 `pkill -f ServerAliveInterval`。
4. **LD_LIBRARY_PATH**：手工起服务要 `export LD_LIBRARY_PATH=/root/envs-gateway/lib`（脚本已带）。
5. **隧道两份 tunnel_target.env**：`$HOME` 和 `~/projectdocker/` 后者优先，换端口都要改。
6. **检索冷启动**：起服务后先打一次真实检索预热（~1min），否则 MCP SSE 超时→"检索服务不可用"。
7. **Redis 锁残留**："该用户已有进行中的请求"→ `docker exec semi-redis redis-cli -a $PW del rl:user:eval1 rl:global`（TTL 600s 兜底）。
8. **容器名**：semi-web/semi-backend/semi-postgres/semi-redis/semi-qdrant；backend 经 `host.docker.internal:8002` 调检索。
9. **TOS/OSS 强制下载**：图→TOS、视频→OSS 都开了强制下载头；文档预览必须走后端代理 `/api/media/doc`，视频必须应用内浮层播放（VideoLightbox）。
10. **docker 组**：旧 ssh 会话拿不到新组身份，`echo $PASS | sudo -S docker ...` 过渡或重启 VM。
11. **密钥安全**：gpu_ssh.env/vm_ssh.env/gpu.env/env/ 一律不 commit。
12. **Playwright-Edge 锁**：报 "Browser is already in use" 时，只杀 CommandLine 含 `playwright-edge-profile` 的 msedge 进程，绝不能杀用户自己的 Edge；控制台截图/快照落盘后随手删，别堆在根目录。
13. **并发容量**：20 并发容量靠 10 条隧道分摊 + GPU 本机 qdrant；VM 侧 qdrant 查询（on-disk mmap HNSW）单查 3~5s、20 并发雪崩——**不要把检索的 QDRANT_URL 指回 VM**。
14. **vLLM 承载 gemma reranker 不可行**（vLLM 0.29 实测）：GemmaForCausalLM 无独立打分头，`/score` 不注册，`--convert classify` 硬转退化（prob 恒 0.5）。重排口径=进程内 FlagLLMReranker（双副本攒批，ret.env 不设 RERANK_API_BASE）。
15. **AutoDL 传大文件**：scp 协议被掐死（"lost connection"），只能 `cat | ssh "cat >"` 管道（~1.2MB/s/流，4 流并行）；先 `gzip -1` 再 `split -b 600M`，源/目的 md5 对上才解压。
16. **GPU 侧服务防自杀**：nohup 启动命令文本含服务名明文，pkill -f 会连同自身 shell 杀掉（exit 127/起不来）；deploy_gpu.sh 已改 pgrep 枚举排除 `$$`/`$PPID` 再 kill。
17. **模型平铺目录**：autodl-fs 权重是网盘 modelscope 布局（无 HF refs/main），`snapshot_download(local_files_only=True)` 报 "Cannot find an appropriate cached snapshot folder"；RAG/embed.py `_plain_snapshot_ok()` 按目录内有 config.json / modules.json（CLIP）兜底直用本地目录。

## 六、收尾

- 任务完成**关实例**（GPU 机 `shutdown -h now`，SSH 拒连即成功）；按量计费关机即停。
- 下次开机不行就控制台克隆换机（系统盘必选），流程见一。
