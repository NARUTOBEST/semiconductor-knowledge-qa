# heldout 泛化集出题管线(v2,2026-10-01)

与 `pipeline/dev/` 平行、互不混放的 heldout 侧出题管线。
总协议见 `../protocol_v2.md`(v2:取消 source 隔离红线,全库遴选文档)。

## 本目录内容
| 文件 | 用途 |
|---|---|
| `fetch_material.py` | 从 Qdrant(16333 转发)抓取本 track 8 篇文档的真实 chunk 原文 |
| `draft_material/` | 8 篇素材 txt + 草稿 json(每篇 25 题,含 evidence 逐字摘录) |
| `build_datasets_heldout.py` | 草稿 -> `eval/qa_sets/v2_20261001/heldout_200.json` + 校验报告 |
| `_extract_result.py` | 出题 agent 结果恢复(从会话 transcript) |
| `_verify_acceptance.py` | evidence 溯源覆盖率校验(自动发现本目录全部草稿) |
| `_prompt_template.txt` | 出题 agent 提示词模板 |
| `acceptance_report.json` / `_acceptance_failures.txt` | AI 验收记录(heldout 侧) |
| `verify_log.md` | 人工核对清单(heldout 200) |

## 沿用的纪律(继承 v1 协议,与出题管线解耦)
- **配置冻结**:出题 commit 后冻结线上配置,测试完成前不改参数
- **一次定版**:冻结配置上只跑一遍;确需重跑须在 run 产物 README 记录原因与两次结果
- **预注册达标线**:跑之前写死合格标准,测完不得修改
(v1 的"来源隔离"红线已于 2026-10-01 由用户决定取消;v1 协议与脚本存档于
`../old_v1_docx方案/heldout_pipeline_v1/`)

heldout 样本**严禁**在调优中泄漏:已跑过的 heldout 题不得进入 dev 或备选池补入。
