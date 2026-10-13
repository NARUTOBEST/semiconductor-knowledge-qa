# dev 调优集出题管线(v2,2026-10-01)

与 `pipeline/heldout/` 平行、互不混放的 dev 侧出题管线。
总协议见 `../protocol_v2.md`。

## 本目录内容
| 文件 | 用途 |
|---|---|
| `fetch_material.py` | 从 Qdrant(16333 转发)抓取本 track 8 篇文档的真实 chunk 原文 |
| `draft_material/` | 8 篇素材 txt + 草稿 json(每篇 25 题,含 evidence 逐字摘录) |
| `build_datasets_dev.py` | 草稿 -> `eval/qa_sets/v2_20261001/dev_200.json` + 校验报告 |
| `_extract_result.py` | 出题 agent 结果恢复(从会话 transcript) |
| `_verify_acceptance.py` | evidence 溯源覆盖率校验(自动发现本目录全部草稿) |
| `_prompt_template.txt` | 出题 agent 提示词模板 |
| `acceptance_report.json` / `_acceptance_failures.txt` | AI 验收记录(dev 侧) |
| `verify_log.md` | 人工核对清单(dev 200) |

dev 题集用于迭代调参与回归对比;对外泛化结论一律以 heldout 产物为准。
