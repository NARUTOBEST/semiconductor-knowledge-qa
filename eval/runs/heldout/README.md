# eval/runs/heldout/ — 独立留出集评测产物

只存放满足 `eval/pipeline/heldout/README.md` 独立性协议的评测产物
(出题与 dev 迭代调参隔离、配置冻结、一次定版、预注册达标线)。

每次评测一个文件夹,必须包含:qa_sets/(冻结题集)、protocol.md(预注册声明)、
source_isolation.json(来源隔离验证)、原始 jsonl 与指标产物。

**口径红线**:dev/ 的指标只能做版本间回归对比;对外宣称泛化效果一律以本目录产物为准。
