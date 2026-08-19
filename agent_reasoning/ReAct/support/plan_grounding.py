# -*- coding: utf-8 -*-
"""计划步骤覆盖度追踪器(异步守护线程)。

设计(对应需求):在 ReAct 工作流中异步起一个线程,随着检索资料不断到达,
持续维护一份"计划每一步 是否已被已检索资料覆盖"的判定文档;在最终交给
grounding 检查之前,由该线程基于维护好的文档调用 LLM 做一次覆盖判定。
若某一步未被覆盖/资料有误,coverage_check_node 据此回退到该步重新检索。

为什么用线程而不是节点:LangGraph 节点是同步串行的,无法在 agent↔tools
循环期间"后台"做事。把追踪器作为不可序列化对象放进
config["configurable"](与 TraceRecorder 同机制),守护线程在每次
tools_node 写入新来源后被唤醒增量重建覆盖文档,与后续 LLM 生成并行。

线程生命周期:plan_node 在 need_plan=True 时创建并 start();finalize/
runner 结束时 close()。daemon=True,进程退出不阻塞。断点续跑跨进程重启后
线程不复存在,节点侧需做"无 tracker 则跳过覆盖判定、fail-open 进 grounding"
的兜底(见 coverage_check_node)。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Callable, Optional

import config as C  # noqa: E402

logger = logging.getLogger("agent")

# 覆盖文档/判定相关参数
_DOC_SNIPPET_LEN = 200     # 每条来源片段进入覆盖文档的截断长度
_DOC_MAX_PER_STEP = 3      # 每个步骤最多挂几条最相关片段
_JUDGE_TIMEOUT = 25.0      # coverage_check_node 等待覆盖判定的最长时间(s);LLM 调用本身 20s,留余量给文档重建
_STOP_TIMEOUT = 1.0        # close 等待守护线程退出(s)

# 判定 LLM 输出解析:{"covered":[true,false,...],"uncovered_steps":[2],"reason":"..."}
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_COVERAGE_PROMPT = """你是检索计划覆盖度评审员。下面给出一个多步检索计划,以及目前为止实际检索到的资料片段。
请逐步判断:每一步计划所需的信息,是否已经被现有检索资料覆盖(即资料中确实包含支撑该步骤结论的内容)。

判定标准:
- 该步骤对应的资料片段确实包含相关且足够的信息 → 覆盖(true)
- 没有任何资料与该步骤相关,或资料明显不相关/内容有误/信息缺失 → 未覆盖(false)

只输出 JSON,不要任何其他文字,格式:
{{"covered": [true/false, ...], "uncovered_steps": [未覆盖的步骤序号,从1开始], "reason": "一句话说明"}}

计划(共 {n} 步):
{steps}

已检索资料覆盖文档:
{doc}
"""


class CoverageTracker:
    """后台维护覆盖文档并按需执行 LLM 覆盖判定的线程安全追踪器。"""

    def __init__(self, trace_id: str,
                 llm_caller: Callable[..., tuple[Any, Optional[Exception]]]):
        self.trace_id = trace_id
        self._llm_caller = llm_caller  # 复用 llm_create_with_retry(主/备切换)
        self._lock = threading.RLock()
        self._wake = threading.Event()

        self._steps: list[str] = []
        self._question: str = ""
        # chunk_key -> source_dict(content 截断保留)
        self._sources: dict[str, dict[str, Any]] = {}

        # 持续维护的覆盖文档(文本);来源变化后由守护线程重建
        self._doc: str = ""
        self._doc_dirty = True

        # 待处理的判定请求(同一时刻只保留最新一次)
        self._judge_answer: Optional[str] = None
        self._judge_future: Optional[_FutureBox] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ 生命周期
    def start(self) -> "CoverageTracker":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._run, name=f"cov-{self.trace_id}", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(_STOP_TIMEOUT)

    # ------------------------------------------------------------ 外部喂入
    def set_plan(self, steps: list[str], question: str = "") -> None:
        with self._lock:
            self._steps = [str(s).strip() for s in steps if str(s).strip()]
            self._question = question or ""
            self._doc_dirty = True
        self._wake.set()

    def update_sources(self, sources: dict[str, dict[str, Any]]) -> None:
        """合并新到达的来源(通常传 state['collected_sources'] 的全量或增量)。"""
        if not sources:
            return
        changed = False
        with self._lock:
            for k, s in sources.items():
                if k and k not in self._sources:
                    self._sources[k] = s
                    changed = True
            if changed:
                self._doc_dirty = True
        if changed:
            self._wake.set()

    # ------------------------------------------------------------ 判定触发
    def request_judgment(self, answer: str) -> Optional["_FutureBox"]:
        """请求一次 LLM 覆盖判定(由守护线程执行),返回 FutureBox;无计划返回 None。

        coverage_check_node 在 grounding 之前调用,并在其上 block 至多 _JUDGE_TIMEOUT。
        若上一次判定尚未完成,直接复用其 future(不重复提交)。
        """
        with self._lock:
            if not self._steps:
                return None
            if self._judge_future is not None and not self._judge_future.done:
                return self._judge_future
            box = _FutureBox()
            self._judge_answer = answer or ""
            self._judge_future = box
        self._wake.set()
        return box

    # ------------------------------------------------------------ 守护线程
    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                if self._doc_dirty:
                    self._rebuild_doc()
                self._maybe_judge()
            except Exception:  # 守护线程绝不允许抛出逃逸
                logger.exception("coverage tracker loop error")

    def _rebuild_doc(self) -> None:
        """重建"步骤→相关资料片段"覆盖文档(轻量关键词归属,非权威判定)。"""
        with self._lock:
            steps = list(self._steps)
            sources = list(self._sources.values())
            self._doc_dirty = False
        if not steps:
            self._doc = ""
            return

        blocks: list[str] = []
        for i, step in enumerate(steps, 1):
            step_tokens = _keywords(step)
            ranked = []
            for s in sources:
                text = f"{s.get('heading', '')} {s.get('content', '')}"
                score = _overlap(step_tokens, _keywords(text))
                if score > 0:
                    ranked.append((score, s))
            ranked.sort(key=lambda x: x[0], reverse=True)
            top = ranked[:_DOC_MAX_PER_STEP]
            if not top:
                blocks.append(f"步骤{i}. {step}\n  - 无明显相关检索片段")
                continue
            lines = [f"步骤{i}. {step}"]
            for _, s in top:
                snippet = (s.get("content") or "")[:_DOC_SNIPPET_LEN]
                meta = f"[{s.get('source_stem', '')} {s.get('page', '')}]".strip()
                lines.append(f"  - {meta} {snippet}")
            blocks.append("\n".join(lines))
        with self._lock:
            self._doc = "\n\n".join(blocks)

    def _maybe_judge(self) -> None:
        with self._lock:
            box = self._judge_future
            answer = self._judge_answer
            steps = list(self._steps)
            doc = self._doc
            if box is None or box.done:
                return
            if self._doc_dirty:
                # 资料刚更新,文档尚在重建中;等下一轮唤醒再判,保证基于最新文档
                return
            # 取出请求,避免重复判定
            self._judge_future = None
            self._judge_answer = None

        if not steps or not doc:
            box.set_result(None)
            return

        steps_text = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        prompt = _COVERAGE_PROMPT.format(n=len(steps), steps=steps_text, doc=doc)
        try:
            resp, err = self._llm_caller(
                trace_id=f"cov-{self.trace_id}",
                model=C.OPENAI_TEXT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                timeout=20,
            )
            if err is not None:
                logger.info("coverage judge LLM fail: %s", err)
                box.set_result(None)
                return
            raw = (resp.choices[0].message.content or "").strip()
            box.set_result(_parse_judgment(raw, len(steps)))
        except Exception as e:  # 判定失败 fail-open,不阻断主流程
            logger.info("coverage judge error: %s", e)
            box.set_result(None)


class _FutureBox:
    """最小的一次性同步占位(避免引入 concurrent.futures 的额外语义)。"""

    def __init__(self) -> None:
        self._ev = threading.Event()
        self._result: Any = None

    def set_result(self, value: Any) -> None:
        self._result = value
        self._ev.set()

    def result(self, timeout: Optional[float] = None) -> Any:
        self._ev.wait(timeout)
        return self._result

    @property
    def done(self) -> bool:
        return self._ev.is_set()


# ------------------------------------------------------------ 纯函数工具
def _keywords(text: str) -> set[str]:
    """极简分词:CJK 二元组 + 英文/数字小写词。够用且无外部依赖。"""
    text = (text or "").lower()
    toks: set[str] = set()
    for w in re.findall(r"[a-z0-9]+", text):
        if len(w) >= 2:
            toks.add(w)
    cjk = re.findall(r"[一-鿿]+", text)
    for seg in cjk:
        for i in range(len(seg) - 1):
            toks.add(seg[i:i + 2])
        if len(seg) == 1:
            toks.add(seg)
    return toks


def _overlap(a: set[str], b: set[str]) -> int:
    return len(a & b)


def _parse_judgment(raw: str, n_steps: int) -> Optional[dict[str, Any]]:
    """解析判定 LLM 输出为 {covered:[...], uncovered_steps:[...], reason:str}。

    解析失败返回 None(调用方 fail-open)。
    """
    if not raw:
        return None
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    covered = obj.get("covered")
    if not isinstance(covered, list) or len(covered) != n_steps:
        return None
    covered_b = [bool(x) for x in covered]
    uncov = [i + 1 for i, c in enumerate(covered_b) if not c]
    reason = str(obj.get("reason") or "").strip()
    return {"covered": covered_b, "uncovered_steps": uncov, "reason": reason}
