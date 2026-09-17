// SSE 流式调用后端 /api/chat
// 后端经 next.config.js rewrites 代理到 Python(8001),同源,免 CORS。
// 协议:data: {"type":"tier|escalation|clarify|status|sources|token|done|error", ...}\n\n

import type { Message, Source } from "./types";
import { newId } from "./storage";

export interface StreamHandlers {
  onStatus?: (msg: string) => void;
  onSources?: (sources: Source[]) => void;
  onToken?: (delta: string) => void;
  /** 复杂度路由结果:每个流的第一个事件。tier=simple|react */
  onTier?: (tier: string, confidence: number, source: string) => void;
  /** 升级到更高 tier 重跑:清空当前输出并展示"深入分析" */
  onEscalation?: (fromTier: string, toTier: string, reason: string) => void;
  /** 澄清反问:信息不足,助手先提问;options 为可选候选 */
  onClarify?: (question: string, options: string[]) => void;
  onError?: (msg: string) => void;
  onDone?: () => void;
}

/** 从 localStorage 取 token,拼 Authorization header(无 token 时返回空对象)。 */
function authHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** token 失效时清除并跳转登录页。 */
function handleUnauthorized() {
  if (typeof window === "undefined") return;
  localStorage.removeItem("token");
  localStorage.removeItem("user");
  window.location.href = "/login";
}

/**
 * 发起一次流式对话。返回一个可中止的控制器。
 * @param message   本次用户消息
 * @param history   历史 Message 列表(转成 {role,content} 发后端)
 * @param handlers  各事件回调
 * @param signal    可选 AbortSignal(用于停止生成)
 * @param threadId  会话 id,作为 LangGraph checkpoint 的 thread_id;
 *                  同一对话复用它即可跨请求续跑工作记忆
 */
export async function streamChat(
  message: string,
  history: Message[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
  threadId?: string
): Promise<void> {
  const payload: Record<string, unknown> = {
    message,
    history: history
      .filter((m) => m.content && !m.streaming && !m.error)
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-10), // 只发最近 5 轮(10 条)，与后端 history[-10:] 对齐
  };
  if (threadId) payload.thread_id = threadId;

  let resp: Response;
  try {
    resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (e: any) {
    if (e?.name === "AbortError") {
      handlers.onDone?.();
      return;
    }
    handlers.onError?.(`无法连接后端: ${e?.message || e}`);
    handlers.onDone?.();
    return;
  }

  // 401:token 缺失/过期,跳转登录
  if (resp.status === 401) {
    // 也要收尾:onDone 让页面把 streaming 复位、占位消息落地,
    // 防止整页跳转被拦截时 UI 永久卡在生成态
    handlers.onError?.("登录已过期,请重新登录");
    handlers.onDone?.();
    handleUnauthorized();
    return;
  }

  // 400/413:输入校验失败,提取后端返回的具体错误信息
  if (!resp.ok) {
    let errMsg = `后端错误 (HTTP ${resp.status})`;
    try {
      const data = await resp.json();
      errMsg = data.error || data.detail || errMsg;
    } catch {}
    handlers.onError?.(errMsg);
    handlers.onDone?.();
    return;
  }

  if (!resp.body) {
    handlers.onError?.("后端返回空流");
    handlers.onDone?.();
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 按 SSE 事件边界(\n\n)切分
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        // 每行形如 data: {...}
        for (const line of rawEvent.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const jsonStr = trimmed.slice(5).trim();
          if (!jsonStr) continue;
          let obj: any;
          try {
            obj = JSON.parse(jsonStr);
          } catch {
            continue;
          }
          switch (obj.type) {
            case "tier":
              handlers.onTier?.(obj.tier, obj.confidence ?? 0, obj.source || "");
              break;
            case "escalation":
              // 升级重跑:清空旧输出
              handlers.onEscalation?.(
                obj.from_tier || "",
                obj.to_tier || "",
                obj.reason || ""
              );
              break;
            case "clarify":
              handlers.onClarify?.(obj.question || "", obj.options || []);
              break;
            case "status":
              handlers.onStatus?.(obj.message || "");
              break;
            case "sources":
              handlers.onSources?.(obj.items || []);
              break;
            case "token":
              handlers.onToken?.(obj.delta || "");
              break;
            case "error":
              handlers.onError?.(obj.message || "未知错误");
              break;
            case "done":
              handlers.onDone?.();
              return;
          }
        }
      }
    }
    // 流自然结束但没收到 done,也收尾
    handlers.onDone?.();
  } catch (e: any) {
    if (e?.name === "AbortError") {
      handlers.onDone?.();
      return;
    }
    handlers.onError?.(`流读取中断: ${e?.message || e}`);
    handlers.onDone?.();
  }
}

export function mkMessage(role: "user" | "assistant", content = ""): Message {
  return {
    id: newId(),
    role,
    content,
    createdAt: Date.now(),
  };
}
