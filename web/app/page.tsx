"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { TitleBar } from "@/components/TitleBar";
import { ChatBox } from "@/components/ChatBox";
import { streamChat, mkMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import {
  loadConversations,
  saveConversations,
  deleteConversationServer,
  newId,
  titleFromMessage,
} from "@/lib/storage";
import type { Conversation, Message } from "@/lib/types";

export default function Page() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("");
  // 本次请求所选推理范式(simple/react),用于展示对应 UI
  const [tier, setTier] = useState<string>("");
  const abortRef = useRef<AbortController | null>(null);

  // ---- 路由守卫:未登录跳转 /login ----
  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    }
  }, [authLoading, user, router]);

  // 首次挂载:从服务端载入会话(异步)
  const hydratedRef = useRef(false);
  useEffect(() => {
    loadConversations()
      .then((list) => {
        // 慢网络下用户可能已在等待期间新建/删除会话:
        // 本地已有变更时不整体覆盖,只合并进服务端未见的会话
        setConversations((prev) =>
          prev.length
            ? [...list.filter((s) => !prev.some((p) => p.id === s.id)), ...prev]
            : list
        );
        setActiveId((cur) => cur ?? (list[0]?.id ?? null));
      })
      .finally(() => {
        hydratedRef.current = true;
      });
  }, []);

  // 持久化:非流式时保存(流式中每 token 改动不落盘,done 时统一存)。
  // hydrated 前不保存:首次挂载时初始空列表会先于异步加载完成,
  // 把 IndexedDB 离线缓存清空(恰好在离线兜底最需要它的时候)
  useEffect(() => {
    if (!streaming && hydratedRef.current) saveConversations(conversations);
  }, [conversations, streaming]);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  // 更新某条消息(函数式,避免闭包旧值)
  const patchMessage = useCallback(
    (convId: string, msgId: string, fn: (m: Message) => Message) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id !== convId
            ? c
            : {
                ...c,
                updatedAt: Date.now(),
                messages: c.messages.map((m) => (m.id === msgId ? fn(m) : m)),
              }
        )
      );
    },
    []
  );

  const newConversation = useCallback(() => {
    // 已有空白对话则直接选用
    const empty = conversations.find((c) => c.messages.length === 0);
    if (empty) {
      setActiveId(empty.id);
      return;
    }
    const c: Conversation = {
      id: newId(),
      title: "新对话",
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
  }, [conversations]);

  const deleteConversation = useCallback(
    (id: string) => {
      deleteConversationServer(id).catch(() => {});
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (activeId === id) {
          setActiveId(next[0]?.id ?? null);
        }
        return next;
      });
    },
    [activeId]
  );

  const renameConversation = useCallback((id: string, title: string) => {
    setConversations((prev) =>
      prev.map((c) => (c.id === id ? { ...c, title } : c))
    );
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setStatus("");
  }, []);

  const send = useCallback(
    (text: string) => {
      if (streaming) return;

      // 确保有活动对话
      let convId = activeId;
      let history: Message[] = [];
      if (!convId) {
        const c: Conversation = {
          id: newId(),
          title: "新对话",
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        convId = c.id;
        setConversations((prev) => [c, ...prev]);
        setActiveId(c.id);
      } else {
        history = active?.messages ?? [];
      }
      const cid = convId;

      const userMsg = mkMessage("user", text);
      const aiMsg = mkMessage("assistant", "");
      aiMsg.streaming = true;

      // 第一条用户消息 -> 设标题
      const isFirst = (active?.messages.length ?? 0) === 0;
      setConversations((prev) =>
        prev.map((c) =>
          c.id !== cid
            ? c
            : {
                ...c,
                title: isFirst ? titleFromMessage(text) : c.title,
                updatedAt: Date.now(),
                messages: [...c.messages, userMsg, aiMsg],
              }
        )
      );

      setStreaming(true);
      setStatus("");
      setTier("");

      const controller = new AbortController();
      abortRef.current = controller;

      streamChat(
        text,
        history,
        {
          onTier: (t) => setTier(t),
          onEscalation: (_from, to, _reason) => {
            // 升级到更高 tier 重跑:清空旧输出,展示深入分析提示
            patchMessage(cid, aiMsg.id, (m) => ({ ...m, content: "" }));
            setTier(to);
            setStatus("正在深入分析…");
          },
          onClarify: (question, options) => {
            // 信息不足:助手反问。正文写入反问句(既展示,也随下一轮 history 上送
            // 保持上下文连贯),clarify 卡片提供可点候选;结束本次生成。
            patchMessage(cid, aiMsg.id, (m) => ({
              ...m,
              content: question,
              clarify: { question, options },
              streaming: false,
            }));
            setStreaming(false);
            setStatus("");
            abortRef.current = null;
          },
          onStatus: (m) => setStatus(m),
          onSources: (items) =>
            patchMessage(cid, aiMsg.id, (m) => ({ ...m, sources: items })),
          onToken: (delta) => {
            setStatus("");
            patchMessage(cid, aiMsg.id, (m) => ({
              ...m,
              content: m.content + delta,
            }));
          },
          onError: (m) => {
            // 错误即收尾:done 是唯一可靠的终端事件,但连接关闭/代理缓冲等
            // 场景下它可能不来;与 onDone 幂等(react 路径 error 后仍会跟 done)。
            patchMessage(cid, aiMsg.id, (msg) => ({
              ...msg,
              error: m,
              streaming: false,
            }));
            setStreaming(false);
            setStatus("");
            abortRef.current = null;
          },
          onDone: () => {
            patchMessage(cid, aiMsg.id, (m) => ({ ...m, streaming: false }));
            setStreaming(false);
            setStatus("");
            abortRef.current = null;
          },
        },
        controller.signal,
        cid
      );
    },
    [streaming, activeId, active, patchMessage]
  );

  // ---- 认证加载中 / 未登录:显示加载页 ----
  if (authLoading || !user) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-main">
        <div className="text-[14px] text-t3">加载中…</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-main">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onSelect={(id) => setActiveId(id)}
        onNew={newConversation}
        onDelete={deleteConversation}
        onRename={renameConversation}
      />
      <main className="flex h-full min-w-0 flex-1 flex-col">
        <TitleBar
          title={active?.title ?? ""}
          onToggleSidebar={() => setSidebarOpen((v) => !v)}
          onNew={newConversation}
        />
        <div className="min-h-0 flex-1">
          <ChatBox
            conversation={active}
            streaming={streaming}
            status={status}
            tier={tier}
            onSend={send}
            onStop={stop}
          />
        </div>
      </main>
    </div>
  );
}
