"use client";

import { useEffect, useRef } from "react";
import { Sparkles, BookOpen, FlaskConical, Cpu, AlertCircle } from "lucide-react";
import { MessageItem } from "./MessageItem";
import { Composer } from "./Composer";
import type { Conversation } from "@/lib/types";

interface Props {
  conversation: Conversation | null;
  streaming: boolean;
  status: string;            // 检索/加载状态文案
  onSend: (text: string) => void;
  onStop: () => void;
}

const SUGGESTIONS = [
  { icon: BookOpen, text: "ALD 原子层沉积的基本原理是什么?" },
  { icon: FlaskConical, text: "TMA(三甲基铝)前驱体有哪些安全注意事项?" },
  { icon: Cpu, text: "wafer chuck 的温度控制如何实现?" },
  { icon: Sparkles, text: "ALD 与 CVD 的主要区别?" },
];

export function ChatBox({ conversation, streaming, status, onSend, onStop }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const messages = conversation?.messages ?? [];

  // 流式 token 到达时只在"用户本来就贴着底部"时才自动滚动:
  // 否则长回答生成期间用户向上翻阅会被每个 token 强行拽回底部
  useEffect(() => {
    if (stickToBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
    }
  }, [messages.length, messages[messages.length - 1]?.content, status]);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  // 新消息加入(用户发送/收到新 assistant 消息)时恢复贴底跟随
  const prevCountRef = useRef(messages.length);
  useEffect(() => {
    if (messages.length > prevCountRef.current) {
      stickToBottomRef.current = true;
    }
    prevCountRef.current = messages.length;
  }, [messages.length]);

  // 空对话:欢迎页 + 建议问题
  if (!conversation || messages.length === 0) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex flex-1 flex-col items-center justify-center px-6">
          <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#3b82f6] to-[#8b5cf6] shadow-lg shadow-[#3b82f6]/20">
            <Sparkles size={30} className="text-white" />
          </div>
          <h1 className="text-[26px] font-semibold text-t1">半导体知识助手</h1>
          <p className="mt-2 text-[13px] text-t2">
            半导体设备与工艺学习问答 · 基于 ALD 知识库检索增强
          </p>
          <div className="mt-8 grid w-full max-w-[640px] grid-cols-1 gap-2.5 sm:grid-cols-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s.text}
                onClick={() => onSend(s.text)}
                className="group flex items-center gap-2.5 rounded-xl border border-[#2f2f33] bg-[#1f1f22] px-3.5 py-3 text-left text-[13px] text-t2 hover:border-[#3a3a3e] hover:bg-[#27272a] hover:text-t1 transition-colors"
              >
                <s.icon size={16} className="shrink-0 text-t3 group-hover:text-[#60a5fa]" />
                <span className="line-clamp-2">{s.text}</span>
              </button>
            ))}
          </div>
        </div>
        <Composer onSend={onSend} onStop={onStop} streaming={streaming} />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* 消息流 */}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[820px] px-[80px] py-6">
          <div className="flex flex-col gap-6">
            {messages.map((m) => (
              <MessageItem key={m.id} msg={m} />
            ))}
            {/* 检索/加载状态(流式中且当前 assistant 还没吐字) */}
            {streaming && status && (
              <div className="flex items-center gap-2 text-[12px] text-t3 animate-fade-in">
                <span className="flex gap-1">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-t3 [animation-delay:0ms]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-t3 [animation-delay:150ms]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-t3 [animation-delay:300ms]" />
                </span>
                {status}
              </div>
            )}
          </div>
          <div ref={bottomRef} className="h-1" />
        </div>
      </div>
      <Composer onSend={onSend} onStop={onStop} streaming={streaming} />
    </div>
  );
}
