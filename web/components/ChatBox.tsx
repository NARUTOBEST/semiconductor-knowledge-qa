"use client";

import { useEffect, useRef } from "react";
import { Sparkles, BookOpen, Cpu, AlertCircle } from "lucide-react";
import { MessageItem } from "./MessageItem";
import { Composer } from "./Composer";
import type { Conversation } from "@/lib/types";
import { BRAND_NAME, BRAND_SHORT } from "@/lib/brand";

interface Props {
  conversation: Conversation | null;
  streaming: boolean;
  status: string;            // 检索/加载状态文案
  tier?: string;             // 本次请求所选推理范式 simple|react
  onSend: (text: string) => void;
  onStop: () => void;
}

const TIER_LABEL: Record<string, string> = {
  simple: "快速直答",
  react: "知识问答",
};

const SUGGESTIONS = [
  { icon: AlertCircle, text: "设备出现报警代码时,应按什么步骤排查处理?" },
  { icon: BookOpen, text: "键合机/固晶机的日常保养与点检项目有哪些?" },
  { icon: Cpu, text: "更换密封圈或滤芯的标准操作步骤(SOP)是什么?" },
  { icon: Sparkles, text: "在哪里可以查到某款设备的技术规格参数与手册?" },
];

export function ChatBox({ conversation, streaming, status, tier, onSend, onStop }: Props) {
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
          <h1 className="text-[26px] font-semibold text-t1">{BRAND_NAME}</h1>
          <p className="mt-2 text-[13px] text-t2">
            {BRAND_SHORT} · 基于公司内部设备手册与技术资料检索
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
              <MessageItem key={m.id} msg={m} onClarifyPick={onSend} />
            ))}
            {/* 检索/加载状态(流式中且当前 assistant 还没吐字) */}
            {streaming && status && (
              <div className="flex items-center gap-2 text-[12px] text-t3 animate-fade-in">
                {tier && TIER_LABEL[tier] && (
                  <span className="rounded-full border border-[#3a3a3e] bg-[#27272a] px-2 py-0.5 text-[11px] text-t2">
                    {TIER_LABEL[tier]}
                  </span>
                )}
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
