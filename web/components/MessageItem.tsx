"use client";

import { useState } from "react";
import {
  Check,
  Copy,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
  Volume2,
  Sparkles,
  User,
  AlertTriangle,
  FileText,
  HelpCircle,
} from "lucide-react";
import { Markdown } from "./Markdown";
import type { Message } from "@/lib/types";

/** AI 头像:渐变圆 + 星标(代豆包官方头像) */
function AiAvatar() {
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-[#3b82f6] to-[#8b5cf6]">
      <Sparkles size={15} className="text-white" />
    </div>
  );
}

function UserAvatar() {
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#3f3f46]">
      <User size={15} className="text-t2" />
    </div>
  );
}

/** 来源引用条(命中知识库的文档+页码) */
function SourcesBar({ sources }: { sources: NonNullable<Message["sources"]> }) {
  if (!sources.length) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-1.5">
      {sources.slice(0, 6).map((s, i) => (
        <span
          key={i}
          title={`${s.source_stem} ${s.page} · ${s.heading}`}
          className="inline-flex items-center gap-1 rounded-md bg-[#1f1f22] border border-[#2f2f33] px-2 py-0.5 text-[11px] text-t2 hover:text-t1 hover:border-[#3a3a3e] transition-colors"
        >
          <FileText size={11} />
          <span className="max-w-[160px] truncate">{s.source_stem}</span>
          <span className="text-t3">{s.page}</span>
        </span>
      ))}
    </div>
  );
}

/** 澄清反问卡片:信息不足时助手提问,候选可点击直接回复 */
function ClarifyCard({
  clarify,
  onPick,
}: {
  clarify: NonNullable<Message["clarify"]>;
  onPick?: (text: string) => void;
}) {
  return (
    <div className="mb-2 rounded-xl border border-[#3a3a6e] bg-[#1b1f3a] px-3.5 py-3">
      <div className="flex items-start gap-2 text-[13px] text-t1">
        <HelpCircle size={15} className="mt-0.5 shrink-0 text-[#8b9cf6]" />
        <span>{clarify.question}</span>
      </div>
      {clarify.options.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-2 pl-6">
          {clarify.options.map((opt, i) => (
            <button
              key={i}
              onClick={() => onPick?.(opt)}
              className="rounded-lg border border-[#3a3a6e] bg-[#23284a] px-3 py-1.5 text-[12px] text-t2 transition-colors hover:border-[#5b67d6] hover:bg-[#2b3160] hover:text-t1"
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/** AI 消息底部操作按钮栏 */
function ActionRow({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };
  const btn =
    "p-1.5 rounded-md text-t3 hover:text-t1 hover:bg-[#27272a] transition-colors";
  return (
    <div className="mt-2 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
      <button className={btn} onClick={copy} title="复制">
        {copied ? <Check size={15} /> : <Copy size={15} />}
      </button>
      <button className={btn} title="朗读">
        <Volume2 size={15} />
      </button>
      <button className={btn} title="赞">
        <ThumbsUp size={15} />
      </button>
      <button className={btn} title="踩">
        <ThumbsDown size={15} />
      </button>
      <button className={btn} title="重新生成">
        <RefreshCw size={15} />
      </button>
    </div>
  );
}

export function MessageItem(
  { msg, onClarifyPick }: { msg: Message; onClarifyPick?: (text: string) => void }
) {
  if (msg.role === "user") {
    // 用户消息:气泡在左,头像在右(豆包风格)
    return (
      <div className="flex justify-end gap-3 animate-fade-in">
        <div className="max-w-[75%] rounded-xl rounded-tr-sm bg-accent px-3.5 py-2.5 text-[14px] leading-relaxed text-white whitespace-pre-wrap break-words">
          {msg.content}
        </div>
        <UserAvatar />
      </div>
    );
  }

  // AI 消息:头像在左,内容直接渲染在底色上(无气泡)
  return (
    <div className="group flex gap-3 animate-fade-in">
      <AiAvatar />
      <div className="min-w-0 flex-1">
        {msg.sources && msg.sources.length > 0 && (
          <SourcesBar sources={msg.sources} />
        )}
        {msg.error ? (
          <div className="flex items-start gap-2 rounded-lg border border-[#5b2a2a] bg-[#2a1717] px-3 py-2 text-[13px] text-[#f0a0a0]">
            <AlertTriangle size={15} className="mt-0.5 shrink-0" />
            <span>{msg.error}</span>
          </div>
        ) : msg.clarify ? (
          <ClarifyCard clarify={msg.clarify} onPick={onClarifyPick} />
        ) : (
          <div className="relative">
            <Markdown content={msg.content} />
            {msg.streaming && <span className="stream-cursor" />}
          </div>
        )}
        {!msg.streaming && msg.content && !msg.error && !msg.clarify && (
          <ActionRow content={msg.content} />
        )}
      </div>
    </div>
  );
}
