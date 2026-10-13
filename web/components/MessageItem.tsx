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
  Film,
  Play,
  HelpCircle,
  ExternalLink,
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

/** 图片放大浮层:点击遮罩或图片关闭 */
function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-8 cursor-zoom-out"
      onClick={onClose}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt="来源图片放大"
        className="max-h-full max-w-full rounded-lg object-contain"
        onClick={onClose}
      />
    </div>
  );
}

/** 视频播放浮层:<video> 标签不走导航,不受对象存储 Content-Disposition: attachment
 *  强制下载头影响(新标签页打开签名链接会被浏览器当附件下载,故必须应用内播放) */
function VideoLightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-8"
      onClick={onClose}
    >
      <div
        className="flex max-h-full w-full max-w-4xl flex-col gap-2"
        onClick={(e) => e.stopPropagation()}
      >
        <video
          src={src}
          controls
          autoPlay
          playsInline
          className="max-h-[80vh] w-full rounded-lg bg-black"
        />
        <div className="flex justify-center">
          <a
            href={src}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md border border-[#3a3a3e] bg-[#1f1f22] px-3 py-1 text-[12px] text-t2 hover:text-t1 hover:border-[#5b67d6]"
            onClick={(e) => e.stopPropagation()}
          >
            下载视频
          </a>
        </div>
      </div>
    </div>
  );
}

/** 来源引用条(命中知识库的文档+页码;图像块渲染缩略图,视频块渲染入口) */
function SourcesBar({ sources }: { sources: NonNullable<Message["sources"]> }) {
  const [zoom, setZoom] = useState<string | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  if (!sources.length) return null;
  return (
    <>
      <div className="mb-2 flex flex-wrap gap-1.5">
        {sources.slice(0, 6).map((s, i) => {
          const tip = `${s.source_stem} ${s.page} · ${s.heading}${s.description ? ` · ${s.description}` : ""}`;
          // 图像块(TOS 签名链接):缩略图,点击放大
          if (s.image_url && /^https?:/.test(s.image_url)) {
            return (
              <button
                key={i}
                title={tip}
                onClick={() => setZoom(s.image_url!)}
                className="group/img relative overflow-hidden rounded-md border border-[#2f2f33] transition-colors hover:border-[#5b67d6]"
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={s.image_url}
                  alt={s.heading || s.source_stem}
                  className="h-16 w-24 object-cover"
                  loading="lazy"
                />
                <span className="absolute inset-x-0 bottom-0 truncate bg-black/60 px-1 py-0.5 text-[10px] text-white/90">
                  {s.source_stem}
                </span>
              </button>
            );
          }
          // 视频块:应用内播放入口(点击弹出播放浮层;新标签页会被强制下载,见 VideoLightbox)
          if (s.video_url && /^https?:/.test(s.video_url)) {
            return (
              <button
                key={i}
                onClick={() => setVideo(s.video_url!)}
                title={tip}
                className="inline-flex items-center gap-1 rounded-md bg-[#1f1f22] border border-[#2f2f33] px-2 py-0.5 text-[11px] text-t2 hover:text-t1 hover:border-[#5b67d6] transition-colors"
              >
                <Film size={11} className="text-[#8b9cf6]" />
                <span className="max-w-[160px] truncate">{s.source_stem}</span>
                <span className="text-t3">{s.page}</span>
                <Play size={9} className="text-t3" />
              </button>
            );
          }
          // 普通文档块:有原文档直链时可点击在线打开(带 #page= 定位),否则原样式
          const docHref = s.doc_url
            ? s.doc_url + (s.doc_page ? `#page=${s.doc_page}` : "")
            : null;
          const chip = (
            <>
              <FileText size={11} />
              <span className="max-w-[160px] truncate">{s.source_stem}</span>
              <span className="text-t3">{s.page}</span>
              {docHref && <ExternalLink size={9} className="text-t3" />}
            </>
          );
          if (docHref) {
            return (
              <a
                key={i}
                href={docHref}
                target="_blank"
                rel="noopener noreferrer"
                title={`${tip} · 点击在线查看原文档`}
                className="inline-flex items-center gap-1 rounded-md bg-[#1f1f22] border border-[#2f2f33] px-2 py-0.5 text-[11px] text-t2 hover:text-t1 hover:border-[#5b67d6] transition-colors"
              >
                {chip}
              </a>
            );
          }
          return (
            <span
              key={i}
              title={tip}
              className="inline-flex items-center gap-1 rounded-md bg-[#1f1f22] border border-[#2f2f33] px-2 py-0.5 text-[11px] text-t2 hover:text-t1 hover:border-[#3a3a3e] transition-colors"
            >
              {chip}
            </span>
          );
        })}
      </div>
      {zoom && <Lightbox src={zoom} onClose={() => setZoom(null)} />}
      {video && <VideoLightbox src={video} onClose={() => setVideo(null)} />}
    </>
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
      {clarify.options.length > 0 ? (
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
      ) : (
        // 无选项口径:请用户在输入框直接说明对象(设备名/型号)
        <div className="mt-1.5 pl-6 text-[12px] text-t3">
          请在下方输入框直接说明设备名称或型号
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
  // 正文引用 [文档名 pN] → 原文档直链映射(stem → doc_url + #page 锚点),
  // 供 Markdown 把可点击的引用转成链接
  const citeLinks: Record<string, string> = {};
  for (const s of msg.sources ?? []) {
    if (s.doc_url) {
      citeLinks[s.source_stem] =
        s.doc_url + (s.doc_page ? `#page=${s.doc_page}` : "");
    }
  }
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
            <Markdown content={msg.content} citeLinks={citeLinks} />
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
