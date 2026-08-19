"use client";

import { useState, memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy } from "lucide-react";

/** 代码块:带语言标识 + 复制按钮(豆包风格) */
function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };
  return (
    <div className="my-3 overflow-hidden rounded-lg bg-[#1e1e1e] border border-[#2f2f33]">
      <div className="flex items-center justify-between px-3 py-1.5 text-[12px] text-t3 border-b border-[#2f2f33]">
        <span className="font-mono">{lang || "text"}</span>
        <button
          onClick={onCopy}
          className="flex items-center gap-1 hover:text-t1 transition-colors"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-[13px] leading-6 text-t1 font-mono">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MarkdownImpl({ content }: { content: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // 代码块:抽取语言与文本,渲染带复制按钮的块
          pre({ children }) {
            const codeEl: any = Array.isArray(children) ? children[0] : children;
            const className: string = codeEl?.props?.className || "";
            const lang = /language-(\w+)/.exec(className)?.[1] || "";
            const raw = codeEl?.props?.children;
            const code = Array.isArray(raw)
              ? raw.join("")
              : String(raw ?? "").replace(/\n$/, "");
            return <CodeBlock lang={lang} code={code} />;
          },
          // 外链新窗口打开
          a({ children, href }) {
            return (
              <a href={href} target="_blank" rel="noreferrer noopener">
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
