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

/** 把正文里的来源引用 [文档名 pN] / [文档名] 转成 Markdown 链接。
 *  仅当文档名精确命中来源的 source_stem(有原文档直链)才转,避免误链。
 *  链接目标用 <url> 角括号包裹(签名 URL 长,防解析截断);链接文本转义
 *  Markdown 特殊字符,保持与原引用显示一致。 */
function linkifyCites(content: string, citeLinks?: Record<string, string>) {
  if (!citeLinks || Object.keys(citeLinks).length === 0) return content;
  const esc = (s: string) => s.replace(/([_[\]*])/g, "\\$1");
  return content.replace(
    /\[([^\[\]\n]+?)(?:\s+(p\d+(?:[-–]\d+)?))?\]/g,
    (m, stem: string, pg?: string) => {
      const href = citeLinks[stem];
      return href ? `[${esc(stem)}${pg ? " " + pg : ""}](<${href}>)` : m;
    },
  );
}

function MarkdownImpl({
  content,
  citeLinks,
}: {
  content: string;
  citeLinks?: Record<string, string>;
}) {
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
        {linkifyCites(content, citeLinks)}
      </ReactMarkdown>
    </div>
  );
}

export const Markdown = memo(MarkdownImpl);
