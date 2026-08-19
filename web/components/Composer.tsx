"use client";

import { useRef, useEffect, useState } from "react";
import { ArrowUp, Square, Paperclip, Mic, Globe, Lightbulb, Image as ImageIcon } from "lucide-react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  streaming: boolean;
  disabled?: boolean;
}

const SKILLS = [
  { icon: Globe, label: "联网搜索" },
  { icon: Lightbulb, label: "深度思考" },
  { icon: ImageIcon, label: "图像生成" },
];

export function Composer({ onSend, onStop, streaming, disabled }: Props) {
  const [text, setText] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  // 自适应高度
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || streaming || disabled) return;
    onSend(t);
    setText("");
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="px-[80px] pb-4 pt-2">
      <div className="mx-auto max-w-[820px]">
        <div className="rounded-2xl bg-card border border-transparent focus-within:border-[#3a3a3e] transition-colors">
          {/* 输入区 */}
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="给半导体知识助手发消息…  (Enter 发送 / Shift+Enter 换行)"
            className="block w-full resize-none bg-transparent px-4 pt-3 text-[14px] leading-relaxed text-t1 placeholder:text-t3 focus:outline-none"
            style={{ minHeight: 24 }}
          />
          {/* 功能按钮栏 */}
          <div className="flex items-center gap-1 px-2.5 pb-2.5 pt-1.5">
            <button
              className="flex h-7 w-7 items-center justify-center rounded-md text-t4 hover:bg-[#3f3f46] hover:text-t1 transition-colors"
              title="添加附件"
            >
              <Paperclip size={18} />
            </button>
            {SKILLS.map((s) => (
              <button
                key={s.label}
                className="flex items-center gap-1 rounded-md px-2 h-7 text-[12px] text-t4 hover:bg-[#3f3f46] hover:text-t1 transition-colors"
                title={s.label}
              >
                <s.icon size={14} />
                {s.label}
              </button>
            ))}
            <div className="flex-1" />
            {streaming ? (
              <button
                onClick={onStop}
                className="flex h-8 w-8 items-center justify-center rounded-full bg-[#3f3f46] text-t1 hover:bg-[#52525b] transition-colors"
                title="停止生成"
              >
                <Square size={14} className="fill-current" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!text.trim() || disabled}
                className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-white disabled:bg-[#3a3a3e] disabled:text-t3 transition-colors hover:bg-[#1d4ed8]"
                title="发送"
              >
                <ArrowUp size={18} />
              </button>
            )}
          </div>
        </div>
        <div className="mt-1.5 text-center text-[11px] text-t3">
          半导体知识助手基于本地 ALD 知识库检索,内容仅供参考,请注意核实。
        </div>
      </div>
    </div>
  );
}
