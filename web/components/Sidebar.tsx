"use client";

import { useState } from "react";
import {
  Search,
  Plus,
  MessageSquare,
  Trash2,
  Pencil,
  Check,
  X,
  Sparkles,
  ChevronDown,
} from "lucide-react";
import type { Conversation } from "@/lib/types";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  open: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
}

export function Sidebar({
  conversations,
  activeId,
  open,
  onSelect,
  onNew,
  onDelete,
  onRename,
}: Props) {
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  const filtered = conversations.filter((c) =>
    c.title.toLowerCase().includes(query.trim().toLowerCase())
  );

  const startEdit = (c: Conversation) => {
    setEditingId(c.id);
    setEditText(c.title);
  };
  const commitEdit = () => {
    if (editingId && editText.trim()) {
      onRename(editingId, editText.trim());
    }
    setEditingId(null);
  };

  return (
    <aside
      className={`flex h-full flex-col bg-sb border-r border-line transition-[width] duration-200 ${
        open ? "w-[260px]" : "w-0 overflow-hidden"
      }`}
    >
      {/* 顶部:Logo + 新对话 */}
      <div className="flex items-center justify-between px-3 pt-3 pb-2">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-[#3b82f6] to-[#8b5cf6]">
            <Sparkles size={15} className="text-white" />
          </div>
          <span className="text-[14px] font-medium text-t1">半导体知识助手</span>
        </div>
        <button
          onClick={onNew}
          className="flex h-7 w-7 items-center justify-center rounded-md text-t4 hover:bg-hover hover:text-t1 transition-colors"
          title="新对话"
        >
          <Plus size={18} />
        </button>
      </div>

      {/* 搜索框 */}
      <div className="px-3 pb-2">
        <div className="flex h-9 items-center gap-2 rounded-lg bg-[#1f1f22] px-2.5">
          <Search size={15} className="text-t3" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索对话"
            className="flex-1 bg-transparent text-[13px] text-t1 placeholder:text-t3 focus:outline-none"
          />
          <kbd className="hidden sm:inline rounded border border-[#3a3a3e] px-1.5 py-0.5 text-[10px] text-t3">
            Ctrl K
          </kbd>
        </div>
      </div>

      {/* 历史对话列表 */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        <div className="px-1.5 py-2 text-[12px] font-medium text-t2">
          历史对话
        </div>
        {filtered.length === 0 ? (
          <div className="px-2 py-6 text-center text-[12px] text-t3">
            {query ? "无匹配对话" : "暂无对话,点击上方 + 开始"}
          </div>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {filtered.map((c) => {
              const active = c.id === activeId;
              const editing = editingId === c.id;
              return (
                <li key={c.id}>
                  <div
                    onClick={() => !editing && onSelect(c.id)}
                    onDoubleClick={() => startEdit(c)}
                    className={`group flex h-8 items-center gap-2 rounded-md px-2 cursor-pointer transition-colors ${
                      active ? "bg-sel" : "hover:bg-hover"
                    }`}
                  >
                    <MessageSquare
                      size={15}
                      className={`shrink-0 ${active ? "text-t1" : "text-t3"}`}
                    />
                    {editing ? (
                      <input
                        autoFocus
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") commitEdit();
                          if (e.key === "Escape") setEditingId(null);
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="flex-1 bg-[#1f1f22] rounded px-1.5 py-0.5 text-[13px] text-t1 focus:outline-none"
                      />
                    ) : (
                      <span
                        className={`flex-1 truncate text-[13px] ${
                          active ? "text-t1" : "text-t1/90"
                        }`}
                      >
                        {c.title}
                      </span>
                    )}

                    {/* 操作按钮:hover 显示 */}
                    {editing ? (
                      <div className="flex items-center gap-0.5">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            commitEdit();
                          }}
                          className="p-1 rounded text-t3 hover:text-t1"
                        >
                          <Check size={14} />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingId(null);
                          }}
                          className="p-1 rounded text-t3 hover:text-t1"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            startEdit(c);
                          }}
                          className="p-1 rounded text-t3 hover:bg-[#3f3f46] hover:text-t1"
                          title="重命名"
                        >
                          <Pencil size={13} />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onDelete(c.id);
                          }}
                          className="p-1 rounded text-t3 hover:bg-[#3f3f46] hover:text-[#f0a0a0]"
                          title="删除"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* 底部:账号栏 */}
      <div className="flex h-12 items-center gap-2 border-t border-line px-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-[#10b981] to-[#3b82f6] text-[12px] font-medium text-white">
          学
        </div>
        <span className="flex-1 text-[13px] text-t1">学习者</span>
        <ChevronDown size={16} className="text-t3" />
      </div>
    </aside>
  );
}
