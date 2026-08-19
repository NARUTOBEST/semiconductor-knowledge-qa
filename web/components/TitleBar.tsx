"use client";

import {
  PanelLeft,
  Plus,
  LogOut,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";

interface Props {
  title: string;
  onToggleSidebar: () => void;
  onNew: () => void;
}

export function TitleBar({ title, onToggleSidebar, onNew }: Props) {
  const { user, logout } = useAuth();
  const router = useRouter();
  const sideBtn =
    "flex h-8 w-8 items-center justify-center rounded-md text-t4 hover:bg-[#27272a] hover:text-t1 transition-colors";

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };

  return (
    <header className="relative flex h-12 items-center justify-between px-3">
      {/* 左:侧栏切换 + 新对话 */}
      <div className="flex items-center gap-0.5">
        <button className={sideBtn} onClick={onToggleSidebar} title="历史对话">
          <PanelLeft size={19} />
        </button>
        <button className={sideBtn} onClick={onNew} title="新对话">
          <Plus size={19} />
        </button>
      </div>

      {/* 中:标题 + 副标题(居中) */}
      <div className="pointer-events-none absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center">
        <span className="max-w-[420px] truncate text-[14px] font-medium text-t1">
          {title || "半导体知识助手"}
        </span>
        <span className="text-[11px] text-t2">
          AI 生成可能有误,注意核实
        </span>
      </div>

      {/* 右:用户名 + 登出 */}
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-t2">{user?.username}</span>
        <button className={sideBtn} onClick={handleLogout} title="退出登录">
          <LogOut size={18} />
        </button>
      </div>
    </header>
  );
}
