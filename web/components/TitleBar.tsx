"use client";

import {
  PanelLeft,
  Plus,
  LogOut,
  UserX,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import { BRAND_NAME } from "@/lib/brand";

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

  // 账号注销:永久删除账号 + 全部会话/长期记忆/短期流水。需密码二次确认。
  const handleDeleteAccount = async () => {
    if (
      !window.confirm(
        "注销将永久删除你的账号,以及全部会话记录和长期记忆数据,且不可恢复。\n\n确定要注销吗?"
      )
    ) {
      return;
    }
    const password = window.prompt("请输入登录密码以确认注销:");
    if (password === null) return; // 用户取消
    try {
      const token = localStorage.getItem("token");
      const resp = await fetch("/api/auth/me", {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ password }),
      });
      if (resp.status === 401) {
        alert("密码错误,注销未执行。");
        return;
      }
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        alert("注销失败:" + (data.detail || `HTTP ${resp.status}`));
        return;
      }
      logout();
      router.replace("/login");
    } catch {
      alert("网络异常,注销未完成,请重试。");
    }
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
          {title || BRAND_NAME}
        </span>
        <span className="text-[11px] text-t2">
          AI 生成可能有误,注意核实
        </span>
      </div>

      {/* 右:用户名 + 注销账号 + 登出 */}
      <div className="flex items-center gap-2">
        <span className="text-[12px] text-t2">{user?.username}</span>
        <button
          className={sideBtn}
          onClick={handleDeleteAccount}
          title="注销账号(永久删除账号与全部数据)"
        >
          <UserX size={18} />
        </button>
        <button className={sideBtn} onClick={handleLogout} title="退出登录">
          <LogOut size={18} />
        </button>
      </div>
    </header>
  );
}
