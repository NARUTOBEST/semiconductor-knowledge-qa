"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Sparkles, Eye, EyeOff } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { BRAND_NAME } from "@/lib/brand";

export default function RegisterPage() {
  const router = useRouter();
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPwd, setShowPwd] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!username.trim()) {
      setError("请输入用户名");
      return;
    }
    if (username.trim().length < 3 || username.trim().length > 20) {
      setError("用户名长度需为 3-20 个字符");
      return;
    }
    if (password.length < 6) {
      setError("密码至少 6 个字符");
      return;
    }
    if (password !== confirm) {
      setError("两次输入的密码不一致");
      return;
    }
    setLoading(true);
    try {
      const resp = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: username.trim(), password }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.detail || "注册失败");
        return;
      }
      login(data.token, { username: data.username, role: data.role });
      router.replace("/");
    } catch {
      setError("网络错误,请检查后端是否运行");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-main">
      <div className="w-full max-w-[380px] px-6">
        <div className="mb-6 flex flex-col items-center">
          <div className="mb-3 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-[#3b82f6] to-[#8b5cf6] shadow-lg shadow-[#3b82f6]/20">
            <Sparkles size={30} className="text-white" />
          </div>
          <h1 className="text-[22px] font-semibold text-t1">注册账号</h1>
          <p className="mt-1 text-[13px] text-t2">创建账号以使用{BRAND_NAME}</p>
        </div>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="用户名(3-20位,字母/数字/下划线)"
            autoFocus
            className="h-11 rounded-xl bg-[#1f1f22] border border-[#2f2f33] px-4 text-[14px] text-t1 placeholder:text-t3 focus:border-[#3a3a3e] focus:outline-none transition-colors"
          />
          <div className="relative">
            <input
              type={showPwd ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="密码(至少6位)"
              className="h-11 w-full rounded-xl bg-[#1f1f22] border border-[#2f2f33] px-4 pr-10 text-[14px] text-t1 placeholder:text-t3 focus:border-[#3a3a3e] focus:outline-none transition-colors"
            />
            <button
              type="button"
              onClick={() => setShowPwd((v) => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-t3 hover:text-t1 transition-colors"
            >
              {showPwd ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
          <input
            type={showPwd ? "text" : "password"}
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            placeholder="确认密码"
            className="h-11 rounded-xl bg-[#1f1f22] border border-[#2f2f33] px-4 text-[14px] text-t1 placeholder:text-t3 focus:border-[#3a3a3e] focus:outline-none transition-colors"
          />
          {error && (
            <div className="rounded-lg border border-[#5b2a2a] bg-[#2a1717] px-3 py-2 text-[13px] text-[#f0a0a0]">
              {error}
            </div>
          )}
          <button
            type="submit"
            disabled={loading}
            className="h-11 rounded-xl bg-gradient-to-r from-[#3b82f6] to-[#6366f1] text-[14px] font-medium text-white hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {loading ? "注册中…" : "注 册"}
          </button>
        </form>
        <p className="mt-4 text-center text-[13px] text-t2">
          已有账号?{" "}
          <button
            onClick={() => router.push("/login")}
            className="text-[#60a5fa] hover:underline"
          >
            登录
          </button>
        </p>
      </div>
    </div>
  );
}
