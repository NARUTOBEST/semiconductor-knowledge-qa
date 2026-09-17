"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { Sparkles, Upload, FileText, Loader2, CheckCircle, XCircle, ArrowLeft } from "lucide-react";

interface Task {
  task_id: string;
  filename: string;
  status: string;
  message: string;
  started_at: number;
}

function authHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

const STATUS_CONFIG: Record<string, { icon: typeof Loader2; color: string; label: string }> = {
  pending:   { icon: Loader2,      color: "text-t2",        label: "排队中" },
  cleaning:  { icon: Loader2,      color: "text-[#60a5fa]", label: "清洗中" },
  ingesting: { icon: Loader2,      color: "text-[#60a5fa]", label: "入库中" },
  done:      { icon: CheckCircle,  color: "text-[#10b981]", label: "完成" },
  error:     { icon: XCircle,      color: "text-[#f0a0a0]", label: "失败" },
};

export default function AdminPage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!loading && (!user || user.role !== "admin")) {
      router.replace("/");
    }
  }, [loading, user, router]);

  const refreshTasks = useCallback(async () => {
    try {
      const resp = await fetch("/api/admin/upload/tasks", { headers: authHeaders() });
      if (resp.ok) setTasks(await resp.json());
    } catch {}
  }, []);

  useEffect(() => {
    if (user?.role === "admin") {
      refreshTasks();
      const timer = setInterval(() => {
        setTasks(prev => {
          if (prev.some(t => ["pending","cleaning","ingesting"].includes(t.status)) || uploading) {
            refreshTasks();
          }
          return prev;
        });
      }, 2000);
      return () => clearInterval(timer);
    }
  }, [user, uploading, refreshTasks]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("仅支持 PDF 文件");
      return;
    }
    setUploading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const resp = await fetch("/api/admin/upload", {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || data.detail || "上传失败");
      } else {
        refreshTasks();
      }
    } catch (e: any) {
      setError(`网络错误: ${e?.message || e}`);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  if (loading || !user) {
    return <div className="flex h-screen items-center justify-center text-t2">加载中...</div>;
  }

  return (
    <div className="flex h-screen flex-col bg-main">
      <header className="flex h-12 items-center gap-3 border-b border-line px-4">
        <button
          onClick={() => router.push("/")}
          className="flex h-8 w-8 items-center justify-center rounded-md text-t4 hover:bg-[#27272a] hover:text-t1"
        >
          <ArrowLeft size={19} />
        </button>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-gradient-to-br from-[#3b82f6] to-[#8b5cf6]">
            <Sparkles size={14} className="text-white" />
          </div>
          <span className="text-[14px] font-medium text-t1">设备文档管理</span>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[640px] px-6 py-8">
          <div className="mb-8">
            <h2 className="mb-3 text-[16px] font-medium text-t1">上传设备文档</h2>
            <p className="mb-4 text-[13px] text-t2">
              上传设备手册、维护规程(SOP)、规格书等 PDF,自动执行:MinerU 清洗 → 切块 → 嵌入 → 入库。
              处理完成后即可检索;其他格式(Word/PPT/视频)可经离线清洗管线处理。
            </p>
            <label
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-[#3a3a3e] bg-[#1f1f22] px-6 py-10 transition-colors hover:border-[#4a4a4e] hover:bg-[#27272a] ${uploading ? "opacity-50" : ""}`}
            >
              {uploading ? (
                <Loader2 size={28} className="mb-2 animate-spin text-t3" />
              ) : (
                <Upload size={28} className="mb-2 text-t3" />
              )}
              <span className="text-[13px] text-t2">
                {uploading ? "上传中..." : "点击选择 PDF 文件"}
              </span>
              <span className="mt-1 text-[11px] text-t3">上限 200MB</span>
              <input
                ref={fileRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={handleUpload}
                disabled={uploading}
              />
            </label>
            {error && (
              <div className="mt-3 rounded-lg border border-[#5b2a2a] bg-[#2a1717] px-3 py-2 text-[13px] text-[#f0a0a0]">
                {error}
              </div>
            )}
          </div>

          <div>
            <h2 className="mb-3 text-[16px] font-medium text-t1">处理记录</h2>
            {tasks.length === 0 ? (
              <p className="py-8 text-center text-[13px] text-t3">暂无上传记录</p>
            ) : (
              <div className="flex flex-col gap-2">
                {tasks.map((task) => {
                  const cfg = STATUS_CONFIG[task.status] || STATUS_CONFIG.pending;
                  const Icon = cfg.icon;
                  const isActive = ["pending","cleaning","ingesting"].includes(task.status);
                  return (
                    <div
                      key={task.task_id}
                      className="flex items-start gap-3 rounded-lg border border-[#2f2f33] bg-[#1f1f22] px-4 py-3"
                    >
                      <FileText size={18} className="mt-0.5 shrink-0 text-t3" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between">
                          <span className="truncate text-[13px] text-t1">{task.filename}</span>
                          <span className={`ml-2 shrink-0 text-[12px] ${cfg.color}`}>
                            <Icon size={13} className={`mr-1 inline ${isActive ? "animate-spin" : ""}`} />
                            {cfg.label}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[12px] text-t2">{task.message}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
