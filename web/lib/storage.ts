// 会话持久化:服务端 SQLite 优先 + IndexedDB 缓存

import type { Conversation } from "./types";

// ============ IndexedDB 缓存(替代 localStorage,容量无上限)============
const DB_NAME = "semi_agent_db";
const STORE_NAME = "conversations";
const DB_VERSION = 1;

/** 打开(或创建)IndexedDB。 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE_NAME)) {
        req.result.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** 写入 IndexedDB 缓存(异步,存全部会话)。 */
async function cacheToDB(list: Conversation[]): Promise<void> {
  if (typeof window === "undefined") return;
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).clear();            // 清旧数据
    tx.objectStore(STORE_NAME).put(list, "all");   // 存全部(一个 key)
    tx.oncomplete = () => db.close();
  } catch {
    /* IndexedDB 不可用,忽略(服务端有备份) */
  }
}

/** 从 IndexedDB 读取缓存(异步,离线兜底)。 */
async function loadFromDB(): Promise<Conversation[]> {
  if (typeof window === "undefined") return [];
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get("all");
      req.onsuccess = () => {
        db.close();
        const arr = req.result as Conversation[] | undefined;
        if (!arr || !Array.isArray(arr)) return resolve([]);
        resolve(
          arr.map((c) => ({
            ...c,
            messages: c.messages.map((m) => ({ ...m, streaming: false })),
          }))
        );
      };
      req.onerror = () => {
        db.close();
        resolve([]);
      };
    });
  } catch {
    return [];
  }
}

// ============ 服务端 API ============

/** 从 localStorage 取 token(token 太小,留在 localStorage)。 */
function authHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** 从服务端加载全部会话(含消息)。 */
async function apiGetConversations(): Promise<Conversation[]> {
  const resp = await fetch("/api/conversations", {
    headers: authHeaders(),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

/** 批量同步会话到服务端(upsert)。 */
async function apiSyncConversations(list: Conversation[]): Promise<void> {
  const resp = await fetch("/api/conversations", {
    method: "PUT",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(list),
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
}

/** 从服务端删除一条会话。 */
export async function deleteConversationServer(id: string): Promise<void> {
  await fetch(`/api/conversations/${id}`, {
    method: "DELETE",
    headers: authHeaders(),
  });
}

// ============ 对外接口 ============

/**
 * 加载会话:服务端优先,IndexedDB 兜底。
 * 服务端不可用时用本地缓存,保证离线可用。
 */
export async function loadConversations(): Promise<Conversation[]> {
  try {
    const serverList = await apiGetConversations();
    cacheToDB(serverList);  // 异步写缓存,不等
    return serverList;
  } catch {
    return await loadFromDB();  // 服务端挂了 -> 读 IndexedDB 缓存
  }
}

/** 防抖同步定时器 */
let syncTimer: ReturnType<typeof setTimeout> | null = null;

/**
 * 保存会话:立即写 IndexedDB + 1 秒防抖同步到服务端。
 * 服务端不可用时静默失败(本地缓存仍在)。
 */
export function saveConversations(list: Conversation[]): void {
  // 立即写 IndexedDB(异步,不阻塞)
  cacheToDB(list);

  // 防抖同步到服务端(1 秒内的多次修改合并为一次请求)
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(() => {
    apiSyncConversations(list).catch(() => {
      /* 服务端不可用,本地缓存已有,不阻塞用户 */
    });
  }, 1000);
}

export function newId(): string {
  return (
    Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
  );
}

// 用第一句用户消息生成会话标题(截断 20 字)
export function titleFromMessage(text: string): string {
  const t = text.trim().replace(/\s+/g, " ");
  return t.length > 20 ? t.slice(0, 20) + "…" : t || "新对话";
}
