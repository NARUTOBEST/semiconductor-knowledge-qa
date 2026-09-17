// 类型定义:会话与消息

export type Role = "user" | "assistant";

export interface Source {
  source_stem: string;
  page: string;          // "p12" 或 "p12-14"
  heading: string;
  score: number;
  content: string;       // 摘录前 160 字
}

export interface Clarify {
  question: string;      // 助手反问的问题
  options: string[];     // 可选候选答案(可空)
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  sources?: Source[];    // 仅 assistant 有:命中的知识库来源
  clarify?: Clarify;     // 仅 assistant 有:信息不足时的反问卡片
  streaming?: boolean;   // 是否正在流式生成
  error?: string;
  createdAt: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}
