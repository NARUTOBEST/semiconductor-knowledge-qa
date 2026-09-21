// 类型定义:会话与消息

export type Role = "user" | "assistant";

export interface Source {
  source_stem: string;
  page: string;          // "p12" 或 "p12-14"
  heading: string;
  score: number;
  content: string;       // 摘录前 160 字
  // 多媒体透传(后端仅在有可访问 URL 时下发,均可选)
  image_url?: string;    // 图像块/文本块内嵌图的签名链接 → 缩略图,点击放大
  video_url?: string;    // 视频块签名链接 → 视频入口
  item_type?: string;    // 图像块类型标签(如 portrait/figure)
  description?: string;  // 图像描述(caption+description 前 160 字)
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
