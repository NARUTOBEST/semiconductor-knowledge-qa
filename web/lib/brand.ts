// 品牌/产品名集中定义。默认中性名,部署时可用环境变量覆盖为具体公司名:
//   NEXT_PUBLIC_BRAND_NAME="XX 半导体设备知识库"
//   NEXT_PUBLIC_BRAND_SHORT="设备知识库"
// Next.js 仅把 NEXT_PUBLIC_ 前缀的变量内联到前端。
export const BRAND_NAME =
  process.env.NEXT_PUBLIC_BRAND_NAME?.trim() || "半导体设备知识问答系统";

export const BRAND_SHORT =
  process.env.NEXT_PUBLIC_BRAND_SHORT?.trim() || "设备知识库助手";
