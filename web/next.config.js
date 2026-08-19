/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 把 /api/* 代理到 Python 后端(8001),前端同源调用,免 CORS,且支持 SSE 流式透传。
  async rewrites() {
    const backend = process.env.BACKEND_URL || "http://127.0.0.1:8001";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
    ];
  },
};

module.exports = nextConfig;
