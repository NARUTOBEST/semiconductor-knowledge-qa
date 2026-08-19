import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // 豆包深色主题配色
        sb: "#1c1c1f",        // 侧边栏背景
        main: "#18181b",      // 主区域背景
        card: "#27272a",      // 输入框/代码块/按钮底
        hover: "#2a2a2d",     // 侧边栏 hover
        sel: "#2d2d30",       // 选中态
        line: "#232325",      // 极淡分割线
        accent: "#2563eb",    // 强调蓝(用户气泡)
        codeyellow: "#fbbf24",// 行内代码高亮
        t1: "#e5e5e7",        // 主文字
        t2: "#86868b",        // 次要文字
        t3: "#71717a",        // 弱化文字/图标
        t4: "#a1a1a6",        // 图标默认
      },
      fontSize: {
        // 统一字号
      },
      borderRadius: {
        none: "0px",
      },
      animation: {
        "blink": "blink 1s step-start infinite",
        "fade-in": "fadeIn 0.2s ease-out",
      },
      keyframes: {
        blink: { "50%": { opacity: "0" } },
        fadeIn: { from: { opacity: "0" }, to: { opacity: "1" } },
      },
    },
  },
  plugins: [],
};
export default config;
