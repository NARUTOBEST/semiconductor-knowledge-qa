import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import { BRAND_NAME, BRAND_SHORT } from "@/lib/brand";

export const metadata: Metadata = {
  title: BRAND_NAME,
  description: `${BRAND_SHORT}:基于公司内部设备手册、维护规程与技术资料的检索问答`,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="bg-main text-t1 h-screen overflow-hidden">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
