import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "레일 표면 점검 대시보드",
  description: "B0441 녹·크랙 구역 및 청소 전후 오염률 기록",
  icons: { icon: "/dolphin-mark.png" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
