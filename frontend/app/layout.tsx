import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { themeInitScript } from "../lib/themeScript";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Time Series Studio",
  description: "Advanced Forecasting Interface",
  icons: {
    icon: '/favicon.svg',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
