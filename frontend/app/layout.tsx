import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sports Identifier — EfficientNetB0",
  description:
    "Upload a sports photograph and identify the sport using a fine-tuned EfficientNetB0 classifier served by Flask.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
