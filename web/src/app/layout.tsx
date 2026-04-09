import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "California Housing Price Predictor",
  description:
    "AI-powered property valuation for the Golden State using machine learning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
