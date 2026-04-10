import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "CalPredict — California Housing Price Predictor",
  description:
    "Predict California housing prices using machine learning. Powered by Gradient Boosting.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased bg-gray-50">
        <Navbar />
        {children}
        <footer className="text-center text-gray-400 text-sm py-8 border-t border-gray-100 mt-16">
          Built with Next.js, scikit-learn & Tailwind CSS · Trained on California Housing Census Data
        </footer>
      </body>
    </html>
  );
}
