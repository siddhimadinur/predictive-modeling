import type { Metadata } from "next";
import Navbar from "@/components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "CalPredict — California Housing Price Predictor",
  description:
    "Predict California housing prices using machine learning. Powered by Gradient Boosting with 97.8% accuracy.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans">
        <Navbar />
        <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">{children}</main>
        <footer className="border-t border-slate-200/60 bg-white mt-16">
          <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
            <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
              <div className="flex items-center gap-2">
                <div className="flex h-7 w-7 items-center justify-center rounded-md bg-brand-600 text-white text-xs font-bold">
                  CP
                </div>
                <span className="text-sm font-semibold text-slate-700">CalPredict</span>
              </div>
              <p className="text-xs text-slate-500">
                Predictions are estimates based on historical data. Not financial advice.
              </p>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
