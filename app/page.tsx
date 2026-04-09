"use client";

import { useState } from "react";
import PredictionForm from "@/components/PredictionForm";
import PredictionResult from "@/components/PredictionResult";
import { HousingInput, PredictionResponse } from "@/lib/types";
import { MODEL_METRICS } from "@/lib/constants";

export default function PredictPage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastInput, setLastInput] = useState<HousingInput | null>(null);

  const handlePredict = async (values: HousingInput, model: string) => {
    setLoading(true);
    setError(null);
    setLastInput(values);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...values, model }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Prediction failed");
      }

      const data: PredictionResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-10">
      {/* Hero */}
      <section className="text-center">
        <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl">
          California Housing <span className="gradient-text">Price Predictor</span>
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-lg text-slate-500">
          Predict home values across California using machine learning trained on
          real census data.
        </p>
        <div className="mx-auto mt-6 flex max-w-lg justify-center gap-6">
          {[
            { value: "3", label: "ML Models" },
            { value: "16,512", label: "Training Samples" },
            { value: "97.8%", label: "R² Accuracy" },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <p className="text-2xl font-bold text-brand-600">{s.value}</p>
              <p className="text-xs text-slate-500">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Main content */}
      <div className="grid gap-8 lg:grid-cols-5">
        {/* Form — wider */}
        <div className="lg:col-span-3">
          <div className="card p-6">
            <h2 className="mb-5 text-lg font-bold text-slate-900">
              Property Details
            </h2>
            <PredictionForm onSubmit={handlePredict} loading={loading} />
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-2">
          {error && (
            <div className="card border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
              {error}
            </div>
          )}

          {result && lastInput ? (
            <PredictionResult
              result={result}
              income={lastInput.median_income}
              latitude={lastInput.latitude}
            />
          ) : (
            <div className="card flex flex-col items-center justify-center px-6 py-16 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-brand-50">
                <svg className="h-8 w-8 text-brand-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3.75h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008z" />
                </svg>
              </div>
              <p className="font-semibold text-slate-700">No prediction yet</p>
              <p className="mt-1 text-sm text-slate-500">
                Fill in the property details and click Predict Price
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
