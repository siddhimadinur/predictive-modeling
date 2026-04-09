"use client";

import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import { PRESETS, formatCurrency } from "@/lib/constants";
import { PredictionResponse } from "@/lib/types";

interface RegionResult {
  name: string;
  prediction: number | null;
  loading: boolean;
}

export default function ExplorerPage() {
  const [results, setResults] = useState<RegionResult[]>(
    PRESETS.map((p) => ({ name: p.name, prediction: null, loading: false }))
  );
  const [compared, setCompared] = useState(false);

  const compareAll = async () => {
    setResults((prev) => prev.map((r) => ({ ...r, loading: true })));
    setCompared(true);

    const promises = PRESETS.map(async (preset, i) => {
      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...preset.values, model: "gradient_boosting" }),
        });
        const data: PredictionResponse = await res.json();
        return data.prediction;
      } catch {
        return null;
      }
    });

    const predictions = await Promise.all(promises);
    setResults(
      PRESETS.map((p, i) => ({
        name: p.name,
        prediction: predictions[i],
        loading: false,
      }))
    );
  };

  const chartData = results
    .filter((r) => r.prediction !== null)
    .map((r) => ({ name: r.name, price: r.prediction }));

  const scatterData = PRESETS.map((p, i) => ({
    income: p.values.median_income * 10,
    prediction: results[i].prediction,
    name: p.name,
  })).filter((d) => d.prediction !== null);

  return (
    <div className="space-y-8">
      <section className="text-center">
        <h1 className="text-3xl font-extrabold text-slate-900 sm:text-4xl">
          California <span className="gradient-text">Explorer</span>
        </h1>
        <p className="mt-2 text-slate-500">
          Compare predicted housing prices across California regions
        </p>
      </section>

      {/* Region Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {PRESETS.map((preset, i) => (
          <div key={preset.name} className="card-hover p-5">
            <h3 className="font-semibold text-slate-900">{preset.name}</h3>
            <p className="mt-1 text-sm text-slate-500">{preset.description}</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-600">
              <div>
                Income: <span className="font-medium">${preset.values.median_income * 10}K</span>
              </div>
              <div>
                Age: <span className="font-medium">{preset.values.housing_median_age}yr</span>
              </div>
              <div>
                Rooms: <span className="font-medium">{preset.values.ave_rooms}/hh</span>
              </div>
              <div>
                Pop: <span className="font-medium">{preset.values.population.toLocaleString()}</span>
              </div>
            </div>

            {results[i].loading ? (
              <div className="mt-4 flex items-center gap-2 text-sm text-brand-600">
                <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Predicting...
              </div>
            ) : results[i].prediction !== null ? (
              <div className="mt-4">
                <p className="text-2xl font-bold text-brand-600">
                  {formatCurrency(results[i].prediction!)}
                </p>
              </div>
            ) : null}
          </div>
        ))}
      </div>

      {/* Compare Button */}
      <div className="text-center">
        <button
          onClick={compareAll}
          disabled={results.some((r) => r.loading)}
          className="btn-primary px-8"
        >
          {results.some((r) => r.loading)
            ? "Comparing..."
            : compared
            ? "Re-Compare All Regions"
            : "Compare All Regions"}
        </button>
      </div>

      {/* Charts */}
      {chartData.length > 0 && (
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="mb-4 font-semibold text-slate-900">
              Predicted Prices by Region
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="name"
                  tick={{ fontSize: 11 }}
                  angle={-15}
                  textAnchor="end"
                  height={60}
                />
                <YAxis tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`} />
                <Tooltip formatter={(v: number) => formatCurrency(v)} />
                <Bar dataKey="price" fill="#4f46e5" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card p-6">
            <h3 className="mb-4 font-semibold text-slate-900">
              Income vs. Predicted Price
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="income"
                  name="Income"
                  unit="K"
                  tick={{ fontSize: 12 }}
                  label={{ value: "Median Income ($K)", position: "bottom", fontSize: 12 }}
                />
                <YAxis
                  dataKey="prediction"
                  name="Price"
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`}
                  tick={{ fontSize: 12 }}
                />
                <ZAxis range={[100, 100]} />
                <Tooltip
                  formatter={(value: number, name: string) =>
                    name === "Price" ? formatCurrency(value) : `$${value}K`
                  }
                />
                <Scatter data={scatterData} fill="#4f46e5" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
