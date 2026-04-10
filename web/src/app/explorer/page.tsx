"use client";

import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { CALIFORNIA_CITIES } from "@/lib/cities";
import { formatCurrency } from "@/lib/constants";

const INFLATION = 4.5;

interface CityResult {
  name: string;
  prediction: number | null;
  loading: boolean;
}

export default function ExplorerPage() {
  const [results, setResults] = useState<CityResult[]>([]);
  const [compared, setCompared] = useState(false);

  const cities = Object.keys(CALIFORNIA_CITIES);

  const compareAll = async () => {
    setResults(cities.map((c) => ({ name: c, prediction: null, loading: true })));
    setCompared(true);

    try {
      const res = await fetch("/predictions.json");
      const predictions: Record<string, number> = await res.json();

      const cityResults = cities.map((city) => {
        const key = `${city}|4|2|3`;
        const pred = predictions[key];
        return {
          name: city,
          prediction: pred ? Math.round(pred * INFLATION) : null,
          loading: false,
        };
      });

      setResults(cityResults);
    } catch {
      setResults(cities.map((c) => ({ name: c, prediction: null, loading: false })));
    }
  };

  const chartData = results
    .filter((r) => r.prediction !== null)
    .sort((a, b) => (b.prediction ?? 0) - (a.prediction ?? 0))
    .map((r) => ({ name: r.name, price: r.prediction }));

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      <section className="text-center">
        <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
          California <span className="text-transparent bg-clip-text bg-gradient-to-r from-sunset-500 to-sunset-400">Explorer</span>
        </h1>
        <p className="mt-2 text-gray-500">Compare predicted housing prices across 20 California cities</p>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cities.slice(0, 8).map((city) => {
          const data = CALIFORNIA_CITIES[city];
          const result = results.find((r) => r.name === city);
          return (
            <div key={city} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4 hover:shadow-md transition-shadow">
              <h3 className="font-semibold text-gray-900">{city}</h3>
              <div className="mt-2 grid grid-cols-2 gap-1 text-xs text-gray-500">
                <div>Income: <span className="font-medium text-gray-700">${(data.median_income * 10).toFixed(0)}K</span></div>
                <div>Age: <span className="font-medium text-gray-700">{data.housing_median_age}yr</span></div>
              </div>
              {result?.loading ? (
                <div className="mt-3 text-sm text-sunset-500 animate-pulse">Predicting...</div>
              ) : result?.prediction ? (
                <p className="mt-3 text-xl font-bold text-sunset-500">{formatCurrency(result.prediction)}</p>
              ) : null}
            </div>
          );
        })}
      </div>

      <div className="text-center">
        <button onClick={compareAll} className="inline-flex items-center justify-center rounded-lg bg-sunset-500 px-8 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-sunset-600 transition-colors">
          {compared ? "Re-Compare All Cities" : "Compare All Cities"}
        </button>
      </div>

      {chartData.length > 0 && (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h3 className="mb-4 font-semibold text-gray-900">Predicted Prices by City (2024 Estimate)</h3>
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis type="number" tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`} />
              <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v: number) => formatCurrency(v)} />
              <Bar dataKey="price" fill="#E85D26" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
