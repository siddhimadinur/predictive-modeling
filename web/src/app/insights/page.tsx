"use client";

import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { MODEL_METRICS, formatCurrency } from "@/lib/constants";

const FEATURE_IMPORTANCE = [
  { feature: "Median Income", importance: 0.52 },
  { feature: "Longitude", importance: 0.14 },
  { feature: "Latitude", importance: 0.13 },
  { feature: "Avg Occupancy", importance: 0.06 },
  { feature: "Housing Age", importance: 0.05 },
  { feature: "Avg Rooms", importance: 0.04 },
  { feature: "Population", importance: 0.03 },
  { feature: "Avg Bedrooms", importance: 0.03 },
];

const TABS = ["Performance", "Feature Importance", "Market Insights"] as const;

export default function InsightsPage() {
  const [activeTab, setActiveTab] = useState<(typeof TABS)[number]>("Performance");

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
      <section className="text-center">
        <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
          Model <span className="text-transparent bg-clip-text bg-gradient-to-r from-sunset-500 to-sunset-400">Insights</span>
        </h1>
        <p className="mt-2 text-gray-500">How our models perform and what drives predictions</p>
      </section>

      <div className="flex justify-center gap-1 rounded-xl bg-gray-100 p-1">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`rounded-lg px-5 py-2 text-sm font-medium transition-all ${
              activeTab === tab
                ? "bg-white text-sunset-700 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === "Performance" && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-3">
            {MODEL_METRICS.map((m) => (
              <div key={m.name} className={`bg-white rounded-2xl border p-5 shadow-sm ${m.is_champion ? "ring-2 ring-sunset-500/30 border-sunset-200" : "border-gray-100"}`}>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-gray-900">{m.display_name}</h3>
                  {m.is_champion && (
                    <span className="rounded-full bg-sunset-500 px-2 py-0.5 text-[10px] font-bold uppercase text-white">Champion</span>
                  )}
                </div>
                <div className="mt-4 space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">R² Score</span>
                    <span className="font-semibold">{(m.val_r2 * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-1.5 w-full rounded-full bg-gray-100">
                    <div className="h-full rounded-full bg-sunset-500" style={{ width: `${m.val_r2 * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">RMSE</span>
                    <span className="font-medium">{formatCurrency(m.val_rmse)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">MAE</span>
                    <span className="font-medium">{formatCurrency(m.val_mae)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h3 className="mb-4 font-semibold text-gray-900">Model Error Comparison (RMSE)</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={MODEL_METRICS.map((m) => ({ name: m.display_name, RMSE: m.val_rmse }))} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tickFormatter={(v) => `$${(v / 1000).toFixed(0)}K`} />
                <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 13 }} />
                <Tooltip formatter={(v: number) => formatCurrency(v)} />
                <Bar dataKey="RMSE" fill="#E85D26" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === "Feature Importance" && (
        <div className="space-y-6">
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h3 className="mb-4 font-semibold text-gray-900">Feature Importance (Gradient Boosting)</h3>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 13 }} />
                <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                <Bar dataKey="importance" fill="#E85D26" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h3 className="mb-3 font-semibold text-gray-900">Key Takeaways</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-sunset-500" /><span><strong>Median Income</strong> is by far the strongest predictor, accounting for over half the model{"'"}s decision-making power.</span></li>
              <li className="flex gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-sunset-500" /><span><strong>Location (lat/long)</strong> together make up ~27% — geography matters significantly in California real estate.</span></li>
              <li className="flex gap-2"><span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-sunset-500" /><span><strong>Housing characteristics</strong> (rooms, age, occupancy) have moderate influence, reflecting property-level factors.</span></li>
            </ul>
          </div>
        </div>
      )}

      {activeTab === "Market Insights" && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-3">
            {[
              { region: "Northern California", range: "$400K-$800K+", desc: "Dominated by Bay Area tech economy. Highest price variance.", color: "bg-pacific-500" },
              { region: "Central California", range: "$300K-$600K", desc: "Agricultural heartland with moderate pricing.", color: "bg-emerald-500" },
              { region: "Southern California", range: "$500K-$1M+", desc: "Entertainment and biotech hubs. Coastal premiums.", color: "bg-amber-500" },
            ].map((r) => (
              <div key={r.region} className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
                <div className="flex items-center gap-2">
                  <div className={`h-3 w-3 rounded-full ${r.color}`} />
                  <h3 className="font-semibold text-gray-900">{r.region}</h3>
                </div>
                <p className="mt-2 text-2xl font-bold text-gray-800">{r.range}</p>
                <p className="mt-2 text-sm text-gray-500">{r.desc}</p>
              </div>
            ))}
          </div>
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h3 className="mb-4 font-semibold text-gray-900">Top Price Drivers</h3>
            <div className="space-y-4">
              {[
                { driver: "Income Level", impact: "High", desc: "Areas with median income above $80K see 2-3x higher home values" },
                { driver: "Coastal Proximity", impact: "High", desc: "Coastal locations command 30-50% premium over equivalent inland properties" },
                { driver: "Housing Stock Age", impact: "Medium", desc: "Newer developments in growing suburbs often price higher" },
                { driver: "Population Density", impact: "Medium", desc: "Moderate density areas balance desirability with affordability" },
              ].map((d) => (
                <div key={d.driver} className="flex items-start gap-4 rounded-lg bg-gray-50 p-4">
                  <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-bold uppercase ${d.impact === "High" ? "bg-red-100 text-red-700" : "bg-amber-100 text-amber-700"}`}>{d.impact}</span>
                  <div>
                    <p className="font-medium text-gray-800">{d.driver}</p>
                    <p className="mt-0.5 text-sm text-gray-500">{d.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
