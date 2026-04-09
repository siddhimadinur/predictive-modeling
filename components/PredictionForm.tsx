"use client";

import { useState } from "react";
import { HousingInput } from "@/lib/types";
import { PRESETS, FEATURE_CONFIG } from "@/lib/constants";
import clsx from "clsx";

interface Props {
  onSubmit: (values: HousingInput, model: string) => void;
  loading: boolean;
}

const DEFAULT_VALUES: HousingInput = {
  median_income: 3.9,
  housing_median_age: 29,
  ave_rooms: 5.4,
  ave_bedrooms: 1.1,
  population: 1425,
  ave_occupancy: 3.1,
  latitude: 35.6,
  longitude: -119.6,
};

const MODELS = [
  { value: "gradient_boosting", label: "Gradient Boosting", badge: "Champion" },
  { value: "random_forest", label: "Random Forest", badge: null },
  { value: "ridge", label: "Ridge Regression", badge: null },
];

export default function PredictionForm({ onSubmit, loading }: Props) {
  const [values, setValues] = useState<HousingInput>(DEFAULT_VALUES);
  const [selectedModel, setSelectedModel] = useState("gradient_boosting");
  const [activePreset, setActivePreset] = useState<number | null>(null);

  const handleChange = (key: keyof HousingInput, val: string) => {
    setValues((prev) => ({ ...prev, [key]: parseFloat(val) || 0 }));
    setActivePreset(null);
  };

  const applyPreset = (index: number) => {
    setValues(PRESETS[index].values);
    setActivePreset(index);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(values, selectedModel);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Model selector */}
      <div>
        <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-500">
          Model
        </label>
        <div className="flex flex-wrap gap-2">
          {MODELS.map((m) => (
            <button
              key={m.value}
              type="button"
              onClick={() => setSelectedModel(m.value)}
              className={clsx(
                "flex items-center gap-2 rounded-lg border px-3.5 py-2 text-sm font-medium transition-all",
                selectedModel === m.value
                  ? "border-brand-300 bg-brand-50 text-brand-700 ring-2 ring-brand-500/20"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
              )}
            >
              {m.label}
              {m.badge && (
                <span className="rounded-full bg-accent-500 px-2 py-0.5 text-[10px] font-bold uppercase text-white">
                  {m.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Presets */}
      <div>
        <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-slate-500">
          Quick Presets
        </label>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p, i) => (
            <button
              key={p.name}
              type="button"
              onClick={() => applyPreset(i)}
              className={clsx(
                "rounded-full border px-3 py-1.5 text-xs font-medium transition-all",
                activePreset === i
                  ? "border-brand-300 bg-brand-50 text-brand-700"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50"
              )}
            >
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Input grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {(Object.keys(FEATURE_CONFIG) as (keyof HousingInput)[]).map((key) => {
          const cfg = FEATURE_CONFIG[key];
          return (
            <div key={key}>
              <label
                htmlFor={key}
                className="mb-1 flex items-baseline justify-between text-sm"
              >
                <span className="font-medium text-slate-700">{cfg.label}</span>
                <span className="text-xs text-slate-400">{cfg.unit}</span>
              </label>
              <input
                id={key}
                type="number"
                min={cfg.min}
                max={cfg.max}
                step={cfg.step}
                value={values[key]}
                onChange={(e) => handleChange(key, e.target.value)}
                className="input-field"
              />
              <div className="mt-1">
                <input
                  type="range"
                  min={cfg.min}
                  max={cfg.max}
                  step={cfg.step}
                  value={values[key]}
                  onChange={(e) => handleChange(key, e.target.value)}
                  className="w-full accent-brand-600"
                />
              </div>
            </div>
          );
        })}
      </div>

      <button type="submit" disabled={loading} className="btn-primary w-full">
        {loading ? (
          <span className="flex items-center gap-2">
            <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Predicting...
          </span>
        ) : (
          "Predict Price"
        )}
      </button>
    </form>
  );
}
