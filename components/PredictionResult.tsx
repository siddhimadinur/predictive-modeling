"use client";

import { PredictionResponse } from "@/lib/types";
import { formatCurrency, getRegion, getIncomeLevel } from "@/lib/constants";

interface Props {
  result: PredictionResponse;
  income: number;
  latitude: number;
}

function GaugeChart({ value }: { value: number }) {
  const min = 50000;
  const max = 600000;
  const pct = Math.min(Math.max((value - min) / (max - min), 0), 1);
  const angle = -90 + pct * 180;
  const r = 80;
  const cx = 100;
  const cy = 95;

  // Arc path
  const startX = cx + r * Math.cos((-90 * Math.PI) / 180);
  const startY = cy + r * Math.sin((-90 * Math.PI) / 180);
  const endX = cx + r * Math.cos((90 * Math.PI) / 180);
  const endY = cy + r * Math.sin((90 * Math.PI) / 180);

  // Needle endpoint
  const needleX = cx + (r - 10) * Math.cos((angle * Math.PI) / 180);
  const needleY = cy + (r - 10) * Math.sin((angle * Math.PI) / 180);

  return (
    <svg viewBox="0 0 200 120" className="w-full max-w-[280px]">
      <defs>
        <linearGradient id="gaugeGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="50%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
      </defs>
      {/* Background arc */}
      <path
        d={`M ${startX} ${startY} A ${r} ${r} 0 0 1 ${endX} ${endY}`}
        fill="none"
        stroke="#e2e8f0"
        strokeWidth="12"
        strokeLinecap="round"
      />
      {/* Colored arc */}
      <path
        d={`M ${startX} ${startY} A ${r} ${r} 0 0 1 ${endX} ${endY}`}
        fill="none"
        stroke="url(#gaugeGrad)"
        strokeWidth="12"
        strokeLinecap="round"
      />
      {/* Needle */}
      <line
        x1={cx}
        y1={cy}
        x2={needleX}
        y2={needleY}
        stroke="#1e293b"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      <circle cx={cx} cy={cy} r="4" fill="#1e293b" />
      {/* Labels */}
      <text x="20" y="115" className="text-[9px] fill-slate-400" textAnchor="middle">$50K</text>
      <text x="180" y="115" className="text-[9px] fill-slate-400" textAnchor="middle">$600K</text>
    </svg>
  );
}

export default function PredictionResult({ result, income, latitude }: Props) {
  const region = getRegion(latitude);
  const incomeLevel = getIncomeLevel(income);

  return (
    <div className="space-y-6">
      {/* Price card */}
      <div className="card overflow-hidden">
        <div className="bg-gradient-to-br from-brand-600 to-brand-800 px-6 py-8 text-center text-white">
          <p className="mb-1 text-sm font-medium text-brand-200">Predicted Home Value</p>
          <p className="text-4xl font-extrabold tracking-tight sm:text-5xl">
            {formatCurrency(result.prediction)}
          </p>
          <p className="mt-2 text-sm text-brand-200">
            via {result.model_used.replace(/_/g, " ")}
          </p>
        </div>

        <div className="flex items-center justify-center py-4">
          <GaugeChart value={result.prediction} />
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-3">
        <div className="card px-4 py-3 text-center">
          <p className="text-xs font-medium text-slate-500">Price / Room</p>
          <p className="mt-1 text-lg font-bold text-slate-900">
            {formatCurrency(result.price_per_room)}
          </p>
        </div>
        <div className="card px-4 py-3 text-center">
          <p className="text-xs font-medium text-slate-500">Price-to-Income</p>
          <p className="mt-1 text-lg font-bold text-slate-900">
            {result.income_to_price_ratio.toFixed(1)}x
          </p>
        </div>
      </div>

      {/* Confidence Interval */}
      {result.confidence_interval && (
        <div className="card px-5 py-4">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
            95% Confidence Interval
          </p>
          <div className="relative h-3 w-full rounded-full bg-slate-100">
            {(() => {
              const lo = result.confidence_interval.lower;
              const hi = result.confidence_interval.upper;
              const scale = (v: number) =>
                Math.min(Math.max(((v - 50000) / 550000) * 100, 0), 100);
              return (
                <div
                  className="absolute inset-y-0 rounded-full bg-brand-200"
                  style={{ left: `${scale(lo)}%`, right: `${100 - scale(hi)}%` }}
                >
                  <div
                    className="absolute top-1/2 h-4 w-1.5 -translate-y-1/2 rounded-full bg-brand-600"
                    style={{
                      left: `${((result.prediction - lo) / (hi - lo)) * 100}%`,
                    }}
                  />
                </div>
              );
            })()}
          </div>
          <div className="mt-2 flex justify-between text-xs text-slate-500">
            <span>{formatCurrency(result.confidence_interval.lower)}</span>
            <span>{formatCurrency(result.confidence_interval.upper)}</span>
          </div>
        </div>
      )}

      {/* Context */}
      <div className="card px-5 py-4">
        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
          Property Context
        </p>
        <div className="space-y-2">
          {[
            { label: "Region", value: region },
            { label: "Income Level", value: incomeLevel },
            {
              label: "Annual Income",
              value: formatCurrency(income * 10000),
            },
          ].map((item) => (
            <div key={item.label} className="flex justify-between text-sm">
              <span className="text-slate-500">{item.label}</span>
              <span className="font-medium text-slate-800">{item.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
