"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Cell,
  LabelList,
} from "recharts";

interface PriceDisplayProps {
  prediction1990: number | null;
  prediction2024: number | null;
  loading: boolean;
}

function formatPrice(value: number): string {
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(2)}M`;
  }
  return `$${(value / 1_000).toFixed(0)}K`;
}

export default function PriceDisplay({
  prediction1990,
  prediction2024,
  loading,
}: PriceDisplayProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-sunset-200 border-t-sunset-500" />
      </div>
    );
  }

  if (prediction2024 === null || prediction1990 === null) {
    return (
      <div className="text-center py-16 text-gray-400">
        <p className="text-lg">Adjust the sliders to see a price estimate</p>
      </div>
    );
  }

  const chartData = [
    { name: "1990 Value", value: prediction1990, color: "#1B6B93" },
    { name: "2024 Estimate", value: prediction2024, color: "#E85D26" },
  ];

  return (
    <div className="space-y-6">
      {/* Main price cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 2024 estimate - primary */}
        <div className="bg-gradient-to-br from-sunset-500 to-orange-400 rounded-2xl p-6 text-white shadow-lg">
          <p className="text-sm font-medium opacity-90 mb-1">
            Estimated 2024 Value
          </p>
          <p className="text-4xl font-extrabold tracking-tight">
            {formatPrice(prediction2024)}
          </p>
          <p className="text-xs opacity-75 mt-2">
            Inflation-adjusted estimate
          </p>
        </div>

        {/* 1990 value - secondary */}
        <div className="bg-gradient-to-br from-pacific-500 to-pacific-400 rounded-2xl p-6 text-white shadow-lg">
          <p className="text-sm font-medium opacity-90 mb-1">
            1990 Census Value
          </p>
          <p className="text-4xl font-extrabold tracking-tight">
            {formatPrice(prediction1990)}
          </p>
          <p className="text-xs opacity-75 mt-2">Raw model output</p>
        </div>
      </div>

      {/* Comparison bar chart */}
      <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-4">
        <p className="text-sm font-semibold text-gray-600 mb-3">
          Price Comparison
        </p>
        <ResponsiveContainer width="100%" height={100}>
          <BarChart data={chartData} layout="vertical" barSize={28}>
            <XAxis
              type="number"
              tickFormatter={(v) => formatPrice(v)}
              fontSize={12}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={100}
              fontSize={12}
              axisLine={false}
              tickLine={false}
            />
            <Bar dataKey="value" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: number) => formatPrice(v)}
                className="text-xs font-semibold"
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-gray-400 text-center">
        Model trained on 1990 Census data. 2024 estimate applies a 4.5x
        California housing inflation adjustment.
      </p>
    </div>
  );
}
