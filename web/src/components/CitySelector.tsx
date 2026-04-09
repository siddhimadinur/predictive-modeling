"use client";

import { cityNames } from "@/lib/cities";

interface CitySelectorProps {
  value: string;
  onChange: (city: string) => void;
}

export default function CitySelector({ value, onChange }: CitySelectorProps) {
  return (
    <div>
      <label className="block text-sm font-semibold text-gray-700 mb-2">
        City / Neighborhood
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-4 py-3 rounded-xl border border-gray-200 bg-white text-gray-800 font-medium shadow-sm focus:outline-none focus:ring-2 focus:ring-sunset-500 focus:border-transparent transition"
      >
        {cityNames.map((city) => (
          <option key={city} value={city}>
            {city}
          </option>
        ))}
      </select>
    </div>
  );
}
