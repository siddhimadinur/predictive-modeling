"use client";

import { useState, useEffect } from "react";
import CitySelector from "@/components/CitySelector";
import PropertySliders from "@/components/PropertySliders";
import PriceDisplay from "@/components/PriceDisplay";
import { CALIFORNIA_CITIES } from "@/lib/cities";

export default function Home() {
  const [city, setCity] = useState("Los Angeles");
  const [rooms, setRooms] = useState(4);
  const [bedrooms, setBedrooms] = useState(2);
  const [householdSize, setHouseholdSize] = useState(3);

  const [prediction1990, setPrediction1990] = useState<number | null>(null);
  const [prediction2024, setPrediction2024] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      try {
        const cityData = CALIFORNIA_CITIES[city];
        const res = await fetch("/api", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            median_income: cityData.median_income,
            housing_median_age: cityData.housing_median_age,
            ave_rooms: rooms,
            ave_bedrooms: bedrooms,
            population: cityData.population,
            ave_occupancy: householdSize,
            latitude: cityData.latitude,
            longitude: cityData.longitude,
          }),
        });
        const data = await res.json();
        setPrediction1990(data.prediction_1990);
        setPrediction2024(data.prediction_2024);
      } catch {
        setPrediction1990(null);
        setPrediction2024(null);
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [city, rooms, bedrooms, householdSize]);

  const cityData = CALIFORNIA_CITIES[city];

  return (
    <main className="max-w-4xl mx-auto px-4 py-8">
      {/* Hero */}
      <div className="bg-gradient-to-r from-sunset-500 via-orange-400 to-yellow-400 rounded-3xl p-8 mb-8 shadow-xl relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(255,255,255,0.15)_0%,transparent_60%)]" />
        <h1 className="text-4xl font-extrabold text-white text-center tracking-tight relative">
          California Housing Price Predictor
        </h1>
        <p className="text-white/90 text-center mt-2 text-lg relative">
          AI-powered property valuation for the Golden State
        </p>
        <div className="flex justify-center gap-4 mt-4 relative">
          <span className="bg-white/20 backdrop-blur-sm text-white text-sm font-semibold px-4 py-1.5 rounded-full">
            Gradient Boosting
          </span>
          <span className="bg-white/20 backdrop-blur-sm text-white text-sm font-semibold px-4 py-1.5 rounded-full">
            16K+ Training Samples
          </span>
          <span className="bg-white/20 backdrop-blur-sm text-white text-sm font-semibold px-4 py-1.5 rounded-full">
            R² 0.83
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left: Inputs */}
        <div className="space-y-6">
          {/* City */}
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-800 mb-4">
              Pick a Location
            </h2>
            <CitySelector value={city} onChange={setCity} />
            <div className="mt-4 grid grid-cols-3 gap-3">
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500">Median Income</p>
                <p className="text-sm font-bold text-gray-800">
                  ${(cityData.median_income * 10).toFixed(0)}K
                </p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500">Housing Age</p>
                <p className="text-sm font-bold text-gray-800">
                  {cityData.housing_median_age} yrs
                </p>
              </div>
              <div className="bg-gray-50 rounded-xl p-3 text-center">
                <p className="text-xs text-gray-500">Population</p>
                <p className="text-sm font-bold text-gray-800">
                  {cityData.population.toLocaleString()}
                </p>
              </div>
            </div>
          </div>

          {/* Sliders */}
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
            <h2 className="text-lg font-bold text-gray-800 mb-4">
              Describe Your Property
            </h2>
            <PropertySliders
              rooms={rooms}
              bedrooms={bedrooms}
              householdSize={householdSize}
              onRoomsChange={setRooms}
              onBedroomsChange={setBedrooms}
              onHouseholdSizeChange={setHouseholdSize}
            />
          </div>
        </div>

        {/* Right: Results */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6">
          <h2 className="text-lg font-bold text-gray-800 mb-4">
            Price Estimate
          </h2>
          <PriceDisplay
            prediction1990={prediction1990}
            prediction2024={prediction2024}
            loading={loading}
          />
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center text-gray-400 text-sm mt-12 pt-6 border-t border-gray-100">
        Built with Next.js, scikit-learn & Tailwind CSS · Trained on California
        Housing Census Data
      </footer>
    </main>
  );
}
