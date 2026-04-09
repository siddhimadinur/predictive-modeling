import { Preset } from "./types";

export const PRESETS: Preset[] = [
  {
    name: "San Francisco Bay Area",
    description: "Tech hub, high-income area near the coast",
    values: {
      median_income: 8.5,
      housing_median_age: 35,
      ave_rooms: 6.2,
      ave_bedrooms: 1.05,
      population: 1500,
      ave_occupancy: 2.8,
      latitude: 37.8,
      longitude: -122.4,
    },
  },
  {
    name: "Los Angeles Metro",
    description: "Entertainment capital, diverse housing market",
    values: {
      median_income: 5.0,
      housing_median_age: 30,
      ave_rooms: 5.0,
      ave_bedrooms: 1.05,
      population: 2000,
      ave_occupancy: 3.2,
      latitude: 34.1,
      longitude: -118.2,
    },
  },
  {
    name: "San Diego Area",
    description: "Coastal city with growing tech presence",
    values: {
      median_income: 5.5,
      housing_median_age: 25,
      ave_rooms: 5.5,
      ave_bedrooms: 1.08,
      population: 1800,
      ave_occupancy: 3.0,
      latitude: 32.7,
      longitude: -117.2,
    },
  },
  {
    name: "Sacramento Valley",
    description: "State capital, affordable alternative to Bay Area",
    values: {
      median_income: 4.2,
      housing_median_age: 20,
      ave_rooms: 5.8,
      ave_bedrooms: 1.1,
      population: 1200,
      ave_occupancy: 2.9,
      latitude: 38.6,
      longitude: -121.5,
    },
  },
  {
    name: "Central Valley",
    description: "Agricultural heartland, most affordable region",
    values: {
      median_income: 2.8,
      housing_median_age: 18,
      ave_rooms: 4.8,
      ave_bedrooms: 1.1,
      population: 1100,
      ave_occupancy: 3.3,
      latitude: 36.0,
      longitude: -120.0,
    },
  },
];

export const FEATURE_CONFIG = {
  median_income: { label: "Median Income", unit: "×$10K", min: 0.5, max: 15, step: 0.1 },
  housing_median_age: { label: "Housing Age", unit: "years", min: 1, max: 52, step: 1 },
  ave_rooms: { label: "Avg Rooms", unit: "per household", min: 1, max: 15, step: 0.1 },
  ave_bedrooms: { label: "Avg Bedrooms", unit: "per household", min: 0.3, max: 5, step: 0.1 },
  population: { label: "Population", unit: "people", min: 3, max: 10000, step: 50 },
  ave_occupancy: { label: "Avg Occupancy", unit: "per household", min: 0.5, max: 10, step: 0.1 },
  latitude: { label: "Latitude", unit: "°N", min: 32.5, max: 42.0, step: 0.1 },
  longitude: { label: "Longitude", unit: "°W", min: -124.5, max: -114.0, step: 0.1 },
} as const;

export const MODEL_METRICS = [
  { name: "gradient_boosting", display_name: "Gradient Boosting", val_rmse: 13656, val_r2: 0.9779, val_mae: 7965, is_champion: true },
  { name: "random_forest", display_name: "Random Forest", val_rmse: 14777, val_r2: 0.9741, val_mae: 8193, is_champion: false },
  { name: "ridge", display_name: "Ridge Regression", val_rmse: 18044, val_r2: 0.9613, val_mae: 13928, is_champion: false },
];

export const REGIONS = [
  { name: "Northern California", lat_range: "≥37.0°", price_range: "$400K–$800K+", description: "Bay Area, Sacramento — tech hub, premium pricing" },
  { name: "Central California", lat_range: "35.0°–36.9°", price_range: "$300K–$600K", description: "Central Valley, Monterey — agricultural, moderate pricing" },
  { name: "Southern California", lat_range: "<35.0°", price_range: "$500K–$1M+", description: "LA, San Diego — entertainment, biotech, diverse market" },
];

export function getRegion(latitude: number): string {
  if (latitude >= 37.0) return "Northern California";
  if (latitude >= 35.0) return "Central California";
  return "Southern California";
}

export function getIncomeLevel(income: number): string {
  const annual = income * 10000;
  if (annual < 30000) return "Low Income";
  if (annual < 60000) return "Moderate Income";
  if (annual < 100000) return "High Income";
  return "Very High Income";
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}
