export const MODEL_METRICS = [
  { name: "gradient_boosting", display_name: "Gradient Boosting", val_rmse: 13656, val_r2: 0.9779, val_mae: 7965, is_champion: true },
  { name: "random_forest", display_name: "Random Forest", val_rmse: 14777, val_r2: 0.9741, val_mae: 8193, is_champion: false },
  { name: "ridge", display_name: "Ridge Regression", val_rmse: 18044, val_r2: 0.9613, val_mae: 13928, is_champion: false },
];

export const REGIONS = [
  { name: "Northern California", lat_range: ">=37.0", price_range: "$400K-$800K+", description: "Bay Area, Sacramento - tech hub, premium pricing" },
  { name: "Central California", lat_range: "35.0-36.9", price_range: "$300K-$600K", description: "Central Valley, Monterey - agricultural, moderate pricing" },
  { name: "Southern California", lat_range: "<35.0", price_range: "$500K-$1M+", description: "LA, San Diego - entertainment, biotech, diverse market" },
];

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}
