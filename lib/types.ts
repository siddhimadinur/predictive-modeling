export interface HousingInput {
  median_income: number;
  housing_median_age: number;
  ave_rooms: number;
  ave_bedrooms: number;
  population: number;
  ave_occupancy: number;
  latitude: number;
  longitude: number;
}

export interface PredictionResponse {
  prediction: number;
  model_used: string;
  confidence_interval?: {
    lower: number;
    upper: number;
  };
  price_per_room: number;
  income_to_price_ratio: number;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  val_rmse: number;
  val_r2: number;
  val_mae: number;
  is_champion: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
  champion: string;
  training_samples: number;
  feature_count: number;
}

export interface Preset {
  name: string;
  description: string;
  values: HousingInput;
}
