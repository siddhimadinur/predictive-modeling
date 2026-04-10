export interface CityData {
  latitude: number;
  longitude: number;
  median_income: number; // 1990 value used by model (in $10Ks)
  display_income: number; // 2024 estimated median household income ($)
  population: number;
  housing_median_age: number;
}

export const CALIFORNIA_CITIES: Record<string, CityData> = {
  "San Francisco": {
    latitude: 37.77, longitude: -122.42,
    median_income: 8.5, display_income: 126000, population: 1800, housing_median_age: 40,
  },
  "Oakland": {
    latitude: 37.80, longitude: -122.27,
    median_income: 5.8, display_income: 85000, population: 2000, housing_median_age: 42,
  },
  "San Jose": {
    latitude: 37.34, longitude: -121.89,
    median_income: 7.8, display_income: 130000, population: 1600, housing_median_age: 28,
  },
  "Palo Alto": {
    latitude: 37.44, longitude: -122.14,
    median_income: 10.5, display_income: 180000, population: 1200, housing_median_age: 38,
  },
  "Berkeley": {
    latitude: 37.87, longitude: -122.27,
    median_income: 6.5, display_income: 100000, population: 1500, housing_median_age: 45,
  },
  "Los Angeles": {
    latitude: 34.05, longitude: -118.24,
    median_income: 5.0, display_income: 75000, population: 2500, housing_median_age: 35,
  },
  "Santa Monica": {
    latitude: 34.02, longitude: -118.49,
    median_income: 8.0, display_income: 115000, population: 1400, housing_median_age: 38,
  },
  "Long Beach": {
    latitude: 33.77, longitude: -118.19,
    median_income: 4.5, display_income: 72000, population: 2200, housing_median_age: 36,
  },
  "Pasadena": {
    latitude: 34.15, longitude: -118.14,
    median_income: 6.0, display_income: 88000, population: 1600, housing_median_age: 40,
  },
  "Irvine": {
    latitude: 33.68, longitude: -117.83,
    median_income: 8.2, display_income: 115000, population: 1300, housing_median_age: 18,
  },
  "San Diego": {
    latitude: 32.72, longitude: -117.16,
    median_income: 5.5, display_income: 89000, population: 1800, housing_median_age: 25,
  },
  "La Jolla": {
    latitude: 32.84, longitude: -117.27,
    median_income: 9.5, display_income: 150000, population: 1000, housing_median_age: 30,
  },
  "Sacramento": {
    latitude: 38.58, longitude: -121.49,
    median_income: 4.2, display_income: 75000, population: 1400, housing_median_age: 25,
  },
  "Santa Barbara": {
    latitude: 34.42, longitude: -119.70,
    median_income: 6.8, display_income: 95000, population: 1100, housing_median_age: 32,
  },
  "Riverside": {
    latitude: 33.95, longitude: -117.40,
    median_income: 3.8, display_income: 72000, population: 1900, housing_median_age: 20,
  },
  "Fresno": {
    latitude: 36.74, longitude: -119.77,
    median_income: 3.0, display_income: 58000, population: 1600, housing_median_age: 22,
  },
  "Bakersfield": {
    latitude: 35.37, longitude: -119.02,
    median_income: 2.8, display_income: 55000, population: 1500, housing_median_age: 18,
  },
  "Stockton": {
    latitude: 37.95, longitude: -121.29,
    median_income: 3.2, display_income: 62000, population: 1700, housing_median_age: 24,
  },
  "Redding": {
    latitude: 40.59, longitude: -122.39,
    median_income: 3.5, display_income: 58000, population: 900, housing_median_age: 20,
  },
  "Palm Springs": {
    latitude: 33.83, longitude: -116.55,
    median_income: 4.0, display_income: 55000, population: 800, housing_median_age: 28,
  },
};

export const cityNames = Object.keys(CALIFORNIA_CITIES);
