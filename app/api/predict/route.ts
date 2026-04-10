import { NextResponse } from "next/server";
import predictions from "@/public/predictions.json";

const INFLATION_MULTIPLIER = 4.5;

// Map city coordinates to city names for lookup
const CITY_LOOKUP: Record<string, string> = {
  "37.77,-122.42": "San Francisco",
  "37.80,-122.27": "Oakland",
  "37.34,-121.89": "San Jose",
  "37.44,-122.14": "Palo Alto",
  "37.87,-122.27": "Berkeley",
  "34.05,-118.24": "Los Angeles",
  "34.02,-118.49": "Santa Monica",
  "33.77,-118.19": "Long Beach",
  "34.15,-118.14": "Pasadena",
  "33.68,-117.83": "Irvine",
  "32.72,-117.16": "San Diego",
  "32.84,-117.27": "La Jolla",
  "38.58,-121.49": "Sacramento",
  "34.42,-119.70": "Santa Barbara",
  "33.95,-117.40": "Riverside",
  "36.74,-119.77": "Fresno",
  "35.37,-119.02": "Bakersfield",
  "37.95,-121.29": "Stockton",
  "40.59,-122.39": "Redding",
  "33.83,-116.55": "Palm Springs",
};

const preds = predictions as Record<string, number>;

function findClosestCity(lat: number, lon: number): string | null {
  let bestCity: string | null = null;
  let bestDist = Infinity;

  for (const [coordKey, cityName] of Object.entries(CITY_LOOKUP)) {
    const [cLat, cLon] = coordKey.split(",").map(Number);
    const dist = Math.sqrt((lat - cLat) ** 2 + (lon - cLon) ** 2);
    if (dist < bestDist) {
      bestDist = dist;
      bestCity = cityName;
    }
  }

  return bestCity;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { latitude, longitude, ave_rooms, ave_bedrooms, ave_occupancy } = body;

    const city = findClosestCity(latitude, longitude);
    if (!city) {
      return NextResponse.json({ error: "Could not match location" }, { status: 400 });
    }

    const rooms = Math.round(Math.min(Math.max(ave_rooms, 1), 5));
    const bedrooms = Math.round(Math.min(Math.max(ave_bedrooms, 1), rooms));
    const household = Math.round(Math.min(Math.max(ave_occupancy, 1), 8));

    const key = `${city}|${rooms}|${bedrooms}|${household}`;
    const prediction = preds[key];

    if (prediction === undefined) {
      return NextResponse.json({ error: `No prediction for key: ${key}` }, { status: 404 });
    }

    const adjusted = Math.round(prediction * INFLATION_MULTIPLIER);

    return NextResponse.json({
      prediction: adjusted,
      prediction_1990: prediction,
      model_used: "gradient_boosting",
      price_per_room: Math.round(adjusted / rooms),
      income_to_price_ratio: +(adjusted / (body.median_income * 10000)).toFixed(1),
      confidence_interval: {
        lower: Math.round(adjusted * 0.85),
        upper: Math.round(adjusted * 1.15),
      },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[API /predict]", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
