import React, { useState, useRef, useEffect, useCallback } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Search,
  ChevronRight,
  Globe,
  X,
  Compass,
  Flame,
  Users,
  Sparkles,
  Zap,
  Navigation,
  Loader2,
} from "lucide-react";
import { CountrySummary } from "@/lib/api";

interface InteractiveWorldMapProps {
  countries: CountrySummary[];
  selectedCountry: CountrySummary | null;
  onSelectCountry: (country: CountrySummary) => void;
  radiusKm: string;
}

type LocationLevel =
  | "world"
  | "continent"
  | "country"
  | "state"
  | "district"
  | "city"
  | "neighborhood"
  | "landmark";

interface LocationNode {
  id: string;
  name: string;
  level: LocationLevel;
  parentId?: string;
  lngLat: [number, number];
  bounds?: [[number, number], [number, number]];
  flag?: string;
  countryCode?: string;
  population?: string;
  timeZone?: string;
  languages?: string[];
  majorPlatforms?: string[];
  trendScore?: number;
  activeCreators?: number;
  totalActiveTrends?: number;
  activityLevel?: "Low" | "Medium" | "High" | "Extremely High";
  trendIntensity?: string;
  intensityScore?: number;
  heatColor?: "red" | "orange" | "yellow" | "green";
  minZoom?: number;
  maxZoom?: number;
  isTrending?: boolean;
}

const LEVEL_LABELS: Record<LocationLevel, string> = {
  world: "World",
  continent: "Continent",
  country: "Country",
  state: "State / Province",
  district: "District / County",
  city: "City",
  neighborhood: "Neighborhood",
  landmark: "Landmark / Venue",
};

const RADIUS_KM_NUMBERS: Record<string, number> = {
  "5 km": 5,
  "10 km": 10,
  "25 km": 25,
  "50 km": 50,
  "100 km": 100,
  "250 km": 250,
  "500 km": 500,
  District: 30,
  State: 150,
  Country: 400,
  Continent: 1200,
};

function createGeoJSONCircle(
  center: [number, number],
  radiusInKm: number,
  points = 64
): GeoJSON.Feature<GeoJSON.Polygon> {
  const coords = { latitude: center[1], longitude: center[0] };
  const ret: [number, number][] = [];
  const distanceX = radiusInKm / (111.32 * Math.cos((coords.latitude * Math.PI) / 180));
  const distanceY = radiusInKm / 110.574;

  for (let i = 0; i < points; i++) {
    const theta = (i / points) * (2 * Math.PI);
    const x = distanceX * Math.cos(theta);
    const y = distanceY * Math.sin(theta);
    ret.push([coords.longitude + x, coords.latitude + y]);
  }
  ret.push(ret[0]);

  return {
    type: "Feature",
    geometry: { type: "Polygon", coordinates: [ret] },
    properties: {},
  };
}

// Extensive Global Administrative Dataset (India, USA, Japan, UK, Germany, UAE, Singapore, Hong Kong, Brazil, Australia, etc.)
const GLOBAL_LOCATIONS: LocationNode[] = [
  // Continents (Zoom: 0 - 3.5)
  { id: "AS", name: "Asia", level: "continent", lngLat: [89.29, 29.84], minZoom: 0, maxZoom: 3.5, trendScore: 98, activeCreators: 450000, trendIntensity: "98% Peak Surge", intensityScore: 98, heatColor: "red", isTrending: true },
  { id: "EU", name: "Europe", level: "continent", lngLat: [15.25, 54.52], minZoom: 0, maxZoom: 3.5, trendScore: 92, activeCreators: 280000, trendIntensity: "92% Strong", intensityScore: 92, heatColor: "orange" },
  { id: "NA", name: "North America", level: "continent", lngLat: [-100.54, 46.07], minZoom: 0, maxZoom: 3.5, trendScore: 95, activeCreators: 390000, trendIntensity: "95% High Intensity", intensityScore: 95, heatColor: "red", isTrending: true },
  { id: "SA", name: "South America", level: "continent", lngLat: [-55.49, -8.78], minZoom: 0, maxZoom: 3.5, trendScore: 89, activeCreators: 210000, trendIntensity: "89% Moderate", intensityScore: 89, heatColor: "orange" },
  { id: "AF", name: "Africa", level: "continent", lngLat: [17.87, -2.2], minZoom: 0, maxZoom: 3.5, trendScore: 84, activeCreators: 150000, trendIntensity: "84% Growing", intensityScore: 84, heatColor: "yellow" },
  { id: "OC", name: "Oceania", level: "continent", lngLat: [134.35, -24.77], minZoom: 0, maxZoom: 3.5, trendScore: 88, activeCreators: 95000, trendIntensity: "88% Steady", intensityScore: 88, heatColor: "yellow" },

  // Countries (Zoom: 1 - 6)
  { id: "IN", name: "India", level: "country", parentId: "AS", countryCode: "IN", flag: "🇮🇳", lngLat: [78.9629, 20.5937], bounds: [[68.1, 6.7], [97.3, 35.5]], minZoom: 1, maxZoom: 6, population: "~1.43 Billion", timeZone: "IST (UTC+5:30)", languages: ["Hindi", "English", "Tamil", "Telugu"], majorPlatforms: ["Instagram Reels", "YouTube Shorts", "Moj"], trendScore: 98, activeCreators: 184000, totalActiveTrends: 184, activityLevel: "Extremely High", trendIntensity: "Peak Velocity (98%)", intensityScore: 98, heatColor: "red", isTrending: true },
  { id: "US", name: "United States", level: "country", parentId: "NA", countryCode: "US", flag: "🇺🇸", lngLat: [-95.7129, 37.0902], bounds: [[-125.0, 24.3], [-66.9, 49.3]], minZoom: 1, maxZoom: 6, population: "~335 Million", timeZone: "EST / PST", languages: ["English", "Spanish"], majorPlatforms: ["TikTok", "Instagram Reels", "YouTube Shorts"], trendScore: 99, activeCreators: 240000, totalActiveTrends: 210, activityLevel: "Extremely High", trendIntensity: "Maximum Intensity (99%)", intensityScore: 99, heatColor: "red", isTrending: true },
  { id: "JP", name: "Japan", level: "country", parentId: "AS", countryCode: "JP", flag: "🇯🇵", lngLat: [138.2529, 36.2048], bounds: [[122.9, 24.2], [153.9, 45.5]], minZoom: 1, maxZoom: 6, population: "~125 Million", timeZone: "JST (UTC+9)", languages: ["Japanese"], majorPlatforms: ["YouTube Shorts", "X", "TikTok"], trendScore: 96, activeCreators: 110000, totalActiveTrends: 155, activityLevel: "Extremely High", trendIntensity: "96% High Surge", intensityScore: 96, heatColor: "red", isTrending: true },
  { id: "GB", name: "United Kingdom", level: "country", parentId: "EU", countryCode: "GB", flag: "🇬🇧", lngLat: [-3.436, 55.3781], minZoom: 1, maxZoom: 6, population: "~67 Million", timeZone: "GMT (UTC+0)", languages: ["English"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 93, activeCreators: 85000, totalActiveTrends: 128, activityLevel: "High", trendIntensity: "93% High Intensity", intensityScore: 93, heatColor: "orange" },
  { id: "DE", name: "Germany", level: "country", parentId: "EU", countryCode: "DE", flag: "🇩🇪", lngLat: [10.4515, 51.1657], minZoom: 1, maxZoom: 6, population: "~84 Million", timeZone: "CET (UTC+1)", languages: ["German", "English"], majorPlatforms: ["YouTube Shorts", "Instagram"], trendScore: 91, activeCreators: 72000, totalActiveTrends: 114, activityLevel: "High", trendIntensity: "91% Strong Surge", intensityScore: 91, heatColor: "orange" },
  { id: "AE", name: "United Arab Emirates", level: "country", parentId: "AS", countryCode: "AE", flag: "🇦🇪", lngLat: [53.8478, 23.4241], minZoom: 1, maxZoom: 6, population: "~10 Million", timeZone: "GST (UTC+4)", languages: ["Arabic", "English"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 94, activeCreators: 62000, totalActiveTrends: 140, activityLevel: "High", trendIntensity: "94% High Surge", intensityScore: 94, heatColor: "red", isTrending: true },
  { id: "SG", name: "Singapore", level: "country", parentId: "AS", countryCode: "SG", flag: "🇸🇬", lngLat: [103.8198, 1.3521], minZoom: 1, maxZoom: 6, population: "~5.6 Million", timeZone: "SGT (UTC+8)", languages: ["English", "Mandarin", "Malay"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 92, activeCreators: 38000, totalActiveTrends: 112, activityLevel: "High", trendIntensity: "92% High Surge", intensityScore: 92, heatColor: "orange" },
  { id: "HK", name: "Hong Kong", level: "country", parentId: "AS", countryCode: "HK", flag: "🇭🇰", lngLat: [114.1694, 22.3193], minZoom: 1, maxZoom: 6, population: "~7.4 Million", timeZone: "HKT (UTC+8)", languages: ["Cantonese", "English"], majorPlatforms: ["YouTube Shorts", "Instagram"], trendScore: 91, activeCreators: 34000, totalActiveTrends: 104, activityLevel: "High", trendIntensity: "91% High Intensity", intensityScore: 91, heatColor: "orange" },

  // States / Provinces (Zoom: 3.5 - 8.5)
  { id: "IN-TN", name: "Tamil Nadu", level: "state", parentId: "IN", countryCode: "IN", flag: "🇮🇳", lngLat: [78.6569, 11.1271], minZoom: 3.5, maxZoom: 8.5, population: "~7.6 Crore", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 96, activeCreators: 34500, totalActiveTrends: 162, activityLevel: "Extremely High", trendIntensity: "96% High Intensity", intensityScore: 96, heatColor: "red" },
  { id: "IN-MH", name: "Maharashtra", level: "state", parentId: "IN", countryCode: "IN", flag: "🇮🇳", lngLat: [75.7139, 19.7515], minZoom: 3.5, maxZoom: 8.5, population: "~12.8 Crore", timeZone: "IST (UTC+5:30)", languages: ["Marathi", "Hindi", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts", "Moj"], trendScore: 98, activeCreators: 49000, totalActiveTrends: 184, activityLevel: "Extremely High", trendIntensity: "98% Peak Surge", intensityScore: 98, heatColor: "red", isTrending: true },
  { id: "US-CA", name: "California", level: "state", parentId: "US", countryCode: "US", flag: "🇺🇸", lngLat: [-119.4179, 36.7783], minZoom: 3.5, maxZoom: 8.5, population: "~39 Million", timeZone: "PST (UTC-8)", languages: ["English", "Spanish"], majorPlatforms: ["TikTok", "Instagram Reels", "YouTube Shorts"], trendScore: 99, activeCreators: 68000, totalActiveTrends: 195, activityLevel: "Extremely High", trendIntensity: "99% Peak Intensity", intensityScore: 99, heatColor: "red", isTrending: true },
  { id: "JP-13", name: "Tokyo", level: "state", parentId: "JP", countryCode: "JP", flag: "🇯🇵", lngLat: [139.6917, 35.6895], minZoom: 3.5, maxZoom: 8.5, population: "~14 Million", timeZone: "JST (UTC+9)", languages: ["Japanese"], majorPlatforms: ["X", "YouTube Shorts"], trendScore: 99, activeCreators: 55000, totalActiveTrends: 175, activityLevel: "Extremely High", trendIntensity: "99% Peak Intensity", intensityScore: 99, heatColor: "red", isTrending: true },
  { id: "GB-ENG", name: "England", level: "state", parentId: "GB", countryCode: "GB", flag: "🇬🇧", lngLat: [-1.1743, 52.3555], minZoom: 3.5, maxZoom: 8.5, population: "~56 Million", timeZone: "GMT (UTC+0)", languages: ["English"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 94, activeCreators: 64000, totalActiveTrends: 135, activityLevel: "High", trendIntensity: "94% High Surge", intensityScore: 94, heatColor: "orange" },
  { id: "DE-BY", name: "Bavaria", level: "state", parentId: "DE", countryCode: "DE", flag: "🇩🇪", lngLat: [11.4979, 48.7904], minZoom: 3.5, maxZoom: 8.5, population: "~13.1 Million", timeZone: "CET (UTC+1)", languages: ["German"], majorPlatforms: ["YouTube Shorts", "Instagram"], trendScore: 90, activeCreators: 24000, totalActiveTrends: 88, activityLevel: "High", trendIntensity: "90% Steady", intensityScore: 90, heatColor: "yellow" },

  // Cities / Districts (Zoom: 6.5 - 12)
  { id: "IN-TN-CHN-CITY", name: "Chennai", level: "city", parentId: "IN-TN", countryCode: "IN", flag: "🇮🇳", lngLat: [80.2707, 13.0827], minZoom: 6.5, maxZoom: 12, population: "~11.5 Million", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 97, activeCreators: 16800, totalActiveTrends: 145, activityLevel: "Extremely High", trendIntensity: "97% Peak Intensity", intensityScore: 97, heatColor: "red", isTrending: true },
  { id: "IN-TN-CBE", name: "Coimbatore", level: "city", parentId: "IN-TN", countryCode: "IN", flag: "🇮🇳", lngLat: [76.9558, 11.0168], minZoom: 6.5, maxZoom: 12, population: "~2.8 Million", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 88, activeCreators: 7400, totalActiveTrends: 92, activityLevel: "High", trendIntensity: "88% High Intensity", intensityScore: 88, heatColor: "orange" },
  { id: "IN-TN-MDU", name: "Madurai", level: "city", parentId: "IN-TN", countryCode: "IN", flag: "🇮🇳", lngLat: [78.1198, 9.9252], minZoom: 6.5, maxZoom: 12, population: "~1.8 Million", timeZone: "IST (UTC+5:30)", languages: ["Tamil"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 72, activeCreators: 4200, totalActiveTrends: 68, activityLevel: "Medium", trendIntensity: "72% Medium Intensity", intensityScore: 72, heatColor: "yellow" },
  { id: "IN-TN-SLM", name: "Salem", level: "city", parentId: "IN-TN", countryCode: "IN", flag: "🇮🇳", lngLat: [78.146, 11.6643], minZoom: 6.5, maxZoom: 12, population: "~1.2 Million", timeZone: "IST (UTC+5:30)", languages: ["Tamil"], majorPlatforms: ["YouTube Shorts", "Moj"], trendScore: 45, activeCreators: 2100, totalActiveTrends: 34, activityLevel: "Low", trendIntensity: "45% Low Intensity", intensityScore: 45, heatColor: "green" },
  { id: "IN-TN-TRY", name: "Trichy", level: "city", parentId: "IN-TN", countryCode: "IN", flag: "🇮🇳", lngLat: [78.7047, 10.7905], minZoom: 6.5, maxZoom: 12, population: "~1.1 Million", timeZone: "IST (UTC+5:30)", languages: ["Tamil"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 65, activeCreators: 3100, totalActiveTrends: 52, activityLevel: "Medium", trendIntensity: "65% Moderate Intensity", intensityScore: 65, heatColor: "yellow" },
  { id: "US-CA-LAC", name: "Los Angeles County", level: "district", parentId: "US-CA", countryCode: "US", flag: "🇺🇸", lngLat: [-118.2437, 34.0522], minZoom: 6.5, maxZoom: 12, population: "~9.8 Million", timeZone: "PST (UTC-8)", languages: ["English", "Spanish"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 99, activeCreators: 42000, totalActiveTrends: 185, activityLevel: "Extremely High", trendIntensity: "99% Maximum Intensity", intensityScore: 99, heatColor: "red" },
  { id: "GB-LON", name: "Greater London", level: "district", parentId: "GB-ENG", countryCode: "GB", flag: "🇬🇧", lngLat: [-0.1276, 51.5074], minZoom: 6.5, maxZoom: 12, population: "~8.9 Million", timeZone: "GMT (UTC+0)", languages: ["English"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 96, activeCreators: 38000, totalActiveTrends: 158, activityLevel: "Extremely High", trendIntensity: "96% High Surge", intensityScore: 96, heatColor: "red" },
  { id: "DE-MUC", name: "Munich", level: "city", parentId: "DE-BY", countryCode: "DE", flag: "🇩🇪", lngLat: [11.582, 48.1351], minZoom: 6.5, maxZoom: 12, population: "~1.5 Million", timeZone: "CET (UTC+1)", languages: ["German", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 89, activeCreators: 14200, totalActiveTrends: 78, activityLevel: "High", trendIntensity: "89% High Intensity", intensityScore: 89, heatColor: "orange" },

  // Neighborhoods & Landmarks (Zoom: 10+)
  { id: "IN-TN-CHN-TNAGAR", name: "T Nagar", level: "neighborhood", parentId: "IN-TN-CHN-CITY", countryCode: "IN", flag: "🇮🇳", lngLat: [80.2337, 13.0418], minZoom: 10, maxZoom: 20, population: "~350,000", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 98, activeCreators: 4200, totalActiveTrends: 82, activityLevel: "Extremely High", trendIntensity: "98% Peak Intensity", intensityScore: 98, heatColor: "red", isTrending: true },
  { id: "IN-TN-CHN-ADYAR", name: "Adyar", level: "neighborhood", parentId: "IN-TN-CHN-CITY", countryCode: "IN", flag: "🇮🇳", lngLat: [80.2574, 13.0012], minZoom: 10, maxZoom: 20, population: "~220,000", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 94, activeCreators: 3400, totalActiveTrends: 70, activityLevel: "High", trendIntensity: "94% High Surge", intensityScore: 94, heatColor: "red" },
  { id: "IN-TN-CHN-VELACHERY", name: "Velachery", level: "neighborhood", parentId: "IN-TN-CHN-CITY", countryCode: "IN", flag: "🇮🇳", lngLat: [80.2206, 12.9815], minZoom: 10, maxZoom: 20, population: "~280,000", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels"], trendScore: 91, activeCreators: 2900, totalActiveTrends: 64, activityLevel: "High", trendIntensity: "91% High Intensity", intensityScore: 91, heatColor: "orange" },
  { id: "IN-TN-CHN-GUINDY", name: "Guindy", level: "neighborhood", parentId: "IN-TN-CHN-CITY", countryCode: "IN", flag: "🇮🇳", lngLat: [80.2121, 13.0067], minZoom: 10, maxZoom: 20, population: "~190,000", timeZone: "IST (UTC+5:30)", languages: ["Tamil", "English"], majorPlatforms: ["Instagram Reels", "YouTube Shorts"], trendScore: 89, activeCreators: 2400, totalActiveTrends: 56, activityLevel: "High", trendIntensity: "89% High Intensity", intensityScore: 89, heatColor: "orange" },
  { id: "US-CA-HOLLYWOOD", name: "Hollywood", level: "neighborhood", parentId: "US-CA-LAC", countryCode: "US", flag: "🇺🇸", lngLat: [-118.3287, 34.0928], minZoom: 10, maxZoom: 20, population: "~150,000", timeZone: "PST (UTC-8)", languages: ["English"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 99, activeCreators: 12500, totalActiveTrends: 120, activityLevel: "Extremely High", trendIntensity: "99% Peak Intensity", intensityScore: 99, heatColor: "red", isTrending: true },
  { id: "US-CA-BEVERLY", name: "Beverly Hills", level: "neighborhood", parentId: "US-CA-LAC", countryCode: "US", flag: "🇺🇸", lngLat: [-118.4004, 34.0736], minZoom: 10, maxZoom: 20, population: "~34,000", timeZone: "PST (UTC-8)", languages: ["English"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 97, activeCreators: 8400, totalActiveTrends: 98, activityLevel: "Extremely High", trendIntensity: "97% High Intensity", intensityScore: 97, heatColor: "red" },
  { id: "JP-SHIBUYA", name: "Shibuya", level: "neighborhood", parentId: "JP-13", countryCode: "JP", flag: "🇯🇵", lngLat: [139.7016, 35.658], minZoom: 10, maxZoom: 20, population: "~230,000", timeZone: "JST (UTC+9)", languages: ["Japanese"], majorPlatforms: ["X", "TikTok"], trendScore: 100, activeCreators: 14500, totalActiveTrends: 135, activityLevel: "Extremely High", trendIntensity: "100% Maximum Intensity", intensityScore: 100, heatColor: "red", isTrending: true },
  { id: "JP-SHINJUKU", name: "Shinjuku", level: "neighborhood", parentId: "JP-13", countryCode: "JP", flag: "🇯🇵", lngLat: [139.7036, 35.6938], minZoom: 10, maxZoom: 20, population: "~340,000", timeZone: "JST (UTC+9)", languages: ["Japanese"], majorPlatforms: ["YouTube Shorts", "X"], trendScore: 98, activeCreators: 11200, totalActiveTrends: 110, activityLevel: "Extremely High", trendIntensity: "98% Peak Intensity", intensityScore: 98, heatColor: "red" },
  { id: "GB-WESTMINSTER", name: "Westminster", level: "neighborhood", parentId: "GB-LON", countryCode: "GB", flag: "🇬🇧", lngLat: [-0.1357, 51.4975], minZoom: 10, maxZoom: 20, population: "~250,000", timeZone: "GMT (UTC+0)", languages: ["English"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 95, activeCreators: 7800, totalActiveTrends: 85, activityLevel: "High", trendIntensity: "95% High Intensity", intensityScore: 95, heatColor: "red" },
  { id: "AE-DOWNTOWN", name: "Downtown Dubai", level: "neighborhood", parentId: "AE", countryCode: "AE", flag: "🇦🇪", lngLat: [55.2744, 25.1972], minZoom: 10, maxZoom: 20, population: "~100,000", timeZone: "GST (UTC+4)", languages: ["Arabic", "English"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 99, activeCreators: 15400, totalActiveTrends: 125, activityLevel: "Extremely High", trendIntensity: "99% Peak Intensity", intensityScore: 99, heatColor: "red", isTrending: true },
  { id: "AE-MARINA", name: "Dubai Marina", level: "neighborhood", parentId: "AE", countryCode: "AE", flag: "🇦🇪", lngLat: [55.1403, 25.0772], minZoom: 10, maxZoom: 20, population: "~55,000", timeZone: "GST (UTC+4)", languages: ["English", "Arabic"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 96, activeCreators: 9800, totalActiveTrends: 92, activityLevel: "Extremely High", trendIntensity: "96% High Intensity", intensityScore: 96, heatColor: "red" },
  { id: "SG-ORCHARD", name: "Orchard", level: "neighborhood", parentId: "SG", countryCode: "SG", flag: "🇸🇬", lngLat: [103.8318, 1.3048], minZoom: 10, maxZoom: 20, population: "~40,000", timeZone: "SGT (UTC+8)", languages: ["English", "Mandarin"], majorPlatforms: ["TikTok", "Instagram Reels"], trendScore: 94, activeCreators: 6200, totalActiveTrends: 78, activityLevel: "High", trendIntensity: "94% High Intensity", intensityScore: 94, heatColor: "red" },
  { id: "SG-MARINABAY", name: "Marina Bay", level: "landmark", parentId: "SG", countryCode: "SG", flag: "🇸🇬", lngLat: [103.8587, 1.2839], minZoom: 10, maxZoom: 20, population: "~15,000", timeZone: "SGT (UTC+8)", languages: ["English"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 97, activeCreators: 8900, totalActiveTrends: 95, activityLevel: "Extremely High", trendIntensity: "97% Peak Surge", intensityScore: 97, heatColor: "red", isTrending: true },
  { id: "HK-MONGKOK", name: "Mong Kok", level: "neighborhood", parentId: "HK", countryCode: "HK", flag: "🇭🇰", lngLat: [114.1688, 22.3193], minZoom: 10, maxZoom: 20, population: "~140,000", timeZone: "HKT (UTC+8)", languages: ["Cantonese", "English"], majorPlatforms: ["YouTube Shorts", "Instagram"], trendScore: 95, activeCreators: 7200, totalActiveTrends: 82, activityLevel: "High", trendIntensity: "95% High Intensity", intensityScore: 95, heatColor: "red" },
  { id: "HK-TSIMSHATSUI", name: "Tsim Sha Tsui", level: "neighborhood", parentId: "HK", countryCode: "HK", flag: "🇭🇰", lngLat: [114.1722, 22.2988], minZoom: 10, maxZoom: 20, population: "~60,000", timeZone: "HKT (UTC+8)", languages: ["Cantonese", "English"], majorPlatforms: ["Instagram Reels", "TikTok"], trendScore: 96, activeCreators: 8100, totalActiveTrends: 88, activityLevel: "Extremely High", trendIntensity: "96% High Surge", intensityScore: 96, heatColor: "red" },
];

export function InteractiveWorldMap({
  countries,
  selectedCountry,
  onSelectCountry,
  radiusKm,
}: InteractiveWorldMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const selectedMarkerRef = useRef<maplibregl.Marker | null>(null);
  const userMarkerRef = useRef<maplibregl.Marker | null>(null);

  // Layer Toggles
  const [showHeatmap, setShowHeatmap] = useState(true);

  // Search & Navigation State
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<LocationNode[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [selectedNode, setSelectedNode] = useState<LocationNode | null>(null);
  const [breadcrumb, setBreadcrumb] = useState<LocationNode[]>([]);
  const [hoverInfo, setHoverInfo] = useState<{ node: LocationNode; x: number; y: number } | null>(null);

  // Convert LocationNode to CountrySummary format for right panel update
  const mapLocationToSummary = (loc: LocationNode): CountrySummary => {
    return {
      id: loc.id,
      name: loc.name,
      code: loc.countryCode || loc.id,
      flag: loc.flag || "📍",
      population: loc.population || "~5.0 Million",
      time_zone: loc.timeZone || "UTC+0",
      languages: loc.languages || ["English"],
      major_platforms: loc.majorPlatforms || ["Instagram Reels", "YouTube Shorts"],
      current_trend_score: loc.trendScore || 95,
      total_active_trends: loc.totalActiveTrends || 120,
    };
  };

  // Point & Click GIS Geocoding Fallback for ANY Location on Earth
  const reverseGeocodeGISPoint = async (lng: number, lat: number) => {
    try {
      const res = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${lat}&lon=${lng}&zoom=14`
      );
      if (res.ok) {
        const data = await res.json();
        const address = data.address || {};
        const placeName =
          address.suburb ||
          address.neighbourhood ||
          address.city ||
          address.town ||
          address.county ||
          address.state ||
          data.name ||
          "Selected Location";
        const stateName = address.state || address.region || "Region";
        const countryName = address.country || "Global";
        const countryCode = (address.country_code || "GL").toUpperCase();

        const customNode: LocationNode = {
          id: `GIS-${lng.toFixed(3)}-${lat.toFixed(3)}`,
          name: placeName,
          level: address.neighbourhood || address.suburb ? "neighborhood" : address.city ? "city" : "state",
          lngLat: [lng, lat],
          countryCode: countryCode,
          flag: "📍",
          population: "~500,000",
          timeZone: "Local Standard Time",
          languages: [address.country_code === "in" ? "Hindi" : "English"],
          majorPlatforms: ["Instagram Reels", "YouTube Shorts", "TikTok"],
          trendScore: 92,
          activeCreators: 8400,
          totalActiveTrends: 76,
          activityLevel: "High",
          trendIntensity: "92% Active GIS Node",
          intensityScore: 92,
          heatColor: "red",
        };

        selectLocationNode(customNode);
      }
    } catch {
      // Fallback
    }
  };

  // Initialize Native MapLibre GL WebGL Engine
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: {
        version: 8,
        name: "ORIGO Full GIS Vector Map",
        sources: {
          cartoDark: {
            type: "raster",
            tiles: [
              "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
              "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
              "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
            ],
            tileSize: 256,
          },
        },
        layers: [
          {
            id: "carto-dark-base",
            type: "raster",
            source: "cartoDark",
            minzoom: 0,
            maxzoom: 19,
            paint: {
              "raster-opacity": 0.92,
              "raster-contrast": 0.3,
            },
          },
        ],
      },
      center: [20, 20],
      zoom: 1.6,
      pitch: 0,
      bearing: 0,
      attributionControl: false,
    });

    map.on("load", () => {
      map.resize();

      // Radius Circle Source & Layers
      map.addSource("radius-circle-source", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

      map.addLayer({
        id: "radius-circle-fill",
        type: "fill",
        source: "radius-circle-source",
        paint: {
          "fill-color": "#D4AF37",
          "fill-opacity": 0.18,
        },
      });

      map.addLayer({
        id: "radius-circle-stroke",
        type: "line",
        source: "radius-circle-source",
        paint: {
          "line-color": "#FACC15",
          "line-width": 2,
          "line-dasharray": [3, 2],
        },
      });

      // Data-Driven Heatmap Layer
      const heatmapFeatures: GeoJSON.FeatureCollection = {
        type: "FeatureCollection",
        features: GLOBAL_LOCATIONS.map((loc) => ({
          type: "Feature",
          properties: { weight: (loc.trendScore || 80) / 100 },
          geometry: { type: "Point", coordinates: loc.lngLat },
        })),
      };

      map.addSource("trend-heatmap-source", {
        type: "geojson",
        data: heatmapFeatures,
      });

      map.addLayer({
        id: "trend-heatmap-layer",
        type: "heatmap",
        source: "trend-heatmap-source",
        layout: { visibility: "visible" },
        paint: {
          "heatmap-weight": ["get", "weight"],
          "heatmap-intensity": 1.8,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0, "rgba(0, 0, 0, 0)",
            0.25, "rgba(34, 197, 94, 0.5)",
            0.5, "rgba(234, 179, 8, 0.75)",
            0.75, "rgba(249, 115, 22, 0.88)",
            1, "rgba(239, 68, 68, 0.98)",
          ],
          "heatmap-radius": 45,
          "heatmap-opacity": 0.75,
        },
      });

      // Add Dynamic Markers for Pre-defined Database Locations
      GLOBAL_LOCATIONS.forEach((loc) => {
        const el = document.createElement("div");
        el.className =
          "cursor-pointer group relative flex flex-col items-center justify-center transition-opacity duration-300 pointer-events-auto";

        const dot = document.createElement("div");
        const dotColor =
          loc.heatColor === "red"
            ? "bg-red-500 shadow-[0_0_15px_#EF4444]"
            : loc.heatColor === "orange"
            ? "bg-orange-400 shadow-[0_0_12px_#F97316]"
            : loc.heatColor === "yellow"
            ? "bg-yellow-400"
            : "bg-emerald-400";

        dot.className = loc.isTrending
          ? `h-4 w-4 rounded-full ${dotColor} border-2 border-black animate-pulse`
          : `h-3 w-3 rounded-full ${dotColor} border border-black hover:scale-125 transition`;
        el.appendChild(dot);

        const label = document.createElement("div");
        label.className =
          "mt-1 rounded-md border border-[oklch(0.85_0.155_86/0.4)] bg-[#0B0B0B]/90 px-2 py-0.5 text-[10px] font-bold text-[var(--gold-bright)] shadow-md pointer-events-none whitespace-nowrap backdrop-blur-md transition group-hover:scale-110";
        label.innerHTML = `${loc.flag ? loc.flag + " " : ""}${loc.name}`;
        el.appendChild(label);

        // Smart Zoom Visibility
        const updateZoomVisibility = () => {
          const z = map.getZoom();
          const minZ = loc.minZoom ?? 0;
          const maxZ = loc.maxZoom ?? 20;
          if (z >= minZ && z <= maxZ) {
            el.style.opacity = "1";
            el.style.pointerEvents = "auto";
          } else {
            el.style.opacity = "0";
            el.style.pointerEvents = "none";
          }
        };

        map.on("zoom", updateZoomVisibility);
        updateZoomVisibility();

        el.addEventListener("mouseenter", (e) => {
          const rect = mapContainerRef.current?.getBoundingClientRect();
          if (rect) {
            setHoverInfo({ node: loc, x: e.clientX - rect.left, y: e.clientY - rect.top });
          }
        });

        el.addEventListener("mouseleave", () => setHoverInfo(null));

        el.addEventListener("click", (e) => {
          e.stopPropagation();
          selectLocationNode(loc);
        });

        new maplibregl.Marker({ element: el })
          .setLngLat(loc.lngLat)
          .addTo(map);
      });
    });

    // Global Click Reverse-Geocoding Listener for ANY point on Earth
    map.on("click", (e) => {
      reverseGeocodeGISPoint(e.lngLat.lng, e.lngLat.lat);
    });

    mapRef.current = map;

    const observer = new ResizeObserver(() => {
      map.resize();
    });
    observer.observe(mapContainerRef.current);

    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Sync Radius Circle
  useEffect(() => {
    if (!mapRef.current || !selectedNode) return;
    const source = mapRef.current.getSource("radius-circle-source") as maplibregl.GeoJSONSource;
    if (source) {
      const radiusKmVal = RADIUS_KM_NUMBERS[radiusKm] || 50;
      const circlePolygon = createGeoJSONCircle(selectedNode.lngLat, radiusKmVal);
      source.setData(circlePolygon);
    }
  }, [selectedNode, radiusKm]);

  // Sync Heatmap Visibility
  useEffect(() => {
    if (!mapRef.current) return;
    if (mapRef.current.getLayer("trend-heatmap-layer")) {
      mapRef.current.setLayoutProperty(
        "trend-heatmap-layer",
        "visibility",
        showHeatmap ? "visible" : "none"
      );
    }
  }, [showHeatmap]);

  // Select location, Fly camera smoothly, and Update Right Panel & Breadcrumbs
  const selectLocationNode = useCallback(
    (loc: LocationNode) => {
      setSelectedNode(loc);

      const buildChain = (n: LocationNode): LocationNode[] => {
        if (!n.parentId) return [n];
        const parent = GLOBAL_LOCATIONS.find((x) => x.id === n.parentId);
        return parent ? [...buildChain(parent), n] : [n];
      };
      setBreadcrumb(buildChain(loc));

      if (mapRef.current) {
        if (loc.bounds) {
          mapRef.current.fitBounds(loc.bounds, { padding: 40, duration: 1600 });
        } else {
          const targetZoom =
            loc.level === "landmark"
              ? 16
              : loc.level === "neighborhood"
              ? 14
              : loc.level === "city"
              ? 11
              : loc.level === "district"
              ? 9
              : loc.level === "state"
              ? 7
              : loc.level === "country"
              ? 4.5
              : 3;

          mapRef.current.flyTo({
            center: loc.lngLat,
            zoom: targetZoom,
            duration: 1600,
            essential: true,
          });
        }

        highlightSelectedMarker(loc.lngLat);
      }

      const summary = mapLocationToSummary(loc);
      onSelectCountry(summary);
    },
    [onSelectCountry]
  );

  // Render Gold Selection Target Marker Ring
  const highlightSelectedMarker = (lngLat: [number, number]) => {
    if (!mapRef.current) return;
    if (selectedMarkerRef.current) selectedMarkerRef.current.remove();

    const ringEl = document.createElement("div");
    ringEl.className = "pointer-events-none relative flex items-center justify-center";
    ringEl.innerHTML = `
      <div class="h-20 w-20 rounded-full border-2 border-[var(--gold-bright)] bg-[#D4AF37]/20 animate-ping"></div>
      <div class="absolute h-10 w-10 rounded-full border-2 stroke-dashed border-[var(--gold-bright)] animate-spin"></div>
      <div class="absolute h-3 w-3 rounded-full bg-[var(--gold-bright)] shadow-[0_0_20px_#FACC15]"></div>
    `;

    selectedMarkerRef.current = new maplibregl.Marker({ element: ringEl })
      .setLngLat(lngLat)
      .addTo(mapRef.current);
  };

  // Worldwide GIS Search with Geocoding API + Local Database Match
  const handleSearch = async (val: string) => {
    setSearchQuery(val);
    if (!val.trim()) {
      setSearchResults([]);
      setDropdownOpen(false);
      return;
    }

    const localMatches = GLOBAL_LOCATIONS.filter((n) =>
      n.name.toLowerCase().includes(val.toLowerCase())
    ).slice(0, 5);

    setSearchResults(localMatches);
    setDropdownOpen(true);

    if (val.length >= 3) {
      setIsSearching(true);
      try {
        const res = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(val)}&limit=5`
        );
        if (res.ok) {
          const apiData = await res.json();
          const apiMatches: LocationNode[] = apiData.map((item: any) => ({
            id: `GEO-${item.place_id}`,
            name: item.display_name.split(",")[0],
            level: item.type === "administrative" ? "state" : "city",
            lngLat: [parseFloat(item.lon), parseFloat(item.lat)],
            flag: "📍",
            population: "~1.2 Million",
            timeZone: "Local Time",
            languages: ["English"],
            majorPlatforms: ["Instagram Reels", "YouTube Shorts", "TikTok"],
            trendScore: 94,
            activeCreators: 10500,
            totalActiveTrends: 88,
            activityLevel: "High",
            trendIntensity: "94% High Surge",
            heatColor: "red",
          }));

          setSearchResults((prev) => {
            const combined = [...prev];
            apiMatches.forEach((m) => {
              if (!combined.some((c) => c.name.toLowerCase() === m.name.toLowerCase())) {
                combined.push(m);
              }
            });
            return combined.slice(0, 8);
          });
        }
      } catch {
        // Fallback to local database matches
      } finally {
        setIsSearching(false);
      }
    }
  };

  const detectCurrentLocation = () => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const coords: [number, number] = [pos.coords.longitude, pos.coords.latitude];
        if (mapRef.current) {
          mapRef.current.flyTo({ center: coords, zoom: 10, duration: 1800 });
          if (userMarkerRef.current) userMarkerRef.current.remove();

          const pinEl = document.createElement("div");
          pinEl.className = "h-5 w-5 rounded-full bg-blue-500 border-2 border-white shadow-[0_0_15px_#3B82F6] animate-pulse";
          userMarkerRef.current = new maplibregl.Marker({ element: pinEl })
            .setLngLat(coords)
            .addTo(mapRef.current);
        }
      },
      () => {
        if (mapRef.current) {
          mapRef.current.flyTo({ center: [78.9629, 20.5937], zoom: 5 });
        }
      }
    );
  };

  const resetMap = () => {
    if (mapRef.current) {
      mapRef.current.flyTo({ center: [20, 20], zoom: 1.6, duration: 1200 });
      const radiusSource = mapRef.current.getSource("radius-circle-source") as maplibregl.GeoJSONSource;
      if (radiusSource) {
        radiusSource.setData({ type: "FeatureCollection", features: [] });
      }
    }
    if (selectedMarkerRef.current) selectedMarkerRef.current.remove();
    setSelectedNode(null);
    setBreadcrumb([]);
    setSearchQuery("");
    setDropdownOpen(false);
  };

  return (
    <div className="relative h-full w-full min-h-[500px] overflow-hidden rounded-2xl border border-[oklch(0.85_0.155_86/0.18)] bg-[#0B0B0B] shadow-2xl">
      {/* Native Mapbox GL Vector Map Container */}
      <div ref={mapContainerRef} className="h-full w-full" />

      {/* Top Left Search Bar & Breadcrumb Controls */}
      <div className="absolute top-3 left-3 right-3 z-30 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        <div className="flex flex-wrap items-center gap-2 pointer-events-auto">
          {/* Location Search Bar with GIS Autocomplete & FlyTo */}
          <div className="relative min-w-[240px]">
            <div className="flex items-center gap-2 rounded-xl border border-[oklch(0.85_0.155_86/0.35)] bg-[#0B0B0B]/90 backdrop-blur-md px-3 py-1.5 shadow-md">
              <Search className="h-3.5 w-3.5 shrink-0 text-[var(--gold-dim)]" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="Search country, state, city, landmark..."
                className="w-full bg-transparent text-[11px] text-foreground placeholder:text-muted-foreground focus:outline-none"
              />
              {isSearching ? (
                <Loader2 className="h-3 w-3 animate-spin text-[var(--gold-bright)]" />
              ) : searchQuery ? (
                <button onClick={() => setSearchQuery("")}>
                  <X className="h-3 w-3 text-muted-foreground hover:text-foreground" />
                </button>
              ) : null}
            </div>

            {/* GIS Autocomplete Dropdown */}
            {dropdownOpen && searchResults.length > 0 && (
              <div className="absolute left-0 right-0 top-full mt-1 max-h-56 overflow-y-auto rounded-xl border border-[oklch(0.85_0.155_86/0.3)] bg-[#111111]/95 backdrop-blur-xl p-1 shadow-2xl z-50">
                {searchResults.map((node) => (
                  <button
                    key={node.id}
                    onClick={() => {
                      selectLocationNode(node);
                      setSearchQuery(node.name);
                      setDropdownOpen(false);
                    }}
                    className="flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[11px] text-foreground transition hover:bg-[oklch(0.85_0.155_86/0.15)]"
                  >
                    <Globe className="h-3 w-3 shrink-0 text-[var(--gold-bright)]" />
                    <span className="font-medium truncate">
                      {node.flag ? node.flag + " " : ""}{node.name}
                    </span>
                    <span className="ml-auto text-[9px] text-muted-foreground capitalize shrink-0">
                      {LEVEL_LABELS[node.level]}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Breadcrumb Administrative Hierarchy */}
          <div className="flex items-center gap-1 rounded-xl border border-[oklch(0.85_0.155_86/0.2)] bg-[#0B0B0B]/90 backdrop-blur-md px-3 py-1.5 flex-wrap">
            <button
              onClick={resetMap}
              className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-[var(--gold-bright)] font-medium"
            >
              <Globe className="h-3 w-3" />
              <span>World</span>
            </button>
            {breadcrumb.map((node, idx) => (
              <React.Fragment key={node.id}>
                <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/40 shrink-0" />
                <button
                  onClick={() => selectLocationNode(node)}
                  className={`text-[10px] transition hover:text-[var(--gold-bright)] ${
                    idx === breadcrumb.length - 1
                      ? "text-[var(--gold-bright)] font-bold"
                      : "text-muted-foreground"
                  }`}
                >
                  {node.flag ? node.flag + " " : ""}{node.name}
                </button>
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* Top Right Layer Controls */}
        <div className="flex items-center gap-1.5 pointer-events-auto">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            title="Toggle Trend Heatmap"
            className={`flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-[11px] font-semibold transition backdrop-blur-md ${
              showHeatmap
                ? "border-[var(--gold-bright)] bg-[oklch(0.85_0.155_86/0.2)] text-[var(--gold-bright)]"
                : "border-white/10 bg-[#0B0B0B]/90 text-muted-foreground hover:text-foreground"
            }`}
          >
            <Flame className="h-3.5 w-3.5" />
            <span>Heatmap</span>
          </button>

          <button
            onClick={detectCurrentLocation}
            title="Detect Current Location"
            className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-[#0B0B0B]/90 px-3 py-1.5 text-[11px] font-semibold text-muted-foreground transition hover:border-[var(--gold-bright)] hover:text-[var(--gold-bright)] backdrop-blur-md"
          >
            <Compass className="h-3.5 w-3.5" />
            <span>GPS</span>
          </button>
        </div>
      </div>

      {/* Rich Data Hover Tooltip */}
      {hoverInfo && (
        <div
          className="absolute z-50 pointer-events-none rounded-xl border border-[oklch(0.85_0.155_86/0.4)] bg-[#0C0C0E]/95 px-3.5 py-2.5 text-xs backdrop-blur-xl shadow-2xl space-y-1"
          style={{
            left: Math.min(hoverInfo.x + 15, 760),
            top: Math.max(hoverInfo.y - 50, 15),
          }}
        >
          <div className="flex items-center gap-2 font-bold text-foreground">
            {hoverInfo.node.flag && <span>{hoverInfo.node.flag}</span>}
            <span>{hoverInfo.node.name}</span>
            <span className="ml-auto text-[9px] uppercase tracking-wider text-[var(--gold-dim)] font-normal">
              {LEVEL_LABELS[hoverInfo.node.level]}
            </span>
          </div>
          <div className="flex flex-col gap-0.5 text-[10px] text-muted-foreground">
            <div className="flex justify-between gap-4">
              <span>Trend Score:</span>
              <strong className="text-[var(--gold-bright)]">
                {hoverInfo.node.trendScore || 95}%
              </strong>
            </div>
            <div className="flex justify-between gap-4">
              <span>Creator Activity:</span>
              <strong className="text-emerald-400">
                {(hoverInfo.node.activeCreators || 12540).toLocaleString()} Active
              </strong>
            </div>
            <div className="flex justify-between gap-4">
              <span>Trend Intensity:</span>
              <strong className="text-amber-400">
                {hoverInfo.node.trendIntensity || "Peak Intensity"}
              </strong>
            </div>
          </div>
          <div className="text-[9px] text-[var(--gold-bright)] font-semibold pt-1 flex items-center gap-1 border-t border-white/10 mt-1">
            <Sparkles className="h-2.5 w-2.5" /> Click to Analyze
          </div>
        </div>
      )}

      {/* Bottom Right Controls */}
      <div className="absolute bottom-4 right-4 z-30 flex flex-col gap-2 rounded-xl border border-white/10 bg-[#0B0B0B]/90 p-1.5 backdrop-blur-md shadow-xl">
        <button
          onClick={() => mapRef.current?.zoomIn()}
          title="Zoom In"
          className="grid h-8 w-8 place-items-center rounded-lg border border-white/5 bg-black/40 text-muted-foreground transition hover:border-[var(--gold)] hover:text-[var(--gold-bright)]"
        >
          <ZoomIn className="h-4 w-4" />
        </button>
        <button
          onClick={() => mapRef.current?.zoomOut()}
          title="Zoom Out"
          className="grid h-8 w-8 place-items-center rounded-lg border border-white/5 bg-black/40 text-muted-foreground transition hover:border-[var(--gold)] hover:text-[var(--gold-bright)]"
        >
          <ZoomOut className="h-4 w-4" />
        </button>
        <button
          onClick={resetMap}
          title="Reset View"
          className="grid h-8 w-8 place-items-center rounded-lg border border-white/5 bg-black/40 text-muted-foreground transition hover:border-[var(--gold)] hover:text-[var(--gold-bright)]"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Bottom Left Selection Badge */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 z-30 flex items-center gap-2.5 rounded-full border border-[oklch(0.85_0.155_86/0.4)] bg-[#0B0B0B]/95 px-4 py-2 text-xs font-semibold text-foreground backdrop-blur-md shadow-xl">
          {selectedNode.flag && <span className="text-base">{selectedNode.flag}</span>}
          <span className="gold-text">{selectedNode.name}</span>
          <span className="h-3 w-[1px] bg-white/20" />
          <span className="text-[10px] capitalize text-muted-foreground">
            {LEVEL_LABELS[selectedNode.level]}
          </span>
          <span className="h-3 w-[1px] bg-white/20" />
          <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <Navigation className="h-3 w-3 text-[var(--gold-bright)]" /> {radiusKm}
          </span>
        </div>
      )}
    </div>
  );
}
