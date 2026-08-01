import React, { useState } from "react";
import {
  Search,
  MapPin,
  Sparkles,
  ArrowRight,
  Globe,
  Layers,
  Activity,
  Users,
  Clock,
} from "lucide-react";
import { Link } from "@tanstack/react-router";
import { CountrySummary } from "@/lib/api";
import { InteractiveWorldMap } from "./InteractiveWorldMap";

interface LandingScreenProps {
  countries: CountrySummary[];
  selectedCountry: CountrySummary | null;
  onSelectCountry: (country: CountrySummary) => void;
  radiusKm: string;
  onChangeRadius: (radius: string) => void;
  onAnalyzeTrends: () => void;
}

const RADIUS_OPTIONS = [
  "5 km",
  "10 km",
  "25 km",
  "50 km",
  "100 km",
  "District",
  "State",
  "Country",
  "Continent",
];

export function LandingScreen({
  countries,
  selectedCountry,
  onSelectCountry,
  radiusKm,
  onChangeRadius,
  onAnalyzeTrends,
}: LandingScreenProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<CountrySummary[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearchQuery(val);
    if (!val.trim()) {
      setSearchResults([]);
      setIsDropdownOpen(false);
      return;
    }
    const filtered = countries.filter(
      (c) =>
        c.name.toLowerCase().includes(val.toLowerCase()) ||
        c.code.toLowerCase().includes(val.toLowerCase()) ||
        c.languages.some((l) => l.toLowerCase().includes(val.toLowerCase())),
    );
    setSearchResults(filtered);
    setIsDropdownOpen(true);
  };

  const handleSelectSearchResult = (country: CountrySummary) => {
    onSelectCountry(country);
    setSearchQuery(country.name);
    setIsDropdownOpen(false);
  };

  return (
    <div className="flex flex-col gap-6 h-full">
      {/* Top Header & Search Bar Bar */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between rounded-2xl border border-[oklch(0.85_0.155_86/0.15)] bg-gradient-to-r from-[#0d0d0d] via-[#121212] to-[#0a0a0a] p-4 lg:p-6 shadow-[var(--shadow-premium)]">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-[var(--gold-bright)] animate-pulse">✦</span>
            <h1 className="font-display text-xl lg:text-2xl tracking-[0.18em] text-foreground">
              AI CREATOR <span className="gold-text">TREND INTELLIGENCE</span>
            </h1>
          </div>
          <p className="text-xs text-muted-foreground">
            Discover location-based social media trends across Instagram Reels & YouTube Shorts
            before creating content.
          </p>
        </div>

        {/* Interactive Controls Bar: Search Box + Radius Selector + Analyze Button */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Location Search Box */}
          <div className="relative min-w-[240px] flex-1 sm:flex-none">
            <div className="flex items-center gap-2 rounded-xl border border-[oklch(0.85_0.155_86/0.25)] bg-[#111111] px-3.5 py-2.5 text-xs text-foreground focus-within:border-[var(--gold-bright)]">
              <Search className="h-4 w-4 shrink-0 text-[var(--gold-dim)]" />
              <input
                type="text"
                value={searchQuery}
                onChange={handleSearchChange}
                placeholder="Search location..."
                className="w-full bg-transparent placeholder:text-muted-foreground focus:outline-none text-xs"
              />
            </div>

            {isDropdownOpen && searchResults.length > 0 && (
              <div className="absolute left-0 right-0 top-full z-50 mt-1 max-h-48 overflow-y-auto rounded-xl border border-[oklch(0.85_0.155_86/0.3)] bg-[#121212]/95 p-1.5 shadow-2xl backdrop-blur-xl">
                {searchResults.map((country) => (
                  <button
                    key={country.id}
                    onClick={() => handleSelectSearchResult(country)}
                    className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs text-foreground transition hover:bg-[oklch(0.85_0.155_86/0.15)]"
                  >
                    <span className="flex items-center gap-2">
                      <span>{country.flag}</span>
                      <span className="font-medium">{country.name}</span>
                    </span>
                    <span className="text-[10px] text-[var(--gold-bright)]">
                      Score: {country.current_trend_score}%
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Radius Dropdown Selector */}
          <div className="flex items-center gap-2 rounded-xl border border-[oklch(0.85_0.155_86/0.25)] bg-[#111111] px-3.5 py-2.5 text-xs text-foreground">
            <MapPin className="h-4 w-4 text-[var(--gold-bright)]" />
            <select
              value={radiusKm}
              onChange={(e) => onChangeRadius(e.target.value)}
              className="bg-transparent font-medium text-foreground focus:outline-none cursor-pointer text-xs"
            >
              {RADIUS_OPTIONS.map((r) => (
                <option key={r} value={r} className="bg-[#111] text-foreground">
                  {r}
                </option>
              ))}
            </select>
          </div>

          {/* Action Button: Analyze Trends */}
          <button
            onClick={onAnalyzeTrends}
            disabled={!selectedCountry}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-[var(--gold)] via-[var(--gold-bright)] to-[var(--gold-dim)] px-5 py-2.5 text-xs font-bold text-black shadow-[0_0_25px_-4px_var(--gold-bright)] transition hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles className="h-4 w-4 fill-black" />
            <span>Analyze Trends</span>
          </button>
        </div>
      </div>

      {/* Main Map View + Right Side Summary Panel */}
      <div className="grid flex-1 gap-6 xl:grid-cols-[1fr_340px] min-h-[520px]">
        {/* Interactive World Map */}
        <div className="relative h-full min-h-[460px] w-full">
          <InteractiveWorldMap
            countries={countries}
            selectedCountry={selectedCountry}
            onSelectCountry={onSelectCountry}
            radiusKm={radiusKm}
          />
        </div>

        {/* Right Side Summary Panel when user selects a location */}
        <div className="flex flex-col justify-between rounded-2xl border border-[oklch(0.85_0.155_86/0.2)] bg-gradient-to-b from-[#121212] via-[#0d0d0d] to-black p-5 shadow-[var(--shadow-premium)]">
          {selectedCountry ? (
            <div className="flex flex-col h-full justify-between gap-6">
              <div className="space-y-4">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-white/10 pb-4">
                  <div className="flex items-center gap-3">
                    <span className="text-3xl">{selectedCountry.flag || "📍"}</span>
                    <div>
                      <h3 className="font-display text-lg tracking-wider text-foreground">
                        {selectedCountry.name}
                      </h3>
                      <p className="text-[10px] uppercase tracking-widest text-[var(--gold-dim)]">
                        📍 Location Intelligence Summary
                      </p>
                    </div>
                  </div>
                  <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.4)] bg-[oklch(0.85_0.155_86/0.12)] px-2.5 py-1 text-center">
                    <div className="text-[9px] uppercase tracking-wider text-muted-foreground">
                      Score
                    </div>
                    <div className="font-display text-base font-bold text-[var(--gold-bright)]">
                      {selectedCountry.current_trend_score}%
                    </div>
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <Users className="h-3.5 w-3.5 text-[var(--gold-dim)]" /> Population
                    </div>
                    <div className="mt-1 font-semibold text-foreground">
                      {selectedCountry.population}
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <Clock className="h-3.5 w-3.5 text-[var(--gold-dim)]" /> Time Zone
                    </div>
                    <div className="mt-1 font-semibold text-foreground truncate">
                      {selectedCountry.time_zone}
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3 col-span-2">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <Globe className="h-3.5 w-3.5 text-[var(--gold-dim)]" /> Languages
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {selectedCountry.languages.map((lang) => (
                        <span
                          key={lang}
                          className="rounded border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] text-foreground"
                        >
                          {lang}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3 col-span-2">
                    <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <Layers className="h-3.5 w-3.5 text-[var(--gold-dim)]" /> Major Social
                      Platforms
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {selectedCountry.major_platforms.map((p) => (
                        <span
                          key={p}
                          className="rounded-md border border-[oklch(0.85_0.155_86/0.3)] bg-[oklch(0.85_0.155_86/0.08)] px-2 py-0.5 text-[10px] text-[var(--gold-bright)]"
                        >
                          {p}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Total Active Trends Highlight */}
                <div className="flex items-center justify-between rounded-xl border border-[oklch(0.85_0.155_86/0.3)] bg-gradient-to-r from-[#181307] to-[#0a0a0a] p-3.5">
                  <div className="flex items-center gap-2.5">
                    <Activity className="h-5 w-5 text-[var(--gold-bright)] animate-pulse" />
                    <div>
                      <div className="text-xs font-semibold text-foreground">
                        Total Active Trends
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        Instagram Reels & YouTube Shorts
                      </div>
                    </div>
                  </div>
                  <span className="font-display text-lg font-bold text-[var(--gold-bright)]">
                    {selectedCountry.total_active_trends}
                  </span>
                </div>
              </div>

              {/* Bottom Deep Intelligence Link */}
              <Link
                to="/creator-intelligence/country/$country"
                params={{ country: selectedCountry.code }}
                className="flex items-center justify-between rounded-xl border border-[oklch(0.85_0.155_86/0.4)] bg-gradient-to-r from-[oklch(0.85_0.155_86/0.18)] to-transparent px-4 py-3 text-xs font-semibold text-[var(--gold-bright)] transition hover:border-[var(--gold-bright)] hover:bg-[oklch(0.85_0.155_86/0.25)]"
              >
                <span>View Detailed Location Intelligence</span>
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center text-center p-6 space-y-3">
              <div className="grid h-12 w-12 place-items-center rounded-full border border-[oklch(0.85_0.155_86/0.3)] bg-white/5">
                <Globe className="h-6 w-6 text-[var(--gold-bright)]" />
              </div>
              <h4 className="font-display text-base text-foreground">Select a Location</h4>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Click any country, state, district, city, or neighborhood on the map or search to view location
                intelligence and analyze trend velocity.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
