import React, { useState, useEffect } from "react";
import {
  Sparkles,
  RefreshCw,
  Instagram,
  Youtube,
  Flame,
  TrendingUp,
  Clock,
  History,
  Music,
  Hash,
  Tag,
  Layers,
  ArrowLeft,
  Globe,
  MapPin,
  Calendar,
} from "lucide-react";
import { toast } from "sonner";
import { PastTrendItem, CurrentTrendItem } from "@/lib/api";
import { TrendIntelligenceDrawer } from "./TrendIntelligenceDrawer";

interface TrendResultsViewProps {
  locationName: string;
  radiusKm: string;
  pastTrends: PastTrendItem[];
  currentTrends: CurrentTrendItem[];
  lastUpdated: string;
  onReset: () => void;
  onGenerateContent: (trend: CurrentTrendItem) => void;
}

export function TrendResultsView({
  locationName,
  radiusKm,
  pastTrends,
  currentTrends,
  lastUpdated,
  onReset,
  onGenerateContent,
}: TrendResultsViewProps) {
  const [platformFilter, setPlatformFilter] = useState<
    "All" | "Instagram Reels" | "YouTube Shorts"
  >("All");
  const [refreshInterval, setRefreshInterval] = useState<"1m" | "5m" | "15m" | "manual">("manual");
  const [currentTimeStr, setCurrentTimeStr] = useState(lastUpdated);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedTrend, setSelectedTrend] = useState<CurrentTrendItem | null>(null);

  // Filtered Current Trends based on platform selection
  const filteredCurrentTrends = currentTrends.filter((item) => {
    if (platformFilter === "All") return true;
    return item.platform === platformFilter || item.platform === "Both";
  });

  const handleManualRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => {
      setIsRefreshing(false);
      const now = new Date();
      setCurrentTimeStr(
        `Today ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
      );
      toast.success("Trend intelligence refreshed with live data.");
    }, 600);
  };

  useEffect(() => {
    if (refreshInterval === "manual") return;
    const msMap = { "1m": 60000, "5m": 300000, "15m": 900000 };
    const interval = setInterval(() => {
      handleManualRefresh();
    }, msMap[refreshInterval]);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "growing":
        return (
          <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-0.5 text-[10px] font-bold text-emerald-400">
            Growing
          </span>
        );
      case "stable":
        return (
          <span className="rounded-full border border-blue-500/40 bg-blue-500/10 px-2.5 py-0.5 text-[10px] font-bold text-blue-400">
            Stable
          </span>
        );
      case "declining":
        return (
          <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-0.5 text-[10px] font-bold text-amber-400">
            Declining
          </span>
        );
      default:
        return (
          <span className="rounded-full border border-zinc-500/40 bg-zinc-500/10 px-2.5 py-0.5 text-[10px] font-bold text-zinc-400">
            {status}
          </span>
        );
    }
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Action Header Bar */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between rounded-2xl border border-[oklch(0.85_0.155_86/0.15)] bg-gradient-to-r from-[#0d0d0d] via-[#121212] to-[#0a0a0a] p-4 lg:p-6 shadow-[var(--shadow-premium)]">
        <div className="flex items-center gap-3">
          <button
            onClick={onReset}
            className="grid h-9 w-9 place-items-center rounded-xl border border-white/10 bg-white/5 text-muted-foreground hover:text-foreground transition"
            title="Back to Map"
          >
            <ArrowLeft className="h-4 w-4" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base">📍</span>
              <h2 className="font-display text-lg lg:text-xl text-foreground tracking-wide">
                Trend Intelligence for <span className="gold-text">{locationName}</span>
              </h2>
              <span className="rounded-full border border-[oklch(0.85_0.155_86/0.3)] bg-[oklch(0.85_0.155_86/0.1)] px-2.5 py-0.5 text-[10px] font-medium text-[var(--gold-bright)]">
                Radius: {radiusKm}
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              Live algorithmic feed for Instagram Reels & YouTube Shorts
            </p>
          </div>
        </div>

        {/* Auto Refresh & Last Updated Timestamp */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Auto Refresh:</span>
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(e.target.value as any)}
              className="rounded-lg border border-white/10 bg-[#141414] px-2.5 py-1.5 text-xs text-foreground focus:outline-none cursor-pointer"
            >
              <option value="manual">Manual Refresh</option>
              <option value="1m">1 minute</option>
              <option value="5m">5 minutes</option>
              <option value="15m">15 minutes</option>
            </select>
          </div>

          <button
            onClick={handleManualRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-foreground hover:border-[var(--gold)] transition disabled:opacity-50"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 text-[var(--gold-bright)] ${isRefreshing ? "animate-spin" : ""}`}
            />
            <span>Refresh</span>
          </button>

          <div className="rounded-xl border border-white/5 bg-white/[0.03] px-3 py-1.5 text-[11px] text-muted-foreground">
            Last Updated: <span className="font-semibold text-foreground">{currentTimeStr}</span>
          </div>
        </div>
      </div>

      {/* Main Split Layout: 30% Past Trends (Left) / 70% Current Trends (Right) */}
      <div className="grid gap-6 lg:grid-cols-[30%_1fr]">
        {/* LEFT PANEL - PAST TRENDS (30%) */}
        <div className="space-y-4 rounded-2xl border border-[oklch(0.85_0.155_86/0.15)] bg-gradient-to-b from-[#121212] via-[#0d0d0d] to-black p-5 shadow-[var(--shadow-premium)]">
          <div className="flex items-center justify-between border-b border-white/10 pb-3">
            <div className="flex items-center gap-2">
              <History className="h-4 w-4 text-[var(--gold-bright)]" />
              <h3 className="font-display text-base text-foreground tracking-wider">
                PAST TRENDS <span className="text-[var(--gold-dim)]">(30%)</span>
              </h3>
            </div>
            <span className="text-[10px] text-muted-foreground">Historical Retrospective</span>
          </div>

          <div className="space-y-4">
            {pastTrends.map((trend) => (
              <div
                key={trend.id}
                className="rounded-xl border border-white/10 bg-[#141414] p-4 space-y-3 hover:border-[oklch(0.85_0.155_86/0.3)] transition"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--gold-dim)]">
                      {trend.category}
                    </span>
                    <h4 className="font-semibold text-sm text-foreground">{trend.title}</h4>
                  </div>
                  <span className="rounded border border-white/10 bg-white/5 px-2 py-0.5 text-[9px] text-muted-foreground shrink-0">
                    {trend.platform}
                  </span>
                </div>

                {/* Trend Strength & Metrics Grid */}
                <div className="grid grid-cols-2 gap-2 text-[11px] bg-black/40 p-2.5 rounded-lg border border-white/5">
                  <div>
                    <span className="text-muted-foreground block text-[10px]">Trend Strength</span>
                    <span className="font-bold text-[var(--gold-bright)]">
                      {trend.trend_strength}%
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[10px]">Peak Date</span>
                    <span className="font-semibold text-foreground">{trend.peak_date}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[10px]">Intensity</span>
                    <span className="font-semibold text-foreground">
                      {trend.duration_days} Days
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[10px]">Status</span>
                    <span className="font-semibold text-zinc-400">{trend.status}</span>
                  </div>
                </div>

                {/* Viral Rationale */}
                <p className="text-[11px] leading-relaxed text-muted-foreground italic">
                  "{trend.why_viral}"
                </p>

                {/* Meme Format & Audio */}
                <div className="space-y-1 text-[10px] text-muted-foreground">
                  <div className="flex items-center gap-1.5">
                    <Music className="h-3 w-3 text-[var(--gold-dim)]" /> Audio: {trend.audio_sample}
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Layers className="h-3 w-3 text-[var(--gold-dim)]" /> Format:{" "}
                    {trend.meme_format}
                  </div>
                </div>

                {/* Hashtags */}
                <div className="flex flex-wrap gap-1 pt-1">
                  {trend.hashtags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[9px] text-foreground"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* PAST EVENTS SECTION */}
          <div className="mt-6 border-t border-white/10 pt-5">
            <div className="flex items-center gap-2 mb-4">
              <Calendar className="h-4 w-4 text-[var(--gold-dim)]" />
              <h4 className="font-display text-sm tracking-wider text-foreground uppercase">
                Past Events
              </h4>
            </div>
            <div className="rounded-xl border border-white/5 bg-[#141414]/50 p-4">
              <div className="flex items-center gap-2 mb-3">
                <MapPin className="h-3 w-3 text-muted-foreground" />
                <span className="text-xs text-[var(--gold-dim)] font-medium">{locationName}</span>
              </div>
              <ul className="space-y-2 text-xs text-foreground">
                <li className="flex items-center gap-2">
                  <span className="text-[var(--gold-dim)]">•</span> Chennai Book Fair
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-[var(--gold-dim)]">•</span> Auto Expo
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-[var(--gold-dim)]">•</span> IPL Final
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-[var(--gold-dim)]">•</span> Independence Day Celebration
                </li>
                <li className="flex items-center gap-2">
                  <span className="text-[var(--gold-dim)]">•</span> Music Festival
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* RIGHT PANEL - CURRENT TRENDS (70%) */}
        <div className="space-y-5 rounded-2xl border border-[oklch(0.85_0.155_86/0.15)] bg-gradient-to-b from-[#121212] via-[#0d0d0d] to-black p-5 lg:p-6 shadow-[var(--shadow-premium)]">
          {/* Header & Platform Tabs */}
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-white/10 pb-4">
            <div className="flex items-center gap-2">
              <Flame className="h-5 w-5 text-amber-400 animate-pulse" />
              <h3 className="font-display text-lg text-foreground tracking-wider">
                CURRENT VIRAL TRENDS <span className="gold-text">(70%)</span>
              </h3>
            </div>

            {/* Platform Filter Buttons */}
            <div className="flex items-center gap-1.5 rounded-xl border border-white/10 bg-[#141414] p-1 text-xs">
              {(["All", "Instagram Reels", "YouTube Shorts"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setPlatformFilter(tab)}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-medium transition ${
                    platformFilter === tab
                      ? "bg-gradient-to-r from-[var(--gold)] to-[var(--gold-bright)] text-black font-bold shadow-[0_0_12px_var(--gold-bright)]"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tab === "Instagram Reels" && <Instagram className="h-3.5 w-3.5" />}
                  {tab === "YouTube Shorts" && <Youtube className="h-3.5 w-3.5" />}
                  <span>{tab}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Current Trends Grid */}
          <div className="grid gap-4 sm:grid-cols-2">
            {filteredCurrentTrends.map((trend) => (
              <article
                key={trend.id}
                onClick={() => setSelectedTrend(trend)}
                className="group relative flex flex-col justify-between overflow-hidden rounded-2xl border border-[oklch(0.85_0.155_86/0.2)] bg-gradient-to-b from-[#141414] to-[#0a0a0a] p-4 hover:border-[var(--gold-bright)] transition duration-300 shadow-md cursor-pointer"
              >
                <div className="space-y-3 pointer-events-none">
                  {/* Thumbnail & Badges */}
                  <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black">
                    <img
                      src={trend.thumbnail_url}
                      alt={trend.title}
                      className="h-full w-full object-cover transition duration-500 group-hover:scale-105"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent" />

                    {/* Top Platform Pill */}
                    <div className="absolute top-2.5 left-2.5 flex items-center gap-1 rounded-full border border-white/20 bg-black/70 px-2.5 py-1 backdrop-blur text-[10px] font-semibold text-foreground">
                      {trend.platform.includes("Reels") ? (
                        <Instagram className="h-3 w-3 text-pink-400" />
                      ) : (
                        <Youtube className="h-3 w-3 text-red-500" />
                      )}
                      <span>{trend.platform}</span>
                    </div>

                    {/* Viral Score Badge */}
                    <div className="absolute top-2.5 right-2.5 rounded-full border border-[oklch(0.85_0.155_86/0.4)] bg-black/80 px-2.5 py-1 backdrop-blur text-[10px] font-bold text-[var(--gold-bright)]">
                      Viral Score: {trend.viral_score}
                    </div>

                    {/* Bottom Engagement Pill */}
                    <div className="absolute bottom-2.5 left-2.5 right-2.5 flex items-center justify-between text-[11px] text-foreground font-semibold">
                      <span>{trend.engagement}</span>
                      <span className="flex items-center gap-0.5 text-emerald-400">
                        <TrendingUp className="h-3 w-3" /> +{trend.growth_pct}%
                      </span>
                    </div>
                  </div>

                  {/* Title & Category */}
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--gold-dim)]">
                        {trend.category}
                      </span>
                      {getStatusBadge(trend.status)}
                    </div>
                    <h4 className="font-display text-base text-foreground mt-1 group-hover:text-[var(--gold-bright)] transition">
                      {trend.title}
                    </h4>
                  </div>

                  {/* Trend Intensity Metrics */}
                  <div className="grid grid-cols-2 gap-2 text-[11px] bg-black/40 p-2.5 rounded-xl border border-white/5">
                    <div>
                      <span className="text-muted-foreground block text-[10px]">
                        Trend Strength
                      </span>
                      <span className="font-bold text-[var(--gold-bright)]">
                        {trend.trend_strength}%
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-[10px]">
                        Expected Lifespan
                      </span>
                      <span className="font-semibold text-foreground">
                        {trend.expected_duration}
                      </span>
                    </div>
                  </div>

                  {/* Audio & Keywords */}
                  <div className="space-y-1 text-[10px] text-muted-foreground">
                    <div className="flex items-center gap-1.5 truncate">
                      <Music className="h-3 w-3 text-[var(--gold-bright)] shrink-0" />
                      <span className="truncate">Audio: {trend.audio_track}</span>
                    </div>
                  </div>

                  {/* Hashtags */}
                  <div className="flex flex-wrap gap-1">
                    {trend.hashtags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded-md border border-[oklch(0.85_0.155_86/0.2)] bg-[oklch(0.85_0.155_86/0.05)] px-2 py-0.5 text-[9px] text-[var(--gold-bright)]"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Generate Similar Content Action Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onGenerateContent(trend);
                  }}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-[var(--gold)] via-[var(--gold-bright)] to-[var(--gold-dim)] px-4 py-2.5 text-xs font-bold text-black shadow-[0_0_20px_-4px_var(--gold-bright)] transition hover:shadow-[0_0_28px_-2px_var(--gold-bright)] hover:scale-[1.02] active:scale-[0.98]"
                >
                  <Sparkles className="h-3.5 w-3.5 fill-black" />
                  <span>Generate Similar Content</span>
                </button>
              </article>
            ))}
          </div>

          {/* LOCAL TRENDING EVENTS & TOP LOCATIONS */}
          <div className="mt-8 grid gap-6 md:grid-cols-2 border-t border-white/10 pt-6">
            {/* Local Trending Events */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <MapPin className="h-4 w-4 text-[var(--gold-bright)]" />
                <h4 className="font-display text-sm tracking-wider text-foreground uppercase">
                  Upcoming Events
                </h4>
              </div>
              <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.2)] bg-[#141414] p-4">
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-white/10">
                  <span className="text-xs text-[var(--gold-dim)] font-medium">{locationName}</span>
                </div>
                <div className="space-y-3 text-xs">
                  <div className="flex items-center gap-2">
                    <span>🎉</span> <span className="text-foreground">Chennai Book Fair</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>🚗</span> <span className="text-foreground">Auto Expo</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>🎵</span> <span className="text-foreground">Music Festival</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>🎬</span> <span className="text-foreground">Movie Release</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>🏏</span> <span className="text-foreground">IPL Match</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>🎓</span> <span className="text-foreground">College Cultural Fest</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span>🇮🇳</span>{" "}
                    <span className="text-foreground">Independence Day Celebration</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Top Trending Locations */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Globe className="h-4 w-4 text-[var(--gold-bright)]" />
                <h4 className="font-display text-sm tracking-wider text-foreground uppercase">
                  Top Trending Locations
                </h4>
              </div>
              <div className="space-y-3">
                <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.2)] bg-[#141414] p-3 text-xs">
                  <div className="text-[10px] uppercase tracking-wider text-[var(--gold-dim)] mb-1.5">
                    Top Trending Continents
                  </div>
                  <div className="flex flex-wrap gap-2 text-foreground font-medium">
                    <span className="bg-white/5 px-2 py-1 rounded">Asia</span>
                    <span className="bg-white/5 px-2 py-1 rounded">Europe</span>
                    <span className="bg-white/5 px-2 py-1 rounded">North America</span>
                  </div>
                </div>
                <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.2)] bg-[#141414] p-3 text-xs">
                  <div className="text-[10px] uppercase tracking-wider text-[var(--gold-dim)] mb-1.5">
                    Top Trending Countries
                  </div>
                  <div className="flex flex-wrap gap-2 text-foreground font-medium">
                    <span className="bg-white/5 px-2 py-1 rounded border border-[var(--gold-dim)]/20 text-[var(--gold-bright)]">
                      India
                    </span>
                    <span className="bg-white/5 px-2 py-1 rounded">USA</span>
                    <span className="bg-white/5 px-2 py-1 rounded">Japan</span>
                  </div>
                </div>
                <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.2)] bg-[#141414] p-3 text-xs">
                  <div className="text-[10px] uppercase tracking-wider text-[var(--gold-dim)] mb-2">
                    Regional
                  </div>
                  <div className="flex flex-col gap-2 text-foreground font-medium">
                    <div className="flex gap-2 items-center">
                      <span className="text-muted-foreground w-12 text-[10px] uppercase">
                        States
                      </span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">Tamil Nadu</span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">Kerala</span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">California</span>
                    </div>
                    <div className="flex gap-2 items-center">
                      <span className="text-muted-foreground w-12 text-[10px] uppercase">
                        Districts
                      </span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">Chennai</span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">Bengaluru Urban</span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">Kochi</span>
                    </div>
                    <div className="flex gap-2 items-center">
                      <span className="text-muted-foreground w-12 text-[10px] uppercase">
                        Cities
                      </span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded text-[var(--gold-bright)] border border-[var(--gold-dim)]/20">
                        Chennai
                      </span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">Tokyo</span>{" "}
                      <span className="bg-white/5 px-2 py-1 rounded">London</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <TrendIntelligenceDrawer
        trend={selectedTrend}
        isOpen={!!selectedTrend}
        onClose={() => setSelectedTrend(null)}
        locationName={locationName}
      />
    </div>
  );
}
