import React, { useState } from "react";
import { Link } from "@tanstack/react-router";
import {
  ArrowLeft,
  Globe,
  Users,
  Flame,
  History,
  TrendingUp,
  Sparkles,
  Instagram,
  Youtube,
  Clock,
  Lightbulb,
  Tag,
  Check,
  Copy,
} from "lucide-react";
import { toast } from "sonner";
import { DetailedCountryIntelligence, CurrentTrendItem } from "@/lib/api";
import { GenerateSimilarContentModal } from "./GenerateSimilarContentModal";
import { api } from "@/lib/api";

interface CountryIntelligenceViewProps {
  data: DetailedCountryIntelligence;
}

export function CountryIntelligenceView({ data }: CountryIntelligenceViewProps) {
  const {
    overview,
    past_trends,
    current_trends,
    top_20_trends,
    historical_timeline,
    trend_forecast,
    trending_reels,
    trending_shorts,
    ai_recommendations,
  } = data;

  const [selectedTrend, setSelectedTrend] = useState<CurrentTrendItem | null>(null);
  const [generatedContent, setGeneratedContent] = useState<any>(null);
  const [isGeneratingModalOpen, setIsGeneratingModalOpen] = useState(false);

  const handleGenerateContent = async (trend: CurrentTrendItem) => {
    setSelectedTrend(trend);
    try {
      toast.loading("Generating original non-plagiarized content script...");
      const res = await api.creatorIntelligence.generateContent(trend.id, trend.title);
      toast.dismiss();
      setGeneratedContent(res);
      setIsGeneratingModalOpen(true);
    } catch (e) {
      toast.dismiss();
      toast.error("Failed to generate content blueprint.");
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Header Banner */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between rounded-2xl border border-[oklch(0.85_0.155_86/0.2)] bg-gradient-to-r from-[#121212] via-[#0d0d0d] to-black p-6 shadow-[var(--shadow-premium)]">
        <div className="flex items-center gap-4">
          <Link
            to="/creator-intelligence"
            className="grid h-10 w-10 place-items-center rounded-xl border border-white/10 bg-white/5 text-muted-foreground hover:text-foreground transition"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div className="flex items-center gap-3">
            <span className="text-4xl">{overview.flag}</span>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-display text-2xl text-foreground tracking-wider">
                  {overview.name}
                </h1>
                <span className="rounded-full border border-[oklch(0.85_0.155_86/0.4)] bg-[oklch(0.85_0.155_86/0.12)] px-3 py-0.5 text-xs font-bold text-[var(--gold-bright)]">
                  Country Trend Score: {overview.current_trend_score}%
                </span>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Deep-dive country intelligence, historical analytics, 90-day forecasts, and AI
                content strategy.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION 1: OVERVIEW METRICS */}
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-[var(--gold-bright)]">✦</span>
          <h2 className="font-display text-lg text-foreground tracking-widest">
            1. COUNTRY <span className="gold-text">OVERVIEW</span>
          </h2>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="rounded-2xl border border-white/10 bg-[#121212] p-4 space-y-1">
            <div className="text-xs text-muted-foreground">Population</div>
            <div className="font-display text-lg font-bold text-foreground">
              {overview.population}
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-[#121212] p-4 space-y-1">
            <div className="text-xs text-muted-foreground">Time Zone</div>
            <div className="font-display text-lg font-bold text-foreground truncate">
              {overview.time_zone}
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-[#121212] p-4 space-y-1">
            <div className="text-xs text-muted-foreground">Languages</div>
            <div className="font-semibold text-sm text-[var(--gold-bright)] truncate">
              {overview.languages.join(", ")}
            </div>
          </div>
          <div className="rounded-2xl border border-white/10 bg-[#121212] p-4 space-y-1">
            <div className="text-xs text-muted-foreground">Total Active Trends</div>
            <div className="font-display text-lg font-bold text-[var(--gold-bright)]">
              {overview.total_active_trends} Active
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: TOP 20 CURRENT TRENDS */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Flame className="h-5 w-5 text-amber-400" />
            <h2 className="font-display text-lg text-foreground tracking-widest">
              2. TOP 20 <span className="gold-text">CURRENT TRENDS</span>
            </h2>
          </div>
          <span className="text-xs text-muted-foreground font-mono">Ranked by Viral Velocity</span>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {top_20_trends.slice(0, 20).map((trend, idx) => (
            <div
              key={trend.id + idx}
              className="group relative flex flex-col justify-between rounded-xl border border-white/10 bg-[#121212] p-3.5 hover:border-[var(--gold-bright)] transition"
            >
              <div className="space-y-2">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="font-bold text-[var(--gold-bright)] font-mono">
                    #{idx + 1} VIRAL
                  </span>
                  <span className="text-muted-foreground">{trend.platform}</span>
                </div>
                <h4 className="font-display text-sm text-foreground truncate group-hover:text-[var(--gold-bright)]">
                  {trend.title}
                </h4>
                <div className="flex items-center justify-between text-[11px] text-muted-foreground bg-black/40 p-2 rounded-lg">
                  <span>{trend.engagement}</span>
                  <span className="text-emerald-400 font-bold">Score: {trend.viral_score}</span>
                </div>
              </div>

              <button
                onClick={() => handleGenerateContent(trend)}
                className="mt-3 flex w-full items-center justify-center gap-1.5 rounded-lg border border-[oklch(0.85_0.155_86/0.3)] bg-[oklch(0.85_0.155_86/0.08)] py-1.5 text-xs text-[var(--gold-bright)] hover:bg-[oklch(0.85_0.155_86/0.2)] transition"
              >
                <Sparkles className="h-3 w-3" />
                <span>Generate Content</span>
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION 3: HISTORICAL TRENDS TIMELINE */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <History className="h-5 w-5 text-[var(--gold-bright)]" />
          <h2 className="font-display text-lg text-foreground tracking-widest">
            3. HISTORICAL <span className="gold-text">TRENDS TIMELINE</span>
          </h2>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {historical_timeline.map((h) => (
            <div
              key={h.period}
              className="rounded-2xl border border-white/10 bg-[#121212] p-4 space-y-2"
            >
              <div className="text-xs font-semibold text-[var(--gold-bright)]">{h.period}</div>
              <div className="font-display text-xl font-bold text-foreground">
                {h.total_viral} Viral Topics
              </div>
              <div className="text-[11px] text-muted-foreground">
                Top Category: {h.top_category}
              </div>
              <div className="text-[11px] font-bold text-emerald-400">{h.growth} Total Growth</div>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION 4: TREND FORECAST */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-emerald-400" />
          <h2 className="font-display text-lg text-foreground tracking-widest">
            4. AI TREND <span className="gold-text">FORECAST</span>
          </h2>
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          {trend_forecast.map((tf) => (
            <div
              key={tf.timeframe}
              className="rounded-2xl border border-[oklch(0.85_0.155_86/0.3)] bg-gradient-to-br from-[#181307] to-black p-5 space-y-2"
            >
              <div className="flex items-center justify-between text-xs text-[var(--gold-dim)]">
                <span className="font-bold">{tf.timeframe}</span>
                <span className="text-emerald-400 font-mono">Confidence: {tf.confidence}</span>
              </div>
              <h3 className="font-display text-base text-foreground">{tf.predicted_top_trend}</h3>
              <p className="text-xs text-muted-foreground">
                High likelihood of momentum acceleration in {overview.name}.
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION 5 & 6: TRENDING REELS & TRENDING SHORTS */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Trending Reels */}
        <div className="space-y-4 rounded-2xl border border-white/10 bg-[#121212] p-5">
          <div className="flex items-center gap-2 border-b border-white/10 pb-3">
            <Instagram className="h-5 w-5 text-pink-400" />
            <h3 className="font-display text-base text-foreground tracking-wider">
              5. TRENDING <span className="text-pink-400">INSTAGRAM REELS</span>
            </h3>
          </div>
          <div className="space-y-3">
            {trending_reels.slice(0, 4).map((reel) => (
              <div
                key={reel.id}
                className="flex gap-3 rounded-xl border border-white/5 bg-black/40 p-3"
              >
                <img
                  src={reel.thumbnail_url}
                  alt={reel.title}
                  className="h-16 w-16 rounded-lg object-cover shrink-0"
                />
                <div className="flex-1 space-y-1 min-w-0">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-[var(--gold-bright)]">{reel.category}</span>
                    <span className="text-muted-foreground">{reel.engagement}</span>
                  </div>
                  <h4 className="font-semibold text-xs text-foreground truncate">{reel.title}</h4>
                  <p className="text-[10px] text-muted-foreground">Audio: {reel.audio_track}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Trending Shorts */}
        <div className="space-y-4 rounded-2xl border border-white/10 bg-[#121212] p-5">
          <div className="flex items-center gap-2 border-b border-white/10 pb-3">
            <Youtube className="h-5 w-5 text-red-500" />
            <h3 className="font-display text-base text-foreground tracking-wider">
              6. TRENDING <span className="text-red-500">YOUTUBE SHORTS</span>
            </h3>
          </div>
          <div className="space-y-3">
            {trending_shorts.slice(0, 4).map((short) => (
              <div
                key={short.id}
                className="flex gap-3 rounded-xl border border-white/5 bg-black/40 p-3"
              >
                <img
                  src={short.thumbnail_url}
                  alt={short.title}
                  className="h-16 w-16 rounded-lg object-cover shrink-0"
                />
                <div className="flex-1 space-y-1 min-w-0">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-[var(--gold-bright)]">{short.category}</span>
                    <span className="text-muted-foreground">{short.engagement}</span>
                  </div>
                  <h4 className="font-semibold text-xs text-foreground truncate">{short.title}</h4>
                  <p className="text-[10px] text-muted-foreground">
                    Target: {short.target_audience}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SECTION 7: AI RECOMMENDATIONS */}
      <section className="space-y-4 rounded-2xl border border-[oklch(0.85_0.155_86/0.3)] bg-gradient-to-b from-[#181307] via-[#0e0a03] to-black p-6">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-[var(--gold-bright)]" />
          <h2 className="font-display text-xl text-foreground tracking-widest">
            7. AI CREATOR <span className="gold-text">RECOMMENDATIONS</span>
          </h2>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 text-xs">
          {/* Content Ideas */}
          <div className="space-y-2">
            <h4 className="font-bold text-[var(--gold-bright)] flex items-center gap-1.5">
              <Lightbulb className="h-4 w-4" /> Top Content Ideas
            </h4>
            <ul className="space-y-2">
              {ai_recommendations.content_ideas.map((idea, i) => (
                <li
                  key={i}
                  className="rounded-xl border border-white/10 bg-black/40 p-3 text-muted-foreground"
                >
                  {idea}
                </li>
              ))}
            </ul>
          </div>

          {/* Titles & Hooks */}
          <div className="space-y-2">
            <h4 className="font-bold text-[var(--gold-bright)] flex items-center gap-1.5">
              <Sparkles className="h-4 w-4" /> High-Retention Hooks
            </h4>
            <ul className="space-y-2">
              {ai_recommendations.hooks.map((hook, i) => (
                <li
                  key={i}
                  className="rounded-xl border border-white/10 bg-black/40 p-3 text-foreground italic"
                >
                  "{hook}"
                </li>
              ))}
            </ul>
          </div>

          {/* Strategy Details */}
          <div className="space-y-3">
            <h4 className="font-bold text-[var(--gold-bright)] flex items-center gap-1.5">
              <Clock className="h-4 w-4" /> Strategic Parameters
            </h4>
            <div className="rounded-xl border border-white/10 bg-black/40 p-4 space-y-2">
              <div>
                <span className="text-[10px] text-muted-foreground block">Best Platform:</span>
                <span className="font-bold text-foreground">
                  {ai_recommendations.best_platform}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground block">Best Upload Window:</span>
                <span className="font-bold text-[var(--gold-bright)]">
                  {ai_recommendations.best_upload_time}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-muted-foreground block">Target Intensity:</span>
                <span className="font-bold text-foreground">
                  {ai_recommendations.suggested_duration}
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Generated Content Modal */}
      <GenerateSimilarContentModal
        content={generatedContent}
        isOpen={isGeneratingModalOpen}
        onClose={() => setIsGeneratingModalOpen(false)}
      />
    </div>
  );
}
