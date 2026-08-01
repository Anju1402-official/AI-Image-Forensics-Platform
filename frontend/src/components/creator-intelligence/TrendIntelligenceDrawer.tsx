import React, { useState, useEffect } from "react";
import {
  X,
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Clock,
  Target,
  Calendar,
  BarChart2,
  CheckCircle2,
  AlertCircle,
  Eye,
  Globe,
} from "lucide-react";
import { CurrentTrendItem } from "@/lib/api";

interface TrendIntelligenceDrawerProps {
  trend: CurrentTrendItem | null;
  isOpen: boolean;
  onClose: () => void;
  locationName: string;
}

export function TrendIntelligenceDrawer({
  trend,
  isOpen,
  onClose,
  locationName,
}: TrendIntelligenceDrawerProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedContent, setGeneratedContent] = useState<any>(null);

  // Use a stable random value or just mock data
  const score = trend ? Math.min(98, trend.viral_score + 10) : 85;

  useEffect(() => {
    if (!isOpen) {
      setGeneratedContent(null);
      setIsGenerating(false);
    }
  }, [isOpen]);

  if (!isOpen || !trend) return null;

  const handleGenerate = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setIsGenerating(false);
      setGeneratedContent({
        script:
          "Hook: Did you know you've been doing this wrong?\n\nBody: [Explains the core concept of the trend]. Here's how to fix it in 3 easy steps.\n\nOutro: Don't forget to save this for later!",
        hook: "Stop scrolling if you want to master [Topic] in 30 seconds.",
        caption:
          "Mastering the latest trend from " +
          locationName +
          " 🌟 This took so long to perfect!\n\n👇 Drop a comment if you want a part 2!",
        cta: "Save this video so you don't lose it!",
        hashtags: trend.hashtags,
        thumbnail:
          "A split screen showing 'Before' (confused) and 'After' (mind-blown) with bold yellow text overlay.",
      });
    }, 2500);
  };

  const getRiskColor = (level: string) => {
    if (level === "Safe") return "text-emerald-400 bg-emerald-400/10 border-emerald-400/20";
    if (level === "Use Carefully") return "text-amber-400 bg-amber-400/10 border-amber-400/20";
    return "text-red-400 bg-red-400/10 border-red-400/20";
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed inset-y-0 right-0 z-50 w-full max-w-2xl bg-[#0a0a0a] border-l border-white/10 shadow-2xl flex flex-col transform transition-transform duration-300 ease-in-out translate-x-0 overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b border-white/10 bg-[#0a0a0a]/90 backdrop-blur-md">
          <div>
            <div className="text-[10px] uppercase tracking-widest text-[var(--gold-dim)] flex items-center gap-2">
              <Sparkles className="h-3 w-3" /> ORIGO Trend Intelligence
            </div>
            <h2 className="text-xl font-display text-foreground mt-1 tracking-wide">
              {trend.title}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="grid h-8 w-8 place-items-center rounded-full bg-white/5 text-muted-foreground hover:bg-white/10 hover:text-foreground transition"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-6 space-y-8 flex-1">
          {/* Opportunity Score */}
          <section className="bg-[#111111] border border-white/10 rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Target className="h-24 w-24 text-[var(--gold-bright)]" />
            </div>
            <div className="flex items-center gap-2 mb-4 relative z-10">
              <Target className="h-4 w-4 text-[var(--gold-bright)]" />
              <h3 className="font-semibold text-foreground uppercase tracking-wider text-sm">
                Opportunity Score
              </h3>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 relative z-10">
              <div>
                <div className="text-[10px] text-muted-foreground uppercase">Popularity</div>
                <div className="text-lg font-bold text-foreground">High</div>
              </div>
              <div>
                <div className="text-[10px] text-muted-foreground uppercase">Growth Rate</div>
                <div className="text-lg font-bold text-emerald-400">+{trend.growth_pct}%</div>
              </div>
              <div>
                <div className="text-[10px] text-muted-foreground uppercase">Competition</div>
                <div className="text-lg font-bold text-amber-400">Medium</div>
              </div>
              <div>
                <div className="text-[10px] text-muted-foreground uppercase">Engagement</div>
                <div className="text-lg font-bold text-foreground">{trend.engagement}</div>
              </div>
            </div>

            <div className="mt-5 flex items-center justify-between border-t border-white/10 pt-4 relative z-10">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-full border-2 border-[var(--gold-bright)] flex items-center justify-center font-bold text-[var(--gold-bright)] shadow-[0_0_10px_var(--gold-dim)]">
                  {score}
                </div>
                <div>
                  <div className="text-xs font-semibold text-foreground">Overall Score</div>
                  <div className="text-[10px] text-emerald-400 flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3" /> Recommendation: Create Now
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Trend Lifetime Prediction */}
          <section className="bg-[#111111] border border-white/10 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              <h3 className="font-semibold text-foreground uppercase tracking-wider text-sm">
                Trend Lifetime Prediction
              </h3>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs">
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-[9px] uppercase text-muted-foreground mb-1">Started On</div>
                <div className="font-semibold text-foreground">5 Days Ago</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-[9px] uppercase text-muted-foreground mb-1">Peak Date</div>
                <div className="font-semibold text-foreground">Tomorrow</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-[9px] uppercase text-muted-foreground mb-1">Remaining</div>
                <div className="font-semibold text-foreground">{trend.expected_duration}</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-[9px] uppercase text-muted-foreground mb-1">End Date</div>
                <div className="font-semibold text-foreground">In 14 Days</div>
              </div>
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                <div className="text-[9px] uppercase text-blue-400 mb-1">Stage</div>
                <div className="font-bold text-blue-400 flex items-center gap-1">
                  <TrendingUp className="h-3 w-3" /> Rising
                </div>
              </div>
            </div>
          </section>

          {/* AI Risk Intelligence */}
          <section className="bg-[#111111] border border-white/10 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <AlertTriangle className="h-4 w-4 text-amber-400" />
              <h3 className="font-semibold text-foreground uppercase tracking-wider text-sm">
                AI Risk Intelligence
              </h3>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 text-[10px] mb-4">
              {[
                "Political",
                "Religious",
                "Copyright",
                "Misinformation",
                "Brand Reputation",
                "Ethical",
              ].map((risk) => (
                <div
                  key={risk}
                  className="flex justify-between items-center border border-white/5 bg-white/5 rounded p-2"
                >
                  <span className="text-muted-foreground">{risk}</span>
                  <span className="text-emerald-400 font-bold">Low</span>
                </div>
              ))}
              <div className="flex justify-between items-center border border-white/5 bg-white/5 rounded p-2 col-span-2 sm:col-span-1 lg:col-span-2">
                <span className="text-muted-foreground">Public Sentiment</span>
                <span className="text-emerald-400 font-bold">Positive (88%)</span>
              </div>
            </div>

            <div
              className={`mt-2 p-3 rounded-xl border flex items-start gap-3 ${getRiskColor("Safe")}`}
            >
              <CheckCircle2 className="h-5 w-5 mt-0.5 shrink-0" />
              <div>
                <div className="font-bold text-sm">Risk Level: Safe</div>
                <div className="text-xs mt-1 opacity-90">
                  Summary: No controversial elements detected. Content aligns with standard
                  community guidelines.
                </div>
                <div className="text-xs font-semibold mt-1">
                  AI Recommendation: Proceed with creation.
                </div>
              </div>
            </div>
          </section>

          {/* AI Best Time to Upload */}
          <section className="bg-[#111111] border border-white/10 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <Clock className="h-4 w-4 text-[var(--gold-dim)]" />
              <h3 className="font-semibold text-foreground uppercase tracking-wider text-sm">
                AI Best Time to Upload
              </h3>
            </div>

            <div className="flex flex-col sm:flex-row gap-6">
              <div className="flex-1 space-y-4">
                <div className="flex justify-between text-xs border-b border-white/10 pb-2">
                  <span className="text-muted-foreground">Current Local Time</span>
                  <span className="text-foreground font-medium">
                    {new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                <div className="flex justify-between text-xs border-b border-white/10 pb-2">
                  <span className="text-muted-foreground">Selected Location</span>
                  <span className="text-foreground font-medium">{locationName}</span>
                </div>
                <div className="flex justify-between text-xs border-b border-white/10 pb-2">
                  <span className="text-muted-foreground">Best Day</span>
                  <span className="text-foreground font-medium text-[var(--gold-bright)]">
                    Today
                  </span>
                </div>
              </div>

              <div className="flex-1 bg-gradient-to-br from-white/5 to-transparent rounded-xl border border-white/10 p-4">
                <div className="text-[10px] uppercase text-muted-foreground mb-3 tracking-wider">
                  Recommended Slots
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="text-xs font-semibold text-foreground mb-1">Today</div>
                    <div className="inline-block bg-[var(--gold)]/20 border border-[var(--gold)]/30 text-[var(--gold-bright)] px-2 py-1 rounded text-xs font-bold">
                      7:00 PM – 8:30 PM
                    </div>
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-foreground mb-1">Tomorrow</div>
                    <div className="inline-block bg-white/10 border border-white/20 text-foreground px-2 py-1 rounded text-xs font-bold">
                      11:30 AM – 1:00 PM
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between text-[10px]">
                  <span className="text-muted-foreground">
                    Peak Audience: <strong className="text-foreground">{trend.platform}</strong>
                  </span>
                  <span className="text-emerald-400">Confidence: 94%</span>
                </div>
              </div>
            </div>
          </section>

          {/* AI Content Generator */}
          <section className="bg-gradient-to-b from-[#181818] to-[#111111] border border-[var(--gold-dim)]/30 rounded-2xl p-5 shadow-[0_0_20px_rgba(0,0,0,0.5)]">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="h-5 w-5 text-[var(--gold-bright)] animate-pulse" />
              <h3 className="font-display text-lg text-[var(--gold-bright)] tracking-wider">
                AI CONTENT GENERATOR
              </h3>
            </div>

            {!generatedContent ? (
              <div className="text-center py-6">
                <p className="text-sm text-muted-foreground mb-6">
                  Instantly generate a highly-optimized script, hook, caption, and thumbnail idea
                  for this exact trend.
                </p>
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="mx-auto flex items-center gap-2 rounded-xl bg-gradient-to-r from-[var(--gold)] via-[var(--gold-bright)] to-[var(--gold-dim)] px-6 py-3 text-sm font-bold text-black shadow-[0_0_25px_-4px_var(--gold-bright)] transition hover:scale-105 active:scale-95 disabled:opacity-50"
                >
                  {isGenerating ? (
                    <>
                      <Sparkles className="h-4 w-4 fill-black animate-spin" />
                      <span>Generating Blueprint...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 fill-black" />
                      <span>Generate Content</span>
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                    <div className="text-[10px] uppercase text-[var(--gold-dim)] font-bold mb-2 tracking-wider">
                      Viral Hook
                    </div>
                    <p className="text-sm text-foreground">{generatedContent.hook}</p>
                  </div>
                  <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                    <div className="text-[10px] uppercase text-[var(--gold-dim)] font-bold mb-2 tracking-wider">
                      Call to Action (CTA)
                    </div>
                    <p className="text-sm text-foreground">{generatedContent.cta}</p>
                  </div>
                </div>

                <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                  <div className="text-[10px] uppercase text-[var(--gold-dim)] font-bold mb-2 tracking-wider">
                    Reel Script
                  </div>
                  <div className="text-sm text-foreground whitespace-pre-wrap">
                    {generatedContent.script}
                  </div>
                </div>

                <div className="bg-black/40 border border-white/10 rounded-xl p-4">
                  <div className="text-[10px] uppercase text-[var(--gold-dim)] font-bold mb-2 tracking-wider">
                    Caption & Hashtags
                  </div>
                  <p className="text-sm text-foreground mb-2">{generatedContent.caption}</p>
                  <div className="flex flex-wrap gap-1">
                    {generatedContent.hashtags.map((tag: string) => (
                      <span
                        key={tag}
                        className="text-xs text-[var(--gold-bright)] bg-[var(--gold-dim)]/10 px-1.5 py-0.5 rounded"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="bg-black/40 border border-white/10 rounded-xl p-4 flex gap-4 items-center">
                  <div className="h-16 w-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center shrink-0">
                    <Eye className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <div className="text-[10px] uppercase text-[var(--gold-dim)] font-bold mb-1 tracking-wider">
                      Thumbnail Idea
                    </div>
                    <p className="text-xs text-foreground">{generatedContent.thumbnail}</p>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </>
  );
}
