import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, Instagram, Youtube, Flame, History, Hourglass, Cpu } from "lucide-react";

interface TrendAnalysisProgressProps {
  locationName: string;
  onComplete: () => void;
}

const STEPS = [
  { label: "Analyzing Instagram Reels...", icon: Instagram, color: "text-pink-400" },
  { label: "Analyzing YouTube Shorts...", icon: Youtube, color: "text-red-500" },
  { label: "Detecting Viral Trends...", icon: Flame, color: "text-amber-400" },
  { label: "Processing Historical Data...", icon: History, color: "text-blue-400" },
  { label: "Estimating Trend Lifespan...", icon: Hourglass, color: "text-emerald-400" },
  { label: "Generating AI Insights...", icon: Cpu, color: "text-[var(--gold-bright)]" },
];

export function TrendAnalysisProgress({ locationName, onComplete }: TrendAnalysisProgressProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStepIndex((prev) => {
        if (prev < STEPS.length - 1) {
          return prev + 1;
        } else {
          clearInterval(interval);
          setTimeout(() => {
            onComplete();
          }, 600);
          return prev;
        }
      });
    }, 800);

    return () => clearInterval(interval);
  }, [onComplete]);

  const currentStep = STEPS[currentStepIndex];
  const Icon = currentStep.icon;
  const progressPct = Math.round(((currentStepIndex + 1) / STEPS.length) * 100);

  return (
    <div className="flex h-full min-h-[500px] w-full flex-col items-center justify-center rounded-2xl border border-[oklch(0.85_0.155_86/0.2)] bg-gradient-to-b from-[#0e0e0e] via-[#080808] to-black p-8 text-center shadow-[var(--shadow-premium)] relative overflow-hidden">
      {/* Background Animated Glow Orb */}
      <div className="pointer-events-none absolute h-96 w-96 rounded-full bg-[oklch(0.85_0.155_86/0.12)] blur-3xl animate-pulse" />

      {/* Main Spinning Radar / Ring */}
      <div className="relative mb-8 grid h-32 w-32 place-items-center">
        <div className="absolute inset-0 rounded-full border-2 border-dashed border-[oklch(0.85_0.155_86/0.4)] animate-spin-slow" />
        <div className="absolute inset-2 rounded-full border border-[oklch(0.85_0.155_86/0.2)]" />
        <div className="grid h-20 w-20 place-items-center rounded-full border border-[oklch(0.85_0.155_86/0.5)] bg-gradient-to-br from-[#1a1408] to-black shadow-[0_0_30px_var(--gold-bright)]">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentStepIndex}
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.5, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Icon className={`h-8 w-8 ${currentStep.color}`} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Location Badge */}
      <div className="mb-3 rounded-full border border-[oklch(0.85_0.155_86/0.3)] bg-[oklch(0.85_0.155_86/0.08)] px-4 py-1 text-xs font-semibold text-[var(--gold-bright)]">
        Target Location: {locationName}
      </div>

      {/* Step Label */}
      <h3 className="font-display text-xl tracking-wider text-foreground mb-2">
        {currentStep.label}
      </h3>

      <p className="text-xs text-muted-foreground max-w-sm mb-6">
        Scanning live algorithms, viral engagements, audio footprints, and historical retention
        curves...
      </p>

      {/* Progress Bar */}
      <div className="w-full max-w-md space-y-2">
        <div className="h-2 w-full overflow-hidden rounded-full bg-white/10 p-0.5">
          <motion.div
            className="h-full rounded-full bg-gradient-to-r from-[var(--gold)] to-[var(--gold-bright)] shadow-[0_0_15px_var(--gold-bright)]"
            initial={{ width: "0%" }}
            animate={{ width: `${progressPct}%` }}
            transition={{ duration: 0.4 }}
          />
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted-foreground font-mono">
          <span>
            STEP {currentStepIndex + 1} OF {STEPS.length}
          </span>
          <span className="text-[var(--gold-bright)] font-bold">{progressPct}%</span>
        </div>
      </div>
    </div>
  );
}
