import React, { useState } from "react";
import {
  X,
  Sparkles,
  Copy,
  Check,
  Film,
  Camera,
  Music,
  FileText,
  Image as ImageIcon,
  Video,
  Tag,
  Clock,
} from "lucide-react";
import { toast } from "sonner";
import { GeneratedContentResponse } from "@/lib/api";

interface GenerateSimilarContentModalProps {
  content: GeneratedContentResponse | null;
  isOpen: boolean;
  onClose: () => void;
}

export function GenerateSimilarContentModal({
  content,
  isOpen,
  onClose,
}: GenerateSimilarContentModalProps) {
  const [copied, setCopied] = useState(false);

  if (!isOpen || !content) return null;

  const handleCopyScript = () => {
    navigator.clipboard.writeText(content.original_script);
    setCopied(true);
    toast.success("Original script copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 lg:p-8">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/80 backdrop-blur-md" onClick={onClose} />

      {/* Modal Card */}
      <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-2xl border border-[oklch(0.85_0.155_86/0.3)] bg-[#0d0d0d] p-6 sm:p-8 shadow-[var(--shadow-premium)] text-foreground z-10 scrollbar-hide">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute right-5 top-5 grid h-9 w-9 place-items-center rounded-full border border-white/10 bg-white/5 text-muted-foreground hover:text-foreground transition"
        >
          <X className="h-4 w-4" />
        </button>

        {/* Modal Header */}
        <div className="space-y-2 border-b border-white/10 pb-5 pr-8">
          <div className="flex items-center gap-2">
            <span className="rounded-md border border-[oklch(0.85_0.155_86/0.4)] bg-[oklch(0.85_0.155_86/0.12)] px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-widest text-[var(--gold-bright)]">
              AI Content Engine
            </span>
            <span className="text-xs text-muted-foreground">
              Original Non-Plagiarized Blueprint
            </span>
          </div>
          <h2 className="font-display text-xl sm:text-2xl text-foreground tracking-wide">
            Inspired by: <span className="gold-text">“{content.trend_title}”</span>
          </h2>
        </div>

        {/* Modal Content Sections */}
        <div className="mt-6 space-y-6 text-xs">
          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Zap className="h-3 w-3 text-[var(--gold-dim)]" /> Intensity
              </div>
              <div className="mt-1 font-semibold text-foreground">{content.suggested_duration}</div>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Video className="h-3 w-3 text-[var(--gold-dim)]" /> Voice-over Style
              </div>
              <div className="mt-1 font-semibold text-foreground truncate">
                {content.voice_over_style}
              </div>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.03] p-3 col-span-2">
              <div className="text-[10px] text-muted-foreground flex items-center gap-1">
                <Tag className="h-3 w-3 text-[var(--gold-dim)]" /> SEO Title
              </div>
              <div className="mt-1 font-semibold text-[var(--gold-bright)] truncate">
                {content.seo_title}
              </div>
            </div>
          </div>

          {/* Hook & CTA Cards */}
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.3)] bg-gradient-to-br from-[#1a1408] to-[#0d0a04] p-4">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[var(--gold-bright)] mb-1.5 flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5" /> High-Retention Hook
              </div>
              <p className="text-xs italic leading-relaxed text-foreground">
                "{content.better_hook}"
              </p>
            </div>

            <div className="rounded-xl border border-[oklch(0.85_0.155_86/0.3)] bg-gradient-to-br from-[#1a1408] to-[#0d0a04] p-4">
              <div className="text-[11px] font-bold uppercase tracking-wider text-[var(--gold-bright)] mb-1.5 flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5" /> High-Converting Call to Action (CTA)
              </div>
              <p className="text-xs italic leading-relaxed text-foreground">
                "{content.better_cta}"
              </p>
            </div>
          </div>

          {/* Original Script Section */}
          <div className="rounded-xl border border-white/10 bg-[#121212] p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 font-display text-sm text-foreground">
                <FileText className="h-4 w-4 text-[var(--gold-bright)]" />
                <span>Original Production Script</span>
              </div>
              <button
                onClick={handleCopyScript}
                className="flex items-center gap-1.5 rounded-lg border border-[oklch(0.85_0.155_86/0.3)] bg-white/5 px-3 py-1.5 text-xs text-[var(--gold-bright)] hover:bg-[oklch(0.85_0.155_86/0.15)] transition"
              >
                {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                <span>{copied ? "Copied!" : "Copy Script"}</span>
              </button>
            </div>
            <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-muted-foreground bg-black/40 p-4 rounded-lg border border-white/5 max-h-48 overflow-y-auto">
              {content.original_script}
            </pre>
          </div>

          {/* Storyboard Panels & Camera Angles */}
          <div className="grid sm:grid-cols-2 gap-4">
            {/* Storyboard breakdown */}
            <div className="rounded-xl border border-white/10 bg-[#121212] p-4 space-y-3">
              <div className="flex items-center gap-2 font-display text-sm text-foreground">
                <Film className="h-4 w-4 text-[var(--gold-bright)]" />
                <span>Storyboard & Scene Frames</span>
              </div>
              <div className="space-y-2">
                {content.storyboard.map((sb) => (
                  <div
                    key={sb.frame}
                    className="rounded-lg border border-white/5 bg-black/40 p-2.5 space-y-1"
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-bold text-[var(--gold-bright)]">Frame #{sb.frame}</span>
                      <span className="text-muted-foreground font-mono">{sb.shot}</span>
                    </div>
                    <p className="text-[11px] text-foreground">{sb.visual}</p>
                    <p className="text-[10px] text-muted-foreground italic">Audio: {sb.audio}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Camera Angles & BGM */}
            <div className="space-y-4">
              <div className="rounded-xl border border-white/10 bg-[#121212] p-4 space-y-2.5">
                <div className="flex items-center gap-2 font-display text-sm text-foreground">
                  <Camera className="h-4 w-4 text-[var(--gold-bright)]" />
                  <span>Suggested Camera Angles</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {content.camera_angles.map((cam) => (
                    <span
                      key={cam}
                      className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-foreground"
                    >
                      {cam}
                    </span>
                  ))}
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#121212] p-4 space-y-2.5">
                <div className="flex items-center gap-2 font-display text-sm text-foreground">
                  <Music className="h-4 w-4 text-[var(--gold-bright)]" />
                  <span>Recommended BGM & Audio Tracks</span>
                </div>
                <div className="space-y-1.5">
                  {content.bgm_suggestions.map((bgm) => (
                    <div
                      key={bgm}
                      className="flex items-center gap-2 text-[11px] text-muted-foreground bg-black/40 px-3 py-1.5 rounded-lg border border-white/5"
                    >
                      <span className="text-[var(--gold-bright)]">🎵</span>
                      <span>{bgm}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Thumbnail Idea & Keywords */}
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="rounded-xl border border-white/10 bg-[#121212] p-4 space-y-2">
              <div className="flex items-center gap-2 font-display text-sm text-foreground">
                <ImageIcon className="h-4 w-4 text-[var(--gold-bright)]" />
                <span>Thumbnail Concept</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {content.thumbnail_idea}
              </p>
            </div>

            <div className="rounded-xl border border-white/10 bg-[#121212] p-4 space-y-2">
              <div className="flex items-center gap-2 font-display text-sm text-foreground">
                <Tag className="h-4 w-4 text-[var(--gold-bright)]" />
                <span>Viral SEO Keywords</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {content.viral_keywords.map((kw) => (
                  <span
                    key={kw}
                    className="rounded-md border border-[oklch(0.85_0.155_86/0.3)] bg-[oklch(0.85_0.155_86/0.08)] px-2 py-0.5 text-[10px] text-[var(--gold-bright)]"
                  >
                    #{kw}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
