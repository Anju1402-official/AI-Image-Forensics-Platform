/**
 * StartupPreview — a standalone review harness for the launch animation.
 *
 * This is the ONLY thing wired to /startup-preview. It renders the animation
 * in isolation (over a decoupled visual copy of the Sign In card) with Replay
 * / Mute controls so the sequence can be approved before it's integrated in
 * front of the real Sign In page. It imports nothing from the auth/login code
 * and has zero effect on the rest of the app.
 */
import { useState } from "react";
import { RotateCcw, Volume2, VolumeX, Eye } from "lucide-react";
import { StartupAnimation } from "./StartupAnimation";
import { SignInPreviewCard } from "./SignInPreviewCard";

export function StartupPreview() {
  const [runId, setRunId] = useState(0); // bump to remount = replay
  const [muted, setMuted] = useState(false);
  const [done, setDone] = useState(false);

  const replay = () => {
    setDone(false);
    setRunId((n) => n + 1);
  };

  return (
    <main className="relative min-h-screen w-full overflow-hidden bg-[#0b0b0b]">
      <StartupAnimation
        key={runId}
        muted={muted}
        onComplete={() => setDone(true)}
        reveal={<SignInPreviewCard hideIcon />}
      />

      {/* Review controls — outside the animation, fixed on top */}
      <div className="fixed left-1/2 top-5 z-[200] flex -translate-x-1/2 items-center gap-2 rounded-full border border-white/10 bg-black/60 px-2 py-1.5 backdrop-blur-xl">
        <span className="px-2 text-[10px] uppercase tracking-[0.2em] text-[var(--gold-dim)]">
          <Eye className="mr-1 inline h-3 w-3" />
          Preview
        </span>
        <button
          type="button"
          onClick={replay}
          className="flex items-center gap-1.5 rounded-full bg-white/5 px-3 py-1.5 text-[11px] font-medium uppercase tracking-widest text-white/80 transition hover:bg-white/10 hover:text-white"
        >
          <RotateCcw className="h-3.5 w-3.5" /> Replay
        </button>
        <button
          type="button"
          onClick={() => setMuted((m) => !m)}
          aria-label={muted ? "Unmute" : "Mute"}
          className="flex items-center gap-1.5 rounded-full bg-white/5 px-3 py-1.5 text-[11px] font-medium uppercase tracking-widest text-white/80 transition hover:bg-white/10 hover:text-white"
        >
          {muted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
          {muted ? "Muted" : "Sound"}
        </button>
      </div>

      {done && (
        <div className="pointer-events-none fixed bottom-5 left-1/2 z-[200] -translate-x-1/2 rounded-full border border-[oklch(0.85_0.155_86/0.25)] bg-black/60 px-4 py-1.5 text-[11px] uppercase tracking-widest text-[var(--gold-bright)] backdrop-blur">
          Sequence complete
        </div>
      )}
    </main>
  );
}
