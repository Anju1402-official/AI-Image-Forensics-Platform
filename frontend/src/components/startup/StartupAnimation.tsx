/**
 * StartupAnimation — the luxury cinematic launch sequence.
 *
 * Five scenes over ~5.9s (matte black → dust → logo → clapper opens with warm
 * light → clap → the slate opens onto the Sign In page as the logo docks into
 * place). Every moving part uses GPU-friendly transform/opacity/filter/clip-
 * path only, so it holds 60fps. Honors `prefers-reduced-motion` (skips to the
 * end) and never throws on missing audio.
 *
 * This component is presentational + self-contained. It renders a full-screen
 * fixed overlay; `reveal` is whatever should appear "through the opening"
 * (a Sign In card). It does not touch routing, auth, or any existing code.
 */
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import { DustField } from "./DustField";
import { AnimatedClapper } from "./AnimatedClapper";
import { playSlateClap } from "./clapperSound";
import {
  EASE_PREMIUM,
  STARTUP_SCENES,
  useStartupSequence,
  type StartupScene,
} from "./useStartupSequence";

interface StartupAnimationProps {
  /** Fired once the sequence completes (or is skipped / reduced-motion). */
  onComplete?: () => void;
  /** Content shown through the opening in the final scene (e.g. Sign In). */
  reveal?: ReactNode;
  /**
   * Integration mode: instead of rendering a `reveal` card, dissolve the
   * overlay to reveal the real page already mounted beneath it, and dock the
   * logo onto the element with id `app-logo-slot` (the real brand logo).
   */
  revealBeneath?: boolean;
  /** Silence the slate clap. Default false. */
  muted?: boolean;
  /** Start automatically on mount. Default true. */
  autoPlay?: boolean;
  /** Allow the user to skip by clicking. Default true. */
  clickToSkip?: boolean;
}

/** The real brand logo tags itself with this id so the animation docks onto it. */
export const APP_LOGO_SLOT_ID = "app-logo-slot";

interface Box {
  x: number;
  y: number;
  scale: number;
}

const CLAPPER_MAX = 240;
const DOCK_SIZE = 56; // matches the login icon frame (h-14 w-14)

export function StartupAnimation({
  onComplete,
  reveal,
  revealBeneath = false,
  muted = false,
  autoPlay = true,
  clickToSkip = true,
}: StartupAnimationProps) {
  // Integration dissolve mode is on when asked for and there's no reveal card.
  const underlay = revealBeneath && !reveal;
  const [mounted, setMounted] = useState(false);
  const [size, setSize] = useState(200);
  const [center, setCenter] = useState<Box>({ x: 0, y: 0, scale: 1 });
  const [dock, setDock] = useState<Box | null>(null);
  const iconSlotRef = useRef<HTMLDivElement | null>(null);

  const { scene, skip } = useStartupSequence({ onComplete, autoPlay });

  // Client-only mount gate (avoids SSR/hydration mismatch for dust & measures).
  useEffect(() => setMounted(true), []);

  // Measure the clapper size and viewport centre; keep in sync on resize.
  useEffect(() => {
    if (!mounted) return;
    const measure = () => {
      const s = Math.max(
        120,
        Math.min(window.innerWidth * 0.62, window.innerHeight * 0.34, CLAPPER_MAX),
      );
      setSize(s);
      setCenter({ x: window.innerWidth / 2 - s / 2, y: window.innerHeight / 2 - s / 2, scale: 1 });
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [mounted]);

  // When the reveal scene starts, measure the Sign In icon slot and compute
  // the dock target so the clapper flies precisely into place.
  useEffect(() => {
    if (scene !== STARTUP_SCENES.REVEAL) return;
    const id = requestAnimationFrame(() => {
      // Preview docks onto the internal reveal-card slot; integration docks
      // onto the real brand logo already on the page beneath the overlay.
      const el = reveal
        ? iconSlotRef.current
        : typeof document !== "undefined"
          ? document.getElementById(APP_LOGO_SLOT_ID)
          : null;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setDock({
        x: r.left + r.width / 2 - size / 2,
        y: r.top + r.height / 2 - size / 2,
        scale: (r.width || DOCK_SIZE) / size,
      });
    });
    return () => cancelAnimationFrame(id);
  }, [scene, size, reveal]);

  // Slate clap sound on the closing beat.
  useEffect(() => {
    if (scene === STARTUP_SCENES.SLATE_CLAP && !muted) playSlateClap(0.5);
  }, [scene, muted]);

  const clapperTarget = useCallback((): Record<string, number | string> => {
    if (scene < STARTUP_SCENES.ICON_IN) {
      return { opacity: 0, x: center.x, y: center.y, scale: 0.92, filter: "blur(10px)" };
    }
    if (scene >= STARTUP_SCENES.REVEAL && dock) {
      return { opacity: 1, x: dock.x, y: dock.y, scale: dock.scale, filter: "blur(0px)" };
    }
    return { opacity: 1, x: center.x, y: center.y, scale: 1, filter: "blur(0px)" };
  }, [scene, center, dock]);

  // Slate is open only while the reveal is in motion; it settles closed so the
  // docked logo matches the resting Sign In icon exactly.
  const slateOpen = scene === STARTUP_SCENES.SLATE_OPEN || scene === STARTUP_SCENES.REVEAL;
  const glow = scene === STARTUP_SCENES.SLATE_OPEN || scene === STARTUP_SCENES.REVEAL;
  const reflection = scene === STARTUP_SCENES.SLATE_CLAP;
  const revealActive = scene >= STARTUP_SCENES.REVEAL;

  return (
    <div
      className="fixed inset-0 z-[100] overflow-hidden"
      onClick={clickToSkip ? () => skip() : undefined}
      role="presentation"
    >
      {/* Matte-black backdrop. In integration mode it dissolves (with a gentle
          camera push) during the reveal so the real page beneath shows through;
          otherwise it stays opaque behind the reveal card. */}
      <motion.div
        aria-hidden
        className="absolute inset-0 bg-[#0b0b0b]"
        style={{ willChange: "opacity, transform" }}
        animate={{
          opacity: underlay && revealActive ? 0 : 1,
          scale: underlay && revealActive ? 1.08 : 1,
        }}
        transition={{ duration: 1.1, ease: EASE_PREMIUM, delay: revealActive ? 0.15 : 0 }}
      />

      {/* Vignette that softly lifts as the scene opens up */}
      <motion.div
        aria-hidden
        className="absolute inset-0"
        style={{ background: "var(--gradient-vignette)" }}
        animate={{ opacity: revealActive ? (underlay ? 0 : 0.4) : 1 }}
        transition={{ duration: 1.2, ease: "easeOut" }}
      />

      {/* Warm spotlight behind the logo */}
      <motion.div
        aria-hidden
        className="absolute left-1/2 top-1/2 h-[70vmin] w-[70vmin] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(circle, oklch(0.85 0.155 86 / 0.16) 0%, oklch(0.85 0.155 86 / 0) 65%)",
          willChange: "opacity, transform",
        }}
        initial={{ opacity: 0, scale: 0.7 }}
        animate={{
          opacity: scene >= STARTUP_SCENES.ICON_IN && !revealActive ? 1 : revealActive ? 0 : 0,
          scale: scene >= STARTUP_SCENES.ICON_IN ? 1 : 0.7,
        }}
        transition={{ duration: 1.4, ease: "easeOut" }}
      />

      {mounted && (
        <motion.div
          className="absolute inset-0"
          animate={{ opacity: revealActive ? 0 : 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <DustField count={22} />
        </motion.div>
      )}

      {/* The Sign In reveal, unveiled "through the opening" via an expanding
          clip-path circle centred on the docked logo. */}
      {reveal && (
        <motion.div
          className="absolute inset-0 flex items-start justify-center overflow-y-auto px-4 py-[8vh] sm:items-center sm:py-4"
          initial={false}
          animate={{
            opacity: revealActive ? 1 : 0,
            clipPath: revealActive ? "circle(150% at 50% 42%)" : "circle(0% at 50% 42%)",
          }}
          transition={{
            opacity: { duration: 0.9, ease: "easeOut", delay: revealActive ? 0.15 : 0 },
            clipPath: { duration: 1.3, ease: EASE_PREMIUM },
          }}
          style={{ willChange: "clip-path, opacity" }}
        >
          {/* The icon slot inside `reveal` is what the clapper docks into; the
              consumer marks it by rendering an element with this ref. We expose
              it through context-free prop wiring: the preview passes a card whose
              icon position we approximate by centring this ref over the card. */}
          <div className="relative w-full max-w-md">
            <div
              ref={iconSlotRef}
              className="pointer-events-none absolute left-1/2 top-8 h-14 w-14 -translate-x-1/2"
              aria-hidden
            />
            {reveal}
          </div>
        </motion.div>
      )}

      {/* The animated clapper / logo */}
      <motion.div
        className="fixed left-0 top-0"
        style={{ width: size, height: size, willChange: "transform, opacity, filter" }}
        initial={{ opacity: 0, x: center.x, y: center.y, scale: 0.92, filter: "blur(10px)" }}
        animate={clapperTarget()}
        transition={{
          duration: scene === STARTUP_SCENES.REVEAL ? 1.2 : 0.9,
          ease: EASE_PREMIUM,
        }}
      >
        <AnimatedClapper
          open={slateOpen}
          glow={glow}
          reflection={reflection}
          className="h-full w-full"
        />
      </motion.div>

      {clickToSkip && scene < STARTUP_SCENES.REVEAL && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            skip();
          }}
          className="absolute bottom-6 right-6 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-[11px] uppercase tracking-widest text-white/60 backdrop-blur transition hover:text-white"
        >
          Skip
        </button>
      )}
    </div>
  );
}
