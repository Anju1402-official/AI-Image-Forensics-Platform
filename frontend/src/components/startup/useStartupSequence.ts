/**
 * Drives the timeline of the luxury startup animation.
 *
 * The sequence is a fixed set of "scenes"; this hook advances a `scene`
 * counter over time and fires `onComplete` at the end. It is deliberately
 * tiny (a handful of timeouts, not a per-frame loop) — Framer Motion runs
 * the actual visual work on the compositor, so React only re-renders ~5
 * times across the whole ~6s sequence.
 *
 * Everything here is SSR-safe and leak-free: all timers are cleared on
 * unmount or replay, and it respects `prefers-reduced-motion`.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/** Named scenes, in order. Scene 0 = idle (not started). */
export const STARTUP_SCENES = {
  IDLE: 0,
  AMBIENCE: 1, // black + dust
  ICON_IN: 2, // logo fades in under a warm spotlight
  SLATE_OPEN: 3, // top slate opens, warm light escapes
  SLATE_CLAP: 4, // slate closes, clap sound, gold reflection
  REVEAL: 5, // slate opens onto the Sign In page, icon docks
  DONE: 6,
} as const;

export type StartupScene = (typeof STARTUP_SCENES)[keyof typeof STARTUP_SCENES];

/** Start time (ms from sequence start) of each scene. */
const SCENE_TIMELINE: ReadonlyArray<{ scene: StartupScene; at: number }> = [
  { scene: STARTUP_SCENES.AMBIENCE, at: 0 },
  { scene: STARTUP_SCENES.ICON_IN, at: 1500 },
  { scene: STARTUP_SCENES.SLATE_OPEN, at: 3800 },
  { scene: STARTUP_SCENES.SLATE_CLAP, at: 6000 },
  { scene: STARTUP_SCENES.REVEAL, at: 7500 },
  { scene: STARTUP_SCENES.DONE, at: 10000 },
];

/** Total run time of the sequence, in ms. */
export const STARTUP_DURATION_MS = SCENE_TIMELINE[SCENE_TIMELINE.length - 1].at;

/** Premium easing curves shared across the animation. */
export const EASE_PREMIUM = [0.16, 1, 0.3, 1] as const; // easeOutExpo-like
export const SPRING_HINGE = { type: "spring", stiffness: 90, damping: 16, mass: 0.9 } as const;

/* ------------------------------------------------------------------ */
/* Session gating (used by the real integration, not the preview)      */
/* ------------------------------------------------------------------ */

const SESSION_KEY = "cineos_startup_played";

export function hasPlayedStartup(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.sessionStorage.getItem(SESSION_KEY) === "1";
  } catch {
    return false;
  }
}

export function markStartupPlayed(): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(SESSION_KEY, "1");
  } catch {
    /* storage disabled — the animation simply plays again next load */
  }
}

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/* ------------------------------------------------------------------ */
/* Hook                                                                */
/* ------------------------------------------------------------------ */

interface UseStartupSequenceOptions {
  /** Called once when the sequence reaches DONE (or is skipped). */
  onComplete?: () => void;
  /** Start automatically on mount. Default true. */
  autoPlay?: boolean;
}

interface UseStartupSequenceResult {
  scene: StartupScene;
  isPlaying: boolean;
  /** Restart from the very beginning. */
  replay: () => void;
  /** Jump straight to the end and fire onComplete. */
  skip: () => void;
}

export function useStartupSequence({
  onComplete,
  autoPlay = true,
}: UseStartupSequenceOptions = {}): UseStartupSequenceResult {
  const [scene, setScene] = useState<StartupScene>(STARTUP_SCENES.IDLE);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Keep the latest onComplete without re-arming the timeline on every render.
  const onCompleteRef = useRef(onComplete);
  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  const clearTimers = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  const finish = useCallback(() => {
    clearTimers();
    setScene(STARTUP_SCENES.DONE);
    onCompleteRef.current?.();
  }, [clearTimers]);

  const play = useCallback(() => {
    clearTimers();

    // Reduced motion: skip the show, land on the Sign In page immediately.
    if (prefersReducedMotion()) {
      finish();
      return;
    }

    setScene(STARTUP_SCENES.AMBIENCE);
    for (const step of SCENE_TIMELINE) {
      if (step.at === 0) continue;
      const id = setTimeout(() => {
        if (step.scene === STARTUP_SCENES.DONE) {
          finish();
        } else {
          setScene(step.scene);
        }
      }, step.at);
      timers.current.push(id);
    }
  }, [clearTimers, finish]);

  const replay = useCallback(() => play(), [play]);
  const skip = useCallback(() => finish(), [finish]);

  useEffect(() => {
    if (autoPlay) play();
    return clearTimers;
    // play/clearTimers are stable; run once on mount when autoPlay.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isPlaying = scene !== STARTUP_SCENES.IDLE && scene !== STARTUP_SCENES.DONE;

  return useMemo(() => ({ scene, isPlaying, replay, skip }), [scene, isPlaying, replay, skip]);
}
