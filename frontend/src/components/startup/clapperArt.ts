/**
 * Shared visual tokens for the movie-slate clapperboard, so the animated
 * hero clapper (`AnimatedClapper`) and the static brand logo (`MovieSlateLogo`)
 * are pixel-identical at rest — which is what makes the startup animation dock
 * seamlessly into the real logo. Change the look in one place, both update.
 */

/** Diagonal gold/black stripes for the clapper's top slate. */
export const STRIPES =
  "repeating-linear-gradient(-45deg, oklch(0.9 0.12 88) 0 7px, oklch(0.16 0.02 70) 7px 14px)";

/** Outer rounded-square frame (matte black, gold hairline) — minus size/shadow,
 * which the caller supplies so the same frame works at any scale. */
export const CLAPPER_FRAME =
  "relative grid place-items-center rounded-[22%] border border-[oklch(0.85_0.155_86/0.35)] bg-gradient-to-br from-[#1a1408] to-black";

/** The inner clapper occupies 60% of the frame. */
export const CLAPPER_INNER = "relative";
export const CLAPPER_INNER_STYLE = { width: "60%", height: "60%" } as const;

/** The hinged top slate (shared class); callers add motion/rotation. */
export const SLATE_CLASS = "absolute left-0 top-0 h-[30%] w-full origin-left rounded-[14%]";
export const SLATE_STYLE = {
  background: STRIPES,
  transformOrigin: "8% 90%",
  boxShadow: "0 1px 0 0 oklch(0.85 0.155 86 / 0.35)",
} as const;
