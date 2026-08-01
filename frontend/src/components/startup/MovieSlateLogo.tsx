/**
 * MovieSlateLogo — the app's brand mark: a champagne-gold movie-slate
 * clapperboard on a matte-black rounded square. Static (no motion), reusable
 * at any size, and visually identical to the resting state of the startup
 * animation's clapper — so the launch sequence docks straight into it.
 *
 * Use it anywhere the brand logo appears (Sign In, Sign Up, the sidebar).
 * Size it via `className` (e.g. `h-14 w-14`).
 */
import {
  CLAPPER_FRAME,
  CLAPPER_INNER,
  CLAPPER_INNER_STYLE,
  SLATE_CLASS,
  SLATE_STYLE,
} from "./clapperArt";
import { ClapperBoardBody } from "./ClapperBoardBody";

interface MovieSlateLogoProps {
  /** Sizing/extra classes for the outer frame (e.g. "h-14 w-14"). */
  className?: string;
  /** Soft gold glow behind the mark. Default on. */
  glow?: boolean;
  /** Snap the slate open on hover of an ancestor `.group` (used in the sidebar). */
  hoverSnap?: boolean;
  /** Optional DOM id — the startup animation docks onto this element. */
  id?: string;
}

export function MovieSlateLogo({
  className = "h-14 w-14",
  glow = true,
  hoverSnap = false,
  id,
}: MovieSlateLogoProps) {
  return (
    <div
      id={id}
      className={`${CLAPPER_FRAME} ${className}`}
      style={{
        boxShadow: glow ? "0 0 24px -6px oklch(0.85 0.155 86 / 0.6)" : undefined,
      }}
      aria-label="ORIGO"
      role="img"
    >
      <div className={CLAPPER_INNER} style={CLAPPER_INNER_STYLE}>
        <ClapperBoardBody />
        <div
          className={`${SLATE_CLASS} ${hoverSnap ? "transition-transform group-hover:animate-clapper-snap" : ""}`}
          style={SLATE_STYLE}
        />
      </div>
    </div>
  );
}
