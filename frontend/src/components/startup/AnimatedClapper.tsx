/**
 * The animatable clapperboard — the hero of the startup sequence. Its resting
 * state (slate closed, no glow) is pixel-identical to `MovieSlateLogo` because
 * both draw from `clapperArt` + `ClapperBoardBody`; that's what lets the
 * animation dock seamlessly into the real brand logo.
 *
 * Purely transform/opacity-driven for 60fps. The parent controls
 * position/scale (for the dock move); this renders the frame, the hinge, the
 * warm escaping light, and the gold reflection sweep.
 */
import { motion } from "framer-motion";
import { SPRING_HINGE } from "./useStartupSequence";
import {
  CLAPPER_FRAME,
  CLAPPER_INNER,
  CLAPPER_INNER_STYLE,
  SLATE_CLASS,
  SLATE_STYLE,
} from "./clapperArt";
import { ClapperBoardBody } from "./ClapperBoardBody";

interface AnimatedClapperProps {
  /** Whether the top slate is open (raised on its hinge). */
  open: boolean;
  /** Whether the warm light spills from the slate gap. */
  glow?: boolean;
  /** Whether a gold reflection is sweeping across the board right now. */
  reflection?: boolean;
  className?: string;
}

export function AnimatedClapper({
  open,
  glow = false,
  reflection = false,
  className = "",
}: AnimatedClapperProps) {
  return (
    <div
      className={`${CLAPPER_FRAME} ${className}`}
      style={{
        boxShadow: "0 0 60px -12px oklch(0.85 0.155 86 / 0.55)",
        willChange: "transform",
      }}
    >
      {/* Warm light escaping from the slate gap when open */}
      <motion.div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-[30%] h-[55%] w-[70%] -translate-x-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(ellipse at center, oklch(0.9 0.13 86 / 0.55) 0%, oklch(0.85 0.155 86 / 0) 70%)",
          filter: "blur(8px)",
          willChange: "opacity, transform",
        }}
        animate={{ opacity: glow ? 1 : 0, scale: glow ? 1 : 0.6 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      />

      {/* The clapperboard graphic (≈60% of the frame) */}
      <div className={CLAPPER_INNER} style={CLAPPER_INNER_STYLE}>
        <ClapperBoardBody />

        {/* Hinged top slate — pivots at its left end */}
        <motion.div
          className={SLATE_CLASS}
          style={{ ...SLATE_STYLE, willChange: "transform" }}
          animate={{ rotate: open ? -34 : 0 }}
          transition={SPRING_HINGE}
        />
      </div>

      {/* Gold reflection sweep across the whole icon */}
      <motion.div
        aria-hidden
        className="pointer-events-none absolute inset-0 overflow-hidden rounded-[22%]"
        initial={false}
        animate={{ opacity: reflection ? 1 : 0 }}
        transition={{ duration: 0.2 }}
      >
        <motion.div
          className="absolute -inset-y-8 w-1/3 -skew-x-12"
          style={{
            background:
              "linear-gradient(90deg, oklch(0.85 0.155 86 / 0) 0%, oklch(0.95 0.12 88 / 0.55) 50%, oklch(0.85 0.155 86 / 0) 100%)",
            filter: "blur(4px)",
            willChange: "transform",
          }}
          animate={reflection ? { x: ["-160%", "260%"] } : { x: "-160%" }}
          transition={{ duration: 0.9, ease: "easeInOut" }}
        />
      </motion.div>
    </div>
  );
}
