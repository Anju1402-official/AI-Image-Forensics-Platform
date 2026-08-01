/**
 * Tiny floating cinematic dust. Deliberately sparse (a couple dozen motes)
 * and rendered with transform/opacity only, so it stays GPU-cheap and holds
 * 60fps. Particle parameters are generated once on the client (inside a lazy
 * useState initialiser) to avoid SSR/hydration mismatches from Math.random.
 */
import { useState } from "react";
import { motion } from "framer-motion";

interface Mote {
  id: number;
  x: number; // vw
  y: number; // vh
  size: number; // px
  drift: number; // px horizontal sway
  rise: number; // px vertical travel
  duration: number; // s
  delay: number; // s
  opacity: number;
}

function createMotes(count: number): Mote[] {
  return Array.from({ length: count }, (_, id) => ({
    id,
    x: Math.random() * 100,
    y: 40 + Math.random() * 60,
    size: 1 + Math.random() * 2.5,
    drift: (Math.random() - 0.5) * 40,
    rise: 60 + Math.random() * 140,
    duration: 6 + Math.random() * 6,
    delay: Math.random() * 4,
    opacity: 0.12 + Math.random() * 0.35,
  }));
}

export function DustField({ count = 22 }: { count?: number }) {
  const [motes] = useState<Mote[]>(() => createMotes(count));

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
      {motes.map((m) => (
        <motion.span
          key={m.id}
          className="absolute rounded-full"
          style={{
            left: `${m.x}vw`,
            top: `${m.y}vh`,
            width: m.size,
            height: m.size,
            background:
              "radial-gradient(circle, oklch(0.9 0.09 86 / 0.9) 0%, oklch(0.85 0.155 86 / 0) 70%)",
            willChange: "transform, opacity",
          }}
          initial={{ opacity: 0, y: 0, x: 0 }}
          animate={{
            opacity: [0, m.opacity, m.opacity, 0],
            y: [-0, -m.rise],
            x: [0, m.drift],
          }}
          transition={{
            duration: m.duration,
            delay: m.delay,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
