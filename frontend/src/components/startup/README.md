# Startup Animation

A luxury, ~5.9s cinematic launch sequence that plays once per browser session
and transitions seamlessly into the Sign In page.

## Files

| File | Purpose |
|---|---|
| `StartupAnimation.tsx` | The animation itself. Full-screen overlay, 5 scenes, GPU transforms only. Props: `onComplete`, `reveal`, `muted`, `autoPlay`, `clickToSkip`. |
| `StartupPreview.tsx` | Standalone review harness (Replay / Mute). Mounted at `/startup-preview`. |
| `AnimatedClapper.tsx` | The hinged clapperboard, styled with the exact login-icon tokens. |
| `DustField.tsx` | Sparse floating dust (client-generated, no hydration mismatch). |
| `SignInPreviewCard.tsx` | Decoupled visual copy of the login card, used only as the preview's reveal target. |
| `useStartupSequence.ts` | Scene timeline hook + `hasPlayedStartup()` / `markStartupPlayed()` session helpers + `prefersReducedMotion()`. |
| `clapperSound.ts` | Web-Audio slate "clack" (no asset, never throws). |

## Reviewing

Run the app and open **`/startup-preview`**. Nothing else in the app is
touched by this route.

## Integrating (only after approval)

The animation is designed to overlay the **real** Sign In page without
modifying it — the login route renders underneath and becomes interactive the
moment the overlay unmounts. Suggested wiring (do this in a small wrapper, not
by editing `login.tsx`):

```tsx
// e.g. a StartupGate that wraps the login route's element
import { useState } from "react";
import { StartupAnimation } from "@/components/startup/StartupAnimation";
import {
  hasPlayedStartup,
  markStartupPlayed,
} from "@/components/startup/useStartupSequence";

function StartupGate({ children }: { children: React.ReactNode }) {
  // Runs once per browser session; returning users skip straight to Sign In.
  const [showIntro, setShowIntro] = useState(() => !hasPlayedStartup());

  return (
    <>
      {children /* the real Sign In page, already mounted underneath */}
      {showIntro && (
        <StartupAnimation
          onComplete={() => {
            markStartupPlayed();
            setShowIntro(false);
          }}
        />
      )}
    </>
  );
}
```

Notes:
- Because the real page is mounted beneath the overlay, no `reveal` prop is
  needed for integration — the clip-path reveal simply unveils the page below.
- `prefers-reduced-motion` and the click/Skip control both jump straight to the
  end and call `onComplete`, so users are never trapped in the intro.
- Audio only fires after a user gesture (browser policy). On a no-gesture load
  it silently no-ops; the visuals are unaffected.
