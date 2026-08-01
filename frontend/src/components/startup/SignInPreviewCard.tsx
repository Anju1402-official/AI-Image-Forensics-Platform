/**
 * A self-contained VISUAL stand-in for the real Sign In card, used only as
 * the reveal target inside the preview so this component stays fully
 * decoupled from the real /login route and its auth logic. It mirrors the
 * existing login card's styling (same tokens, same icon frame) so the dock
 * hand-off looks seamless. In the real integration the animation overlays
 * the actual Sign In page instead of this mock.
 */
import { Eye } from "lucide-react";
import { MovieSlateLogo } from "./MovieSlateLogo";

export function SignInPreviewCard({ hideIcon = false }: { hideIcon?: boolean }) {
  return (
    <div className="relative w-full max-w-md rounded-2xl border border-[oklch(0.85_0.155_86/0.15)] bg-[#0d0d0d]/90 p-8 backdrop-blur-xl shadow-[var(--shadow-premium)]">
      <div className="flex flex-col items-center gap-3 text-center">
        {/* Icon frame — matches the login page. Hidden while the animated
            clapper is still docking into this exact spot. */}
        <div style={{ opacity: hideIcon ? 0 : 1 }}>
          <MovieSlateLogo className="h-14 w-14" />
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-[var(--gold-dim)]">
            AI Pre-Production
          </div>
          <h1 className="font-display text-2xl tracking-[0.18em] text-foreground">STUDIO</h1>
        </div>
      </div>

      <h2 className="mt-8 font-display text-xl text-foreground">Welcome back</h2>
      <p className="mt-1 text-sm text-muted-foreground">Log in to continue to your studio.</p>

      <div className="mt-6 space-y-4">
        <div>
          <label className="mb-1.5 block text-xs uppercase tracking-widest text-[var(--gold-dim)]">
            Email
          </label>
          <div className="flex h-[42px] items-center rounded-lg border border-white/10 bg-black/40 px-4 text-sm text-muted-foreground">
            you@studio.com
          </div>
        </div>
        <div>
          <label className="mb-1.5 block text-xs uppercase tracking-widest text-[var(--gold-dim)]">
            Password
          </label>
          <div className="relative flex h-[42px] items-center rounded-lg border border-white/10 bg-black/40 px-4 text-sm text-muted-foreground">
            ••••••••
            <Eye className="absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          </div>
        </div>
        <div className="flex h-[42px] w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-[var(--gold)] to-[var(--gold-bright)] text-sm font-semibold text-black shadow-[0_0_24px_-6px_var(--gold-bright)]">
          Log In
        </div>
      </div>

      <p className="mt-6 text-center text-sm text-muted-foreground">
        Don't have an account? <span className="text-[var(--gold-bright)]">Sign up</span>
      </p>
    </div>
  );
}
