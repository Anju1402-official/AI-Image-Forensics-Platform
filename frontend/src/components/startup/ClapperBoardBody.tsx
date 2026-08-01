/**
 * The lower "board" of the clapperboard (the slate face with its faint
 * writing lines). Shared by the animated clapper and the static logo so both
 * render the identical board.
 */
export function ClapperBoardBody() {
  return (
    <div className="absolute inset-x-0 bottom-0 top-[34%] rounded-[10%] border border-[oklch(0.85_0.155_86/0.5)] bg-[linear-gradient(180deg,oklch(0.2_0.02_70)_0%,oklch(0.12_0.01_70)_100%)]">
      <div className="absolute inset-x-[14%] top-[24%] h-px bg-[oklch(0.85_0.155_86/0.28)]" />
      <div className="absolute inset-x-[14%] top-[52%] h-px bg-[oklch(0.85_0.155_86/0.2)]" />
      <div className="absolute inset-x-[14%] top-[80%] h-px bg-[oklch(0.85_0.155_86/0.14)]" />
    </div>
  );
}
