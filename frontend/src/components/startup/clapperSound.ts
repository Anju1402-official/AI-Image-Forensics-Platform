/**
 * Synthesises a short, realistic movie-slate "clack" with the Web Audio API,
 * so no audio asset needs to be bundled or fetched. Two stacked transients
 * (a sharp wooden click + a lower body thud) with a fast decay read as a
 * clapperboard snap.
 *
 * Browsers block audio until a user gesture — in the preview this is fine
 * because playback is started by a click. In the real integration (which may
 * run on load) the call simply no-ops if the context can't start; it never
 * throws and never blocks the animation.
 */

type WebkitWindow = Window & { webkitAudioContext?: typeof AudioContext };

let ctx: AudioContext | null = null;

function getContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const Ctor = window.AudioContext ?? (window as WebkitWindow).webkitAudioContext;
  if (!Ctor) return null;
  if (!ctx) ctx = new Ctor();
  return ctx;
}

/**
 * Play the slate clap. `volume` is 0–1. Safe to call anywhere; failures are
 * swallowed so audio is always strictly optional.
 */
export function playSlateClap(volume = 0.5): void {
  try {
    const audio = getContext();
    if (!audio) return;
    if (audio.state === "suspended") void audio.resume();

    const now = audio.currentTime;
    const master = audio.createGain();
    master.gain.value = Math.max(0, Math.min(1, volume));
    master.connect(audio.destination);

    // Sharp wooden click — filtered noise burst.
    const noiseLen = Math.floor(audio.sampleRate * 0.05);
    const noiseBuf = audio.createBuffer(1, noiseLen, audio.sampleRate);
    const data = noiseBuf.getChannelData(0);
    for (let i = 0; i < noiseLen; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / noiseLen);
    }
    const noise = audio.createBufferSource();
    noise.buffer = noiseBuf;
    const hp = audio.createBiquadFilter();
    hp.type = "highpass";
    hp.frequency.value = 1800;
    const clickGain = audio.createGain();
    clickGain.gain.setValueAtTime(1, now);
    clickGain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);
    noise.connect(hp).connect(clickGain).connect(master);
    noise.start(now);
    noise.stop(now + 0.07);

    // Low body thud — quick sine drop.
    const body = audio.createOscillator();
    body.type = "sine";
    body.frequency.setValueAtTime(180, now);
    body.frequency.exponentialRampToValueAtTime(70, now + 0.08);
    const bodyGain = audio.createGain();
    bodyGain.gain.setValueAtTime(0.6, now);
    bodyGain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
    body.connect(bodyGain).connect(master);
    body.start(now);
    body.stop(now + 0.13);
  } catch {
    /* audio unavailable / blocked — ignore, animation continues */
  }
}
