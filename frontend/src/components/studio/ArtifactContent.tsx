import type { Artifact } from "@/lib/api";

/* Renders the `content` JSON of any generated artifact into a readable,
   on-brand layout. One component so the generator result panes and the
   library detail view stay consistent. */

/* ------------------------------------------------------------------ */
/* Shapes of the (dynamic) artifact.content JSON we read below.       */
/* The backend generators are the source of truth; these interfaces   */
/* describe the fields this renderer knows how to display.            */
/* ------------------------------------------------------------------ */

interface ScriptScene {
  heading?: string;
  action?: string;
  dialogue?: string[];
}

interface NamedEntry {
  name?: string;
  description?: string;
}

interface StoryboardPanelContent {
  panel_number?: number;
  shot_type?: string;
  camera_angle?: string;
  description?: string;
  mood_lighting?: string;
  key_visual_elements?: string;
}

type ArtifactContentShape = Record<string, unknown>;

/* Safe accessors — artifact.content is untyped JSON, so narrow before use. */
function str(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

function strArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function objArray<T>(v: unknown): T[] {
  return Array.isArray(v) ? (v.filter((x) => x && typeof x === "object") as T[]) : [];
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[10px] uppercase tracking-[0.24em] text-[var(--gold-dim)]">
        {title}
      </div>
      <div className="text-sm leading-relaxed text-foreground/90">{children}</div>
    </div>
  );
}

function Chips({ items }: { items: string[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {items.map((it, i) => (
        <span
          key={i}
          className="rounded-full border border-[oklch(0.85_0.155_86/0.25)] bg-[oklch(0.85_0.155_86/0.06)] px-2.5 py-1 text-xs text-[var(--gold-bright)]"
        >
          {it}
        </span>
      ))}
    </div>
  );
}

function Bullets({ items }: { items: string[] }) {
  return (
    <ul className="space-y-1.5">
      {items.map((it, i) => (
        <li key={i} className="flex gap-2 text-sm text-foreground/90">
          <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-[var(--gold-bright)]" />
          <span>{it}</span>
        </li>
      ))}
    </ul>
  );
}

export function ArtifactContent({ artifact }: { artifact: Artifact }) {
  const c = (artifact.content ?? {}) as ArtifactContentShape;
  const templateKind = str(c.template_kind);

  if (artifact.kind === "story" || (artifact.kind === "template" && templateKind === "story")) {
    return (
      <div className="space-y-5">
        {str(c.logline) && <Section title="Logline">{str(c.logline)}</Section>}
        <div className="flex flex-wrap gap-2">
          {str(c.genre) && <Chips items={[str(c.genre)!]} />}
          {str(c.tone) && <Chips items={[str(c.tone)!]} />}
        </div>
        {str(c.synopsis) && (
          <Section title="Synopsis">
            <p className="whitespace-pre-line">{str(c.synopsis)}</p>
          </Section>
        )}
        {strArray(c.themes).length > 0 && (
          <Section title="Themes">
            <Chips items={strArray(c.themes)} />
          </Section>
        )}
        {strArray(c.main_characters).length > 0 && (
          <Section title="Main Characters">
            <Bullets items={strArray(c.main_characters)} />
          </Section>
        )}
        {strArray(c.three_act_beats).length > 0 && (
          <Section title="Act Beats">
            <Bullets items={strArray(c.three_act_beats)} />
          </Section>
        )}
        {strArray(c.structure).length > 0 && (
          <Section title="Structure">
            <Bullets items={strArray(c.structure)} />
          </Section>
        )}
        {str(c.prompt_starter) && <Section title="Prompt Starter">{str(c.prompt_starter)}</Section>}
      </div>
    );
  }

  if (artifact.kind === "script" || (artifact.kind === "template" && templateKind === "script")) {
    const scenes = objArray<ScriptScene>(c.scenes);
    return (
      <div className="space-y-5">
        {str(c.logline) && <Section title="Logline">{str(c.logline)}</Section>}
        {scenes.map((s, i) => (
          <div key={i} className="rounded-xl border border-white/5 bg-black/30 p-4">
            <div className="font-mono text-xs uppercase tracking-widest text-[var(--gold-bright)]">
              {s.heading}
            </div>
            {s.action && (
              <p className="mt-2 text-sm leading-relaxed text-foreground/80">{s.action}</p>
            )}
            {strArray(s.dialogue).length > 0 && (
              <div className="mt-3 space-y-1.5">
                {strArray(s.dialogue).map((line, j) => (
                  <p key={j} className="text-sm text-foreground/90">
                    {line}
                  </p>
                ))}
              </div>
            )}
          </div>
        ))}
        {strArray(c.scenes_outline).length > 0 && (
          <Section title="Scene Outline">
            <Bullets items={strArray(c.scenes_outline)} />
          </Section>
        )}
        {str(c.prompt_starter) && <Section title="Prompt Starter">{str(c.prompt_starter)}</Section>}
      </div>
    );
  }

  if (
    artifact.kind === "character" ||
    (artifact.kind === "template" && templateKind === "character")
  ) {
    return (
      <div className="space-y-5">
        {str(c.one_line) && <Section title="Summary">{str(c.one_line)}</Section>}
        <div className="flex flex-wrap gap-2">
          {str(c.role) && <Chips items={[str(c.role)!]} />}
          {str(c.age_range) && <Chips items={[str(c.age_range)!]} />}
        </div>
        {str(c.backstory) && (
          <Section title="Backstory">
            <p className="whitespace-pre-line">{str(c.backstory)}</p>
          </Section>
        )}
        {strArray(c.personality_traits).length > 0 && (
          <Section title="Personality">
            <Chips items={strArray(c.personality_traits)} />
          </Section>
        )}
        {strArray(c.motivations).length > 0 && (
          <Section title="Motivations">
            <Bullets items={strArray(c.motivations)} />
          </Section>
        )}
        {strArray(c.flaws).length > 0 && (
          <Section title="Flaws">
            <Bullets items={strArray(c.flaws)} />
          </Section>
        )}
        {str(c.arc) && <Section title="Arc">{str(c.arc)}</Section>}
        {str(c.visual_description) && <Section title="Visual">{str(c.visual_description)}</Section>}
        {strArray(c.fields).length > 0 && (
          <Section title="Fields">
            <Chips items={strArray(c.fields)} />
          </Section>
        )}
        {str(c.prompt_starter) && <Section title="Prompt Starter">{str(c.prompt_starter)}</Section>}
      </div>
    );
  }

  if (artifact.kind === "world" || (artifact.kind === "template" && templateKind === "world")) {
    const locations = objArray<NamedEntry>(c.locations);
    const factions = objArray<NamedEntry>(c.factions);
    return (
      <div className="space-y-5">
        {str(c.overview) && (
          <Section title="Overview">
            <p className="whitespace-pre-line">{str(c.overview)}</p>
          </Section>
        )}
        {str(c.geography) && <Section title="Geography">{str(c.geography)}</Section>}
        {str(c.history) && <Section title="History">{str(c.history)}</Section>}
        {strArray(c.rules).length > 0 && (
          <Section title="Rules">
            <Bullets items={strArray(c.rules)} />
          </Section>
        )}
        {locations.length > 0 && (
          <Section title="Locations">
            <div className="space-y-2">
              {locations.map((l, i) => (
                <div key={i} className="rounded-lg border border-white/5 bg-black/20 p-3">
                  <div className="text-sm text-foreground">{l.name}</div>
                  <div className="text-xs text-muted-foreground">{l.description}</div>
                </div>
              ))}
            </div>
          </Section>
        )}
        {factions.length > 0 && (
          <Section title="Factions">
            <div className="space-y-2">
              {factions.map((f, i) => (
                <div key={i} className="rounded-lg border border-white/5 bg-black/20 p-3">
                  <div className="text-sm text-foreground">{f.name}</div>
                  <div className="text-xs text-muted-foreground">{f.description}</div>
                </div>
              ))}
            </div>
          </Section>
        )}
        {strArray(c.fields).length > 0 && (
          <Section title="Fields">
            <Chips items={strArray(c.fields)} />
          </Section>
        )}
        {str(c.prompt_starter) && <Section title="Prompt Starter">{str(c.prompt_starter)}</Section>}
      </div>
    );
  }

  if (
    artifact.kind === "storyboard" ||
    (artifact.kind === "template" && templateKind === "storyboard")
  ) {
    const panels = objArray<StoryboardPanelContent>(c.panels);
    return (
      <div className="space-y-5">
        {str(c.summary) && <Section title="Sequence">{str(c.summary)}</Section>}
        {panels.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {panels.map((p, i) => (
              <div key={i} className="rounded-xl border border-white/5 bg-black/30 p-4">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs text-[var(--gold-bright)]">
                    Panel {p.panel_number}
                  </span>
                </div>
                <div className="mt-2 flex aspect-video flex-col items-center justify-center gap-1 rounded-lg border border-dashed border-white/10 bg-black/40 p-2 text-center">
                  <div className="text-xs font-semibold text-foreground">{p.shot_type}</div>
                  <div className="text-[10px] text-muted-foreground">{p.camera_angle}</div>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-foreground/80">{p.description}</p>
                {p.mood_lighting && (
                  <p className="mt-1 text-[11px] text-muted-foreground">Mood: {p.mood_lighting}</p>
                )}
                {p.key_visual_elements && (
                  <p className="text-[11px] text-muted-foreground">
                    Elements: {p.key_visual_elements}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
        {strArray(c.coverage).length > 0 && (
          <Section title="Coverage">
            <Bullets items={strArray(c.coverage)} />
          </Section>
        )}
        {str(c.prompt_starter) && <Section title="Prompt Starter">{str(c.prompt_starter)}</Section>}
      </div>
    );
  }

  // Fallback: pretty-print whatever's there.
  return (
    <pre className="overflow-x-auto rounded-lg border border-white/5 bg-black/30 p-4 text-xs text-foreground/80">
      {JSON.stringify(c, null, 2)}
    </pre>
  );
}
