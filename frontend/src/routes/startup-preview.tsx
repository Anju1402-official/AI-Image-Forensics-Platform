import { createFileRoute } from "@tanstack/react-router";
import { StartupPreview } from "@/components/startup/StartupPreview";

/**
 * Isolated review route for the launch animation. Additive only — it does not
 * change or wrap any existing route. Visit /startup-preview to review, then
 * approve before integrating in front of the real Sign In page.
 */
export const Route = createFileRoute("/startup-preview")({
  head: () => ({ meta: [{ title: "Startup Preview — Studio" }] }),
  component: StartupPreview,
});
