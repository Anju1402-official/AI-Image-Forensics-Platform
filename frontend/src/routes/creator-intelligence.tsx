import React, { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { StudioLayout } from "@/components/dashboard/StudioLayout";
import { LoadingState, ErrorState } from "@/components/dashboard/StatusStates";
import { api, ApiError, CountrySummary, CurrentTrendItem } from "@/lib/api";
import { LandingScreen } from "@/components/creator-intelligence/LandingScreen";
import { TrendAnalysisProgress } from "@/components/creator-intelligence/TrendAnalysisProgress";
import { TrendResultsView } from "@/components/creator-intelligence/TrendResultsView";
import { GenerateSimilarContentModal } from "@/components/creator-intelligence/GenerateSimilarContentModal";
import { NotificationCenter } from "@/components/creator-intelligence/NotificationCenter";
import { toast } from "sonner";

export const Route = createFileRoute("/creator-intelligence")({
  head: () => ({
    meta: [
      { title: "AI Creator Trend Intelligence — Studio" },
      {
        name: "description",
        content:
          "Discover location-based social media trends across Instagram Reels and YouTube Shorts before creating content.",
      },
    ],
  }),
  component: CreatorTrendIntelligencePage,
});

function CreatorTrendIntelligencePage() {
  const [selectedCountry, setSelectedCountry] = useState<CountrySummary | null>(null);
  const [radiusKm, setRadiusKm] = useState<string>("50 km");
  const [stage, setStage] = useState<"landing" | "analyzing" | "results">("landing");
  const [analysisResult, setAnalysisResult] = useState<any>(null);

  // Content Generation Modal State
  const [generatedContent, setGeneratedContent] = useState<any>(null);
  const [isGeneratingModalOpen, setIsGeneratingModalOpen] = useState(false);

  // Load supported countries
  const countriesQuery = useQuery({
    queryKey: ["creatorIntelligence", "countries"],
    queryFn: () => api.creatorIntelligence.countries(),
  });

  // Load notifications
  const notificationsQuery = useQuery({
    queryKey: ["creatorIntelligence", "notifications"],
    queryFn: () => api.creatorIntelligence.notifications(),
  });

  // Auto select India by default if available
  React.useEffect(() => {
    if (countriesQuery.data && countriesQuery.data.length > 0 && !selectedCountry) {
      const india = countriesQuery.data.find((c) => c.code === "IN") || countriesQuery.data[0];
      setSelectedCountry(india);
    }
  }, [countriesQuery.data, selectedCountry]);

  const handleStartAnalysis = async () => {
    if (!selectedCountry) {
      toast.error("Please select a target location first.");
      return;
    }
    setStage("analyzing");
    try {
      const res = await api.creatorIntelligence.analyze({
        location_name: selectedCountry.name,
        country_code: selectedCountry.code,
        radius_km: radiusKm,
      });
      setAnalysisResult(res);
    } catch (err) {
      toast.error("Failed to analyze location trends. Please try again.");
      setStage("landing");
    }
  };

  const handleGenerateContent = async (trend: CurrentTrendItem) => {
    try {
      toast.loading("Generating original content blueprint...");
      const res = await api.creatorIntelligence.generateContent(trend.id, trend.title);
      toast.dismiss();
      setGeneratedContent(res);
      setIsGeneratingModalOpen(true);
    } catch (err) {
      toast.dismiss();
      toast.error("Could not generate content ideas right now.");
    }
  };

  return (
    <StudioLayout>
      <div className="space-y-6">
        {/* Top Header Notification Bar */}
        <div className="flex items-center justify-between">
          <div className="text-[10px] uppercase tracking-[0.24em] text-[var(--gold-dim)]">
            Module: AI Creator Trend Intelligence
          </div>
          {notificationsQuery.data && (
            <NotificationCenter notifications={notificationsQuery.data} />
          )}
        </div>

        {/* Main Stage Switching */}
        {countriesQuery.isLoading ? (
          <LoadingState label="Loading trend intelligence environment…" />
        ) : countriesQuery.isError ? (
          <ErrorState
            message="Could not load map data. Please verify backend service connection."
            onRetry={() => countriesQuery.refetch()}
          />
        ) : stage === "landing" ? (
          <LandingScreen
            countries={countriesQuery.data || []}
            selectedCountry={selectedCountry}
            onSelectCountry={(c) => setSelectedCountry(c)}
            radiusKm={radiusKm}
            onChangeRadius={(r) => setRadiusKm(r)}
            onAnalyzeTrends={handleStartAnalysis}
          />
        ) : stage === "analyzing" ? (
          <TrendAnalysisProgress
            locationName={selectedCountry?.name || "Target Region"}
            onComplete={() => setStage("results")}
          />
        ) : (
          <TrendResultsView
            locationName={selectedCountry?.name || "Target Region"}
            radiusKm={radiusKm}
            pastTrends={analysisResult?.past_trends_30_pct || []}
            currentTrends={analysisResult?.current_trends_70_pct || []}
            lastUpdated={analysisResult?.last_updated || "Today 6:42 PM IST"}
            onReset={() => setStage("landing")}
            onGenerateContent={handleGenerateContent}
          />
        )}
      </div>

      {/* AI Generate Similar Content Modal */}
      <GenerateSimilarContentModal
        content={generatedContent}
        isOpen={isGeneratingModalOpen}
        onClose={() => setIsGeneratingModalOpen(false)}
      />
    </StudioLayout>
  );
}
