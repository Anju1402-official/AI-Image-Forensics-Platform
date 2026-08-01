import React from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { StudioLayout } from "@/components/dashboard/StudioLayout";
import { LoadingState, ErrorState } from "@/components/dashboard/StatusStates";
import { api } from "@/lib/api";
import { CountryIntelligenceView } from "@/components/creator-intelligence/CountryIntelligenceView";

export const Route = createFileRoute("/creator-intelligence/country/$country")({
  head: ({ params }) => ({
    meta: [
      { title: `${params.country.toUpperCase()} — Country Trend Intelligence` },
      {
        name: "description",
        content: `Comprehensive social media trend analytics, 90-day forecast, and AI recommendations for ${params.country}.`,
      },
    ],
  }),
  component: CountryIntelligenceRouteComponent,
});

function CountryIntelligenceRouteComponent() {
  const { country } = Route.useParams();

  const countryQuery = useQuery({
    queryKey: ["creatorIntelligence", "country", country],
    queryFn: () => api.creatorIntelligence.countryDetails(country),
  });

  return (
    <StudioLayout>
      {countryQuery.isLoading ? (
        <LoadingState label={`Analyzing country intelligence for ${country}...`} />
      ) : countryQuery.isError ? (
        <ErrorState
          message={`Failed to load country intelligence data for ${country}.`}
          onRetry={() => countryQuery.refetch()}
        />
      ) : countryQuery.data ? (
        <CountryIntelligenceView data={countryQuery.data} />
      ) : null}
    </StudioLayout>
  );
}
