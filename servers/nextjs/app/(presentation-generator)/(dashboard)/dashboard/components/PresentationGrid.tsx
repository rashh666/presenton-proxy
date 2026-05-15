import React from "react";
import { PresentationCard } from "./PresentationCard";
import { PresentationResponse } from "@/app/(presentation-generator)/services/api/dashboard";
import { EmptyState } from "./EmptyState";

interface PresentationGridProps {
  presentations: PresentationResponse[];
  isLoading?: boolean;
  error?: string | null;
  onPresentationDeleted?: (presentationId: string) => void;
}

export const PresentationGrid = ({
  presentations,
  isLoading = false,
  error = null,
  onPresentationDeleted,
}: PresentationGridProps) => {
  const ShimmerCard = () => (
    <div className="flex flex-col gap-4 min-h-[216px] bg-white/70 rounded-lg p-4 animate-pulse">
      <div className="w-full h-24 bg-gray-200 rounded-lg"></div>
      <div className="space-y-3">
        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        <div className="h-3 bg-gray-200 rounded w-1/2"></div>
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 sm:gap-6 w-full">
        {[...Array(12)].map((_, i) => (
          <ShimmerCard key={i} />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[220px] items-center justify-center rounded-xl border border-[#EDEEEF] bg-white/80">
        <div className="text-center text-gray-500">
          <p className="mb-2">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="text-primary hover:text-primary/80 underline"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  if (!presentations || presentations.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {presentations.map((presentation) => (
        <PresentationCard
          key={presentation.id}
          id={presentation.id}
          title={presentation.title}
          presentation={presentation}
          onDeleted={onPresentationDeleted}
        />
      ))}
    </div>
  );
};
