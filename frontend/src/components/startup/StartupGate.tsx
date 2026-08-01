import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { StartupAnimation } from "./StartupAnimation";
import { hasPlayedStartup, markStartupPlayed } from "./useStartupSequence";

interface StartupGateProps {
  onComplete?: () => void;
}

export function StartupGate({ onComplete }: StartupGateProps) {
  const [show, setShow] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!hasPlayedStartup()) {
      setShow(true);
    }
  }, []);

  if (!show) return null;

  return (
    <StartupAnimation
      revealBeneath
      onComplete={() => {
        markStartupPlayed();
        setShow(false);
        if (onComplete) {
          onComplete();
        }
      }}
    />
  );
}
