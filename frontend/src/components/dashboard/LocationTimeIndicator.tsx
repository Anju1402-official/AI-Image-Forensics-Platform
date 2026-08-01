import { useState, useEffect } from "react";
import { MapPin, Clock, Edit2, Check } from "lucide-react";

export function LocationTimeIndicator() {
  const [location, setLocation] = useState("Detecting location...");
  const [isEditingLocation, setIsEditingLocation] = useState(false);
  const [customLocation, setCustomLocation] = useState("");
  const [time, setTime] = useState("");

  useEffect(() => {
    // Update time every minute
    const updateTime = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    };
    updateTime();
    const timer = setInterval(updateTime, 60000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    // Attempt geolocation
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            // Use a free reverse geocoding API or mock it if needed.
            // For now, let's mock it for the UI as per instructions.
            setTimeout(() => {
              setLocation("Chennai, Tamil Nadu, India");
            }, 1000);
          } catch (error) {
            setLocation("Unknown Location");
          }
        },
        (error) => {
          setLocation("Location Access Denied");
        },
      );
    } else {
      setLocation("Geolocation not supported");
    }
  }, []);

  const handleSaveLocation = () => {
    if (customLocation.trim()) {
      setLocation(customLocation);
    }
    setIsEditingLocation(false);
  };

  return (
    <div className="hidden lg:flex items-center gap-4 border-r border-white/10 pr-4 mr-2">
      {/* Location */}
      <div className="flex items-center gap-2 group">
        <MapPin className="h-4 w-4 text-[var(--gold-bright)]" />
        {isEditingLocation ? (
          <div className="flex items-center gap-1.5">
            <input
              type="text"
              autoFocus
              value={customLocation}
              onChange={(e) => setCustomLocation(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveLocation()}
              className="w-32 bg-[#111111] border border-[var(--gold-dim)] rounded-md px-2 py-0.5 text-xs text-foreground focus:outline-none"
              placeholder="Enter location"
            />
            <button
              onClick={handleSaveLocation}
              className="text-emerald-400 hover:text-emerald-300"
            >
              <Check className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : (
          <div
            className="flex items-center gap-1.5 cursor-pointer"
            onClick={() => {
              setCustomLocation(location);
              setIsEditingLocation(true);
            }}
          >
            <div className="flex flex-col">
              <span className="text-[9px] uppercase tracking-wider text-muted-foreground leading-none">
                Current Location
              </span>
              <span
                className="text-xs font-medium text-foreground max-w-[140px] truncate"
                title={location}
              >
                {location}
              </span>
            </div>
            <Edit2 className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition" />
          </div>
        )}
      </div>

      {/* Time */}
      <div className="flex items-center gap-2">
        <Clock className="h-4 w-4 text-[var(--gold-bright)]" />
        <div className="flex flex-col">
          <span className="text-[9px] uppercase tracking-wider text-muted-foreground leading-none">
            Current Time
          </span>
          <span className="text-xs font-medium text-foreground">{time}</span>
        </div>
      </div>
    </div>
  );
}
