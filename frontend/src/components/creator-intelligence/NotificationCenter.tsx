import React, { useState } from "react";
import { Bell, X, Check, Flame, TrendingUp, Music, MapPin, Clock } from "lucide-react";
import { toast } from "sonner";
import { TrendNotification } from "@/lib/api";

interface NotificationCenterProps {
  notifications: TrendNotification[];
}

export function NotificationCenter({ notifications: initialNotifs }: NotificationCenterProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [notifs, setNotifs] = useState<TrendNotification[]>(initialNotifs);

  const unreadCount = notifs.filter((n) => !n.read).length;

  const handleMarkAllRead = () => {
    setNotifs((prev) => prev.map((n) => ({ ...n, read: true })));
    toast.success("All notifications marked as read");
  };

  const getIcon = (type: string) => {
    switch (type) {
      case "growth":
        return <TrendingUp className="h-4 w-4 text-emerald-400" />;
      case "duration":
        return <Clock className="h-4 w-4 text-amber-400" />;
      case "location":
        return <MapPin className="h-4 w-4 text-blue-400" />;
      case "audio":
        return <Music className="h-4 w-4 text-purple-400" />;
      default:
        return <Flame className="h-4 w-4 text-[var(--gold-bright)]" />;
    }
  };

  return (
    <div className="relative shrink-0">
      {/* Bell Trigger */}
      <button
        onClick={() => setIsOpen((v) => !v)}
        className="relative grid h-10 w-10 place-items-center rounded-full border border-white/10 bg-[#111111] text-muted-foreground transition hover:text-foreground hover:border-[var(--gold)]"
        aria-label="Trend Intelligence Notifications"
      >
        <Bell className="h-4 w-4" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-[var(--gold-bright)] text-[10px] font-bold text-black shadow-[0_0_10px_var(--gold-bright)]">
            {unreadCount}
          </span>
        )}
      </button>

      {/* Notifications Popover */}
      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute right-0 top-full z-50 mt-3 w-80 sm:w-96 rounded-2xl border border-[oklch(0.85_0.155_86/0.25)] bg-[#0e0e0e]/95 p-4 shadow-[var(--shadow-premium)] backdrop-blur-xl space-y-3">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div className="flex items-center gap-2">
                <span className="font-display text-xs uppercase tracking-widest text-[var(--gold-bright)]">
                  Live Trend Updates
                </span>
                {unreadCount > 0 && (
                  <span className="rounded-md border border-[oklch(0.85_0.155_86/0.4)] bg-[oklch(0.85_0.155_86/0.12)] px-1.5 py-0.5 text-[9px] font-bold text-[var(--gold-bright)]">
                    {unreadCount} New
                  </span>
                )}
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="text-muted-foreground hover:text-foreground transition"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* List */}
            <div className="max-h-72 overflow-y-auto space-y-2 pr-1 scrollbar-hide">
              {notifs.map((n) => (
                <div
                  key={n.id}
                  className={`flex items-start gap-3 rounded-xl border p-3 transition ${
                    n.read
                      ? "border-white/5 bg-black/30 opacity-75"
                      : "border-[oklch(0.85_0.155_86/0.2)] bg-[#141414]"
                  }`}
                >
                  <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-white/10 bg-white/5">
                    {getIcon(n.type)}
                  </div>
                  <div className="flex-1 min-w-0 space-y-1">
                    <div className="flex items-center justify-between text-xs font-semibold text-foreground">
                      <span className="truncate">{n.title}</span>
                      <span className="text-[9px] text-muted-foreground">{n.timestamp}</span>
                    </div>
                    <p className="text-[11px] leading-relaxed text-muted-foreground">{n.message}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Footer */}
            {unreadCount > 0 && (
              <button
                onClick={handleMarkAllRead}
                className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-white/10 bg-white/5 py-2 text-center text-xs font-medium text-[var(--gold-bright)] hover:bg-white/10 transition"
              >
                <Check className="h-3.5 w-3.5" />
                <span>Mark all as read</span>
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
