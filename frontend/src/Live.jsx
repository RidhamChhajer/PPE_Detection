import { useEffect, useState } from "react";
import LiveCamera from "./LiveCamera";
import { Users, Percent, AlertTriangle, Video, Clock3 } from "lucide-react";
import {
  GogglesIcon,
  SafetyShoeIcon,
  EarplugsIcon,
  ShieldCheckLogo,
  BekaertLogo,
} from "./components/icons";

function WristIcon({ size = 52, color = "#00a854" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      stroke={color}
      strokeWidth="3.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="m38 26 3-17h18l3 17M38 72l3 18h18l3-18" />
      <rect x="29" y="26" width="42" height="46" rx="11" />
      <path d="M50 38v12l10 6M78 43h4v13h-4M15 85l70-70" />
    </svg>
  );
}
export default function Live({
  connected,
  connecting,
  status,
  catalog,
  error,
  busy,
  onToggle,
  onConnect,
}) {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);
  const observed = (key) => {
    const view = status?.views?.[key === "shoes" ? "lower" : "upper"];
    return status?.running && view?.connected && !view.stale && view.model_ready
      ? view.summary?.[key] || "unknown"
      : "unknown";
  };
  const ready = connected && !!status;
  const metrics = [
    [
      Users,
      "Total workers detected",
      "—",
      "Coming soon",
      "blue",
    ],
    [
      AlertTriangle,
      "Violations",
      "—",
      "Coming soon",
      "red",
    ],
    [
      Percent,
      "Violation percentage",
      "—",
      "Coming soon",
      "blue",
    ],
    [
      Users,
      "Workers currently in camera",
      "—",
      "Camera 1 · Coming soon",
      "green",
    ],
  ];
  return (
    <div className="kiosk-viewport">
      <div className="kiosk-screen">
        {error && (
          <div className="ui-notice error" role="alert">
            {error}
          </div>
        )}
        {status &&
          (!status.views?.upper?.model_ready ||
            !status.views?.lower?.model_ready) && (
            <div className="ui-notice">
              Live camera preview is available. Trained PPE weights are required
              for detection.
            </div>
          )}
        <header className="ppe-header">
          <div className="brand-section">
            <div className="brand-icon-shield">
              <ShieldCheckLogo size={46} />
            </div>
            <div className="brand-text-wrap">
              <h1 className="brand-title" tabIndex={-1}>
                AI Based PPE Monitoring & Warning
              </h1>
              <p className="brand-subtitle">Safety First,No Harm to ANYONE</p>
            </div>
          </div>
          <div className="header-meta-group">
            <div className="time-badge-card">
              <span className="time-date">
                {now.toLocaleDateString("en-GB", {
                  timeZone: "Asia/Kolkata",
                  weekday: "short",
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                })}
              </span>
              <span className="time-clock">
                {now.toLocaleTimeString("en-US", {
                  timeZone: "Asia/Kolkata",
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
            <div className="system-status-pill">
              <span
                className={`live-pulse ${!status?.running ? "is-offline" : ""}`}
              />
              <span className="status-label">
                System
                <br />
                {connecting
                  ? "Connecting"
                  : !ready
                    ? "Offline"
                    : status.running
                      ? "Online"
                      : "Stopped"}
              </span>
            </div>
            <button
              className="ui-button primary detection-toggle"
              disabled={busy || connecting}
              onClick={ready ? onToggle : onConnect}
            >
              {busy
                ? "Please wait…"
                : connecting
                  ? "Connecting…"
                  : !ready
                    ? "Retry connection"
                    : status.running
                      ? "Stop detection"
                      : "Start detection"}
            </button>
          </div>
        </header>
        <section className="ppe-status-grid" aria-label="Visible PPE status">
          {[
            ["goggles", "Safety Goggles", GogglesIcon],
            ["wrist", "Clear Wrists", WristIcon],
            ["shoes", "Safety Shoes", SafetyShoeIcon],
            ["earplugs", "Earplugs", EarplugsIcon],
          ].map(([key, label, Icon]) => {
            const state = key === "earplugs" ? "pending" : observed(key),
              tone =
                state === "detected"
                  ? "green"
                  : state === "violation"
                    ? "red"
                    : "neutral";
            return (
              <div className={`ppe-status-card ${tone}`} key={key} data-status={state}>
                <div className={`ppe-icon-circle ${tone}`}>
                  <Icon
                    size={58}
                    color="#475569"
                  />
                </div>
                <div className={`ppe-pill-badge ${tone}`}>
                  <span className={`pill-icon-circle ${tone}-text`}>
                    {state === "pending" ? (
                      <Clock3 size={14} aria-hidden="true" />
                    ) : tone === "green" ? "✓" : tone === "red" ? "✕" : "–"}
                  </span>
                  <span>{label}</span>
                </div>
                <span className={`detection-result-text ${tone}`}>
                  {state === "pending"
                    ? "IN PROGRESS"
                    : state === "detected"
                      ? "DETECTED"
                      : state === "violation"
                        ? "NOT DETECTED"
                        : "NOT VISIBLE"}
                </span>
              </div>
            );
          })}
        </section>
        <section className="metrics-row">
          {metrics.map(([Icon, label, value, detail, tone]) => (
            <div className="metric-card" key={label}>
              <div className="metric-header">
                <div className={`metric-icon-box ${tone}`}>
                  <Icon size={18} />
                </div>
                <span className="metric-label">{label}</span>
              </div>
              <div className="metric-content">
                <span className="metric-value">{value}</span>
                <span className={`metric-subtext ${tone}`}>{detail}</span>
              </div>
            </div>
          ))}
        </section>
        {["upper", "lower"].map((view, index) => {
          const camera = status?.views?.[view],
            available = status?.running && camera?.connected;
          return (
            <article className={`camera-card ${view}-camera`} key={view}>
              <div className="camera-card-header">
                <div className="camera-header-left">
                  <div className="camera-icon-box">
                    <Video size={18} />
                  </div>
                  <div className="camera-title-wrap">
                    <h2 className="camera-title">
                      Camera {index + 1} –{" "}
                      {index === 0 ? "Upper Body" : "Lower Body"}
                    </h2>
                    <span className="camera-subtitle">
                      Detects:{" "}
                      {index === 0
                        ? "Goggles & Wrist Accessories"
                        : "Safety Shoes"}
                    </span>
                  </div>
                </div>
                <div className="camera-live-badge">
                  <span
                    className={`live-pulse ${!available ? "is-offline" : ""}`}
                  />
                  <span>
                    {available
                      ? camera.model_ready
                        ? camera.stale
                          ? "UPDATING"
                          : "LIVE"
                        : "CAMERA ONLY"
                      : "OFFLINE"}
                  </span>
                </div>
              </div>
              <div className="camera-viewport">
                <LiveCamera
                  key={`${view}-${camera?.camera}-${!!status?.running}`}
                  view={view}
                  message={camera?.message}
                  running={status?.running}
                />
                <div className="camera-telemetry">
                  {available && camera.actual?.width
                    ? `${camera.actual.width} × ${camera.actual.height} · Preview ${camera.preview_fps ?? "—"} FPS · Detection ${camera.processing_fps ?? "—"} FPS${camera.preview_encoder ? ` · ${camera.preview_encoder}` : ""}${camera.stale ? " · Detection results delayed" : ""}`
                    : "No live frames"}
                </div>
              </div>
              <p className="camera-source">
                {catalog?.find(
                  (device) =>
                    device.id === camera?.camera ||
                    device.index === camera?.camera,
                )?.name || "Camera source not available"}
                {camera?.camera != null &&
                status?.views?.upper?.camera === status?.views?.lower?.camera
                  ? " · Shared camera"
                  : ""}
              </p>
              <div className="camera-detects-strip">
                <span className="camera-detects-label">
                  Camera {index + 1} Detects:
                </span>
                {index === 0 ? (
                  <>
                    <div className="detects-item goggles">
                      <GogglesIcon size={24} color="#00a854" />
                      <span>Safety Goggles</span>
                    </div>
                    <span className="detects-separator">|</span>
                    <div className="detects-item wrists">
                      <WristIcon size={22} color="#ef4444" />
                      <span>Wrist Accessories</span>
                    </div>
                  </>
                ) : (
                  <div className="detects-item shoes">
                    <SafetyShoeIcon size={26} color="#00a854" />
                    <span>Safety Shoes</span>
                  </div>
                )}
              </div>
            </article>
          );
        })}
        <footer className="project-footer" aria-label="Project credits">
          <div className="institute-credit">
            <img
              className="vit-logo"
              src="/static/branding/vit-pune.png"
              width="1575"
              height="708"
              alt="Vishwakarma Institute of Technology, Pune"
            />
          </div>
          <div className="team-credit">
            <span>Project team</span>
            <ul aria-label="Project team members">
              {["Ridham Chhajer", "Kedar Naygaonkar", "Kedar Pethkar", "Sanskruti Chaure", "Sahil Nehate", "Om Parmar"].map(name => <li key={name}>{name}</li>)}
            </ul>
          </div>
          <div className="industry-credit">
            <BekaertLogo />
          </div>
        </footer>
      </div>
    </div>
  );
}
