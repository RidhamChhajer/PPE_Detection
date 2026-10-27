import { useEffect, useRef, useState } from "react";
import {
  Upload,
  Download,
  ImagePlay,
  Video,
  ShieldCheck,
  Users,
  AlertTriangle,
  RotateCcw,
  Save,
  Mail,
  Calendar,
  Send,
  Clock,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Camera,
} from "lucide-react";
import { request } from "./api";

function validateSettings(form, report) {
  for (const field of form.querySelectorAll("[aria-invalid]"))
    field.removeAttribute("aria-invalid");
  const invalid = form.querySelector(
    "input:invalid, select:invalid, textarea:invalid",
  );
  if (!invalid) return true;
  invalid.setAttribute("aria-invalid", "true");
  invalid.setAttribute("aria-describedby", "settings-error");
  report(
    `${invalid.getAttribute("aria-label") || "Setting"}: ${invalid.validationMessage}`,
  );
  invalid.focus();
  return false;
}
export function PageHeader({ title, children }) {
  return (
    <header className="operator-heading">
      <span className="eyebrow">AI BASED PPE · OPERATOR CONSOLE</span>
      <h1 tabIndex={-1}>{title}</h1>
      <p>{children}</p>
    </header>
  );
}
export function TestMedia({ connected }) {
  const [file, setFile] = useState(null),
    [preview, setPreview] = useState(""),
    [view, setView] = useState("both");
  const [job, setJob] = useState(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [uploading, setUploading] = useState(false);
  const [frame, setFrame] = useState("");
  const controller = useRef(null),
    acceptPolling = useRef(true),
    input = useRef(null),
    removing = useRef(false),
    previewRef = useRef("");
  useEffect(
    () => () => {
      controller.current?.abort();
      URL.revokeObjectURL(previewRef.current);
    },
    [],
  );
  useEffect(() => {
    if (!connected) return;
    let disposed = false,
      timer,
      active;
    const poll = async () => {
      try {
        active = new AbortController();
        const data = await request("/api/test", {
          signal: AbortSignal.any([active.signal, AbortSignal.timeout(15000)]),
        });
        if (disposed || !acceptPolling.current) return;
        setJob(data);
        if (data.preview_ready && ["running", "complete"].includes(data.state))
          setFrame(`/api/test-frame?t=${Date.now()}`);
        else setFrame("");
        if (data.state === "error") setError(data.message);
      } catch (err) {
        if (!disposed) {
          setError(err.message);
          setJob(null);
          setFrame("");
        }
      } finally {
        if (!disposed) timer = setTimeout(poll, 700);
      }
    };
    poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
      active?.abort();
    };
  }, [connected]);
  const selectFile = (event) => {
    acceptPolling.current = false;
    URL.revokeObjectURL(previewRef.current);
    previewRef.current = "";
    setPreview("");
    setError("");
    setJob(null);
    setFrame("");
    const next = event.target.files[0];
    setFile(null);
    if (!next) return;
    if (!/\.(jpe?g|png|mp4|avi|mov|mkv)$/i.test(next.name)) {
      setError("Choose a JPEG, PNG, MP4, AVI, MOV or MKV file.");
      event.target.value = "";
      return;
    }
    if (next.size > 250 * 1024 * 1024) {
      setError("Choose a file smaller than 250 MB.");
      event.target.value = "";
      return;
    }
    setFile(next);
    previewRef.current = URL.createObjectURL(next);
    setPreview(previewRef.current);
  };
  const run = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Choose an image or video first.");
      input.current?.focus();
      return;
    }
    if (!connected) {
      setError(
        "The application service is unavailable. Restart the application and try again.",
      );
      return;
    }
    setBusy(true);
    setUploading(true);
    setError("");
    setJob(null);
    setFrame("");
    removing.current = false;
    controller.current = new AbortController();
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await request(
        `/api/test/${view}`,
        { method: "POST", body, signal: controller.current.signal },
      );
      if (!removing.current) {
        acceptPolling.current = true;
        setJob(result);
      }
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message);
    } finally {
      if (!removing.current) setBusy(false);
      setUploading(false);
      controller.current = null;
    }
  };
  const remove = async () => {
    removing.current = true;
    setUploading(false);
    controller.current?.abort();
    setBusy(true);
    try {
      if (connected) await request("/api/test", { method: "DELETE" });
      acceptPolling.current = true;
      setFile(null);
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
      setPreview("");
      setFrame("");
      setJob(null);
      setError("");
      if (input.current) input.current.value = "";
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="operator-page">
      <PageHeader title="Test media">
        Check an image or video with upper-body, shoe, or combined detection.
      </PageHeader>
      {!connected && (
        <div className="ui-notice">
          Frontend preview. Uploaded media stays in this browser until you
          connect and run detection.
        </div>
      )}
      <div className="test-layout">
        <form
          noValidate
          className="operator-card"
          onSubmit={run}
          aria-busy={busy}
        >
          <div className="section-title">
            <Upload size={22} />
            <h2>Choose your media</h2>
          </div>
          <label className="upload-zone" htmlFor="test-file">
            <ImagePlay size={36} />
            <strong>{file ? file.name : "Choose an image or video"}</strong>
            <span>JPEG / PNG / video · Up to 250 MB</span>
            <input
              ref={input}
              id="test-file"
              type="file"
              accept=".jpg,.jpeg,.png,.mp4,.avi,.mov,.mkv"
              onChange={selectFile}
              disabled={busy || job?.state === "running"}
              aria-describedby="media-help media-error"
            />
          </label>
          <p id="media-help" className="field-help">
            Test files are temporary. Download a completed export
            before starting another test or closing the service.
          </p>
          <label className="form-field">
            Detection mode
            <select
              aria-label="Detection mode"
              value={view}
              onChange={(e) => setView(e.target.value)}
              disabled={busy || job?.state === "running"}
            >
              <option value="both">Both — goggles, wrists & shoes</option>
              <option value="upper">Upper body — goggles & wrists</option>
              <option value="lower">Lower body — safety shoes</option>
            </select>
          </label>
          <p className="field-help">
            Processed videos are ready to play and download automatically as
            annotated MP4 files without audio. Images do not produce a video.
          </p>
          <p className="inline-error" id="media-error" role="alert">
            {error}
          </p>
          <div className="button-row">
            <button
              className="ui-button primary"
              disabled={busy || job?.state === "running"}
              type="submit"
            >
              {busy
                ? "Uploading…"
                : job?.state === "running"
                  ? "Processing…"
                  : "Run detection"}
            </button>
            <button
              className="ui-button"
              type="button"
              onClick={remove}
              disabled={busy && !uploading}
            >
              Stop & remove
            </button>
          </div>
        </form>
        <article className="operator-card test-result">
          <div className="section-title">
            <Video size={22} />
            <h2>Result preview</h2>
          </div>
          <div className="test-canvas">
            {job?.download_ready ? (
              <video
                src={`/api/test/${encodeURIComponent(job.id)}/video`}
                aria-label="Processed detection video"
                controls
                playsInline
                preload="metadata"
                onError={() => setError("Video playback failed. Try downloading the detection video below.")}
              />
            ) : frame ? (
              <img src={frame} alt="Annotated test result" />
            ) : file && preview ? (
              /\.(jpe?g|png)$/i.test(file.name) ? (
                <img src={preview} alt="Selected image — unprocessed" />
              ) : (
                <video src={preview} controls muted playsInline />
              )
            ) : (
              <div className="feed-empty">
                <ImagePlay size={42} />
                <strong>Your result appears here</strong>
                <span>Choose a file and run detection to see annotations.</span>
              </div>
            )}
          </div>
          <div className="result-caption" aria-live="polite">
            <strong>
              {job?.message ||
                (file
                  ? "Source preview · No detection applied"
                  : "No test running")}
            </strong>
            {job?.frames && (
              <span>
                {job.frames} frames · {job.inference_ms} ms inference
              </span>
            )}
          </div>
          {job?.download_ready && (
            <a
              className="ui-button primary"
              href={`/api/test/${encodeURIComponent(job.id)}/download`}
              download="ppe-detection.mp4"
            >
              <Download size={18} />
              Download detection video
            </a>
          )}
        </article>
      </div>
    </section>
  );
}

function DonutChart({ percentage, size = 88, strokeWidth = 10, color = "#EF4444" }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = Math.max(0, Math.min(100, Number(percentage) || 0));
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="donut-chart-wrap" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#E2E8F0"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={pct > 0 ? color : "#10B981"}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="donut-center-text">
        <strong>{pct.toFixed(1)}%</strong>
      </div>
    </div>
  );
}

function ViolationPieChart({ items = [], totalViolations = 0, size = 210 }) {
  const activeItems = items.filter((i) => (i.count || 0) > 0);

  if (!activeItems || totalViolations === 0 || activeItems.length === 0) {
    return (
      <div className="pie-zero-state">
        <div className="pie-zero-circle">
          <CheckCircle2 size={36} color="#10B981" />
          <strong>100% Compliant</strong>
          <span>Zero Violations</span>
        </div>
      </div>
    );
  }

  const center = size / 2;
  const radius = size * 0.42;
  let accumulatedAngle = -Math.PI / 2;

  const slices = activeItems.map((item) => {
    const sliceAngle = (item.count / totalViolations) * 2 * Math.PI;
    const startAngle = accumulatedAngle;
    const endAngle = accumulatedAngle + sliceAngle;
    accumulatedAngle += sliceAngle;

    const x1 = center + radius * Math.cos(startAngle);
    const y1 = center + radius * Math.sin(startAngle);
    const x2 = center + radius * Math.cos(endAngle);
    const y2 = center + radius * Math.sin(endAngle);

    const largeArcFlag = sliceAngle > Math.PI ? 1 : 0;
    const pathData = activeItems.length === 1
      ? `M ${center} ${center - radius} A ${radius} ${radius} 0 1 1 ${center - 0.001} ${center - radius} Z`
      : `M ${center} ${center} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2} Z`;

    const midAngle = startAngle + sliceAngle / 2;
    const textRadius = radius * 0.66;
    const tx = center + textRadius * Math.cos(midAngle);
    const ty = center + textRadius * Math.sin(midAngle);

    return {
      ...item,
      pathData,
      tx,
      ty,
      percentage: ((item.count / totalViolations) * 100).toFixed(1),
    };
  });

  return (
    <div className="pie-chart-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {slices.map((s) => (
          <path
            key={s.key}
            d={s.pathData}
            fill={s.color}
            stroke="#ffffff"
            strokeWidth="2"
            className="pie-slice"
          >
            <title>{`${s.label}: ${s.count} (${s.percentage}%)`}</title>
          </path>
        ))}
        {slices.map((s) =>
          Number(s.percentage) >= 8 ? (
            <text
              key={`txt-${s.key}`}
              x={s.tx}
              y={s.ty}
              fill="#FFFFFF"
              fontSize="11"
              fontWeight="bold"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {s.percentage}%
            </text>
          ) : null
        )}
      </svg>
    </div>
  );
}

function HourlyTrendChart({ hourly = [], peakViolations = 0 }) {
  const shiftHours = hourly.filter((h) => {
    const hr = parseInt(h.hour, 10);
    return hr >= 8 && hr <= 17;
  });

  const displayHours = shiftHours.length > 0 ? shiftHours : hourly.slice(0, 10);
  const maxV = Math.max(...displayHours.map((h) => h.violations), 1);

  return (
    <div className="trend-chart-container">
      <div className="trend-y-axis">
        <span className="y-axis-title">Non-Compliant Detections</span>
        <div className="y-ticks">
          <span>{Math.max(10, maxV + 2)}</span>
          <span>{Math.round((maxV + 2) / 2)}</span>
          <span>0</span>
        </div>
      </div>

      <div className="trend-bars-area">
        {displayHours.map((item) => {
          const isPeak = item.violations === peakViolations && peakViolations > 0;
          const heightPct = Math.max(6, Math.round((item.violations / (maxV + 2)) * 100));

          return (
            <div key={item.hour} className={`trend-col ${isPeak ? "is-peak" : ""}`}>
              {isPeak && (
                <div className="peak-callout-badge">
                  <span>Highest Violation Period</span>
                  <div className="peak-arrow">↓</div>
                </div>
              )}
              <span className="bar-count-label">{item.violations}</span>
              <div
                className={`trend-bar-rect ${isPeak ? "peak-bar" : "normal-bar"}`}
                style={{ height: `${heightPct}%` }}
              />
              <span className="trend-x-label">{item.slot ? item.slot.replace("-", "-\n") : item.hour}</span>
            </div>
          );
        })}
      </div>
      <div className="trend-x-axis-title">Time (Hour)</div>
    </div>
  );
}

export function Dashboard({ connected, notify }) {
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().slice(0, 10));
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [emailConfig, setEmailConfig] = useState(null);
  const [showEmailConfig, setShowEmailConfig] = useState(false);
  const [sendingEmail, setSendingEmail] = useState(false);
  const [savingEmail, setSavingEmail] = useState(false);
  const [emailStatus, setEmailStatus] = useState("");

  // Load available dates
  useEffect(() => {
    let active = true;
    request("/api/dashboard/dates")
      .then((data) => {
        if (!active) return;
        if (data.dates && data.dates.length > 0) {
          setDates(data.dates);
          if (!data.dates.includes(selectedDate)) {
            setSelectedDate(data.dates[0]);
          }
        }
      })
      .catch((err) => console.error("Could not load dates:", err));
    return () => {
      active = false;
    };
  }, []);

  // Fetch summary data whenever selectedDate changes
  const loadSummary = (date) => {
    setLoading(true);
    request(`/api/dashboard/summary?date=${encodeURIComponent(date)}`)
      .then((data) => {
        setSummary(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to fetch dashboard summary:", err);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadSummary(selectedDate);
    // Auto-refresh today's data periodically if viewing today
    const today = new Date().toISOString().slice(0, 10);
    if (selectedDate === today) {
      const interval = setInterval(() => {
        request(`/api/dashboard/summary?date=${encodeURIComponent(today)}`)
          .then((data) => setSummary(data))
          .catch(() => {});
      }, 3500);
      return () => clearInterval(interval);
    }
  }, [selectedDate]);

  // Load email config
  useEffect(() => {
    request("/api/reports/email-config")
      .then((cfg) => setEmailConfig(cfg))
      .catch(() => {});
  }, []);

  const handleSendEmail = async () => {
    setSendingEmail(true);
    setEmailStatus("Generating and dispatching report...");
    try {
      const res = await request("/api/reports/send-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date: selectedDate }),
      });
      setEmailStatus(res.message || "Report sent successfully!");
      if (notify) notify(res.message || "Email delivered successfully.");
      // Refresh email config to update last_sent
      const updated = await request("/api/reports/email-config");
      setEmailConfig(updated);
    } catch (err) {
      setEmailStatus(`Delivery error: ${err.message}`);
    } finally {
      setSendingEmail(false);
    }
  };

  const handleSaveEmailConfig = async (e) => {
    e.preventDefault();
    setSavingEmail(true);
    try {
      const updated = await request("/api/reports/email-config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(emailConfig),
      });
      setEmailConfig(updated);
      setEmailStatus(`Settings saved! ${updated.last_status || "Schedule active."}`);
      if (notify) notify("Email delivery settings updated.");
    } catch (err) {
      setEmailStatus(`Failed to save: ${err.message}`);
    } finally {
      setSavingEmail(false);
    }
  };

  const todayStr = new Date().toISOString().slice(0, 10);
  const isToday = selectedDate === todayStr;

  return (
    <section className="operator-page">
      <PageHeader title="Daily Safety Compliance Dashboard">
        Real-time inspection telemetry, daily PPE violation breakdown, and automated daily shift audit reports.
      </PageHeader>

      <div className="ui-notice">
        <strong>AUTOMATED AUDIT SCHEDULER ACTIVE</strong> · Reports are automatically generated and emailed daily at <strong>{emailConfig?.schedule_time || "05:30"}</strong> ({emailConfig?.last_status || "Armed"}). Real detection events from live monitoring are stored and aggregated continuously.
      </div>

      {/* Date selector & Actions Bar */}
      <div className="dashboard-toolbar">
        <div className="dashboard-date-group">
          <Calendar size={20} color="var(--color-blue-primary)" />
          <label htmlFor="dashboard-date-select" className="sr-only">Select Audit Date</label>
          <select
            id="dashboard-date-select"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            aria-label="Select date"
          >
            {dates.map((d) => (
              <option key={d} value={d}>
                {d === todayStr ? `Today (${d})` : d}
              </option>
            ))}
          </select>
          <button
            className="ui-button compact"
            type="button"
            title="Refresh statistics"
            onClick={() => loadSummary(selectedDate)}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? "spin" : ""} />
            Refresh
          </button>
        </div>

        <div className="dashboard-actions">
          <a
            className="ui-button primary"
            href={`/api/reports/daily/pdf?date=${encodeURIComponent(selectedDate)}`}
            download={`ppe-daily-report-${selectedDate}.pdf`}
            title="Download PDF report for the selected day"
          >
            <Download size={16} />
            Download PDF Report
          </a>

          <button
            className="ui-button"
            type="button"
            onClick={handleSendEmail}
            disabled={sendingEmail}
            title="Dispatch the selected daily PDF report to email recipients now"
          >
            <Send size={16} />
            {sendingEmail ? "Dispatching…" : "Send PDF via Email"}
          </button>

          <button
            className="ui-button"
            type="button"
            onClick={() => setShowEmailConfig((v) => !v)}
            title="Configure automated 5:30 AM delivery, recipients, and email provider"
          >
            <Mail size={16} />
            Email & 5:30 AM Settings
          </button>
        </div>
      </div>

      {/* Email feedback status if any */}
      {emailStatus && (
        <div
          className={`ui-notice ${emailStatus.includes("error") || emailStatus.includes("Failed") ? "error" : ""}`}
          role="status"
        >
          {emailStatus}
        </div>
      )}

      {/* Expandable Email & Daily Scheduler Configuration Panel */}
      {showEmailConfig && emailConfig && (
        <form className="operator-card email-panel" onSubmit={handleSaveEmailConfig}>
          <div className="section-title">
            <Mail size={22} />
            <h2>Automated Daily Email & PDF Delivery Settings</h2>
          </div>
          <p className="field-help">
            The background service compiles the daily shift report into a PDF and automatically delivers it at the scheduled time. Real-time inference violations and compliance stats are included.
          </p>

          <div className="two-columns">
            <label className="checkbox-row" style={{ marginTop: "12px" }}>
              <input
                type="checkbox"
                checked={emailConfig.enabled}
                onChange={(e) => setEmailConfig({ ...emailConfig, enabled: e.target.checked })}
              />
              <strong>Enable automated daily delivery</strong>
            </label>

            <label className="form-field">
              Daily Delivery Schedule Time
              <div style={{ display: "flex", gap: "8px", alignItems: "center", marginTop: "4px" }}>
                <input
                  type="time"
                  step="60"
                  value={emailConfig.schedule_time || "05:30"}
                  onChange={(e) => setEmailConfig({ ...emailConfig, schedule_time: e.target.value })}
                  style={{ maxWidth: "160px", padding: "6px 10px", fontSize: "14px" }}
                  required
                />
                <span className="field-help" style={{ margin: 0, fontWeight: 600, color: "var(--color-blue-primary)" }}>
                  {emailConfig.last_status || "Active"}
                </span>
              </div>
              <span className="field-help" style={{ marginTop: "4px" }}>
                Tip: If you select a time that has already passed today, it will schedule for tomorrow morning. To test delivery right now, pick 1–2 minutes ahead or click &ldquo;Test &amp; Send PDF Now&rdquo; below.
              </span>
            </label>
          </div>

          <label className="form-field">
            Recipient Emails (comma-separated)
            <input
              type="text"
              value={(emailConfig.recipients || []).join(", ")}
              onChange={(e) =>
                setEmailConfig({
                  ...emailConfig,
                  recipients: e.target.value.split(",").map((s) => s.trim()),
                })
              }
              placeholder="safety-officer@factory.internal, plant-manager@bekaert.com"
              required
            />
          </label>

          <label className="form-field">
            Email Delivery Service
            <select
              value={emailConfig.service || "smtp"}
              onChange={(e) => setEmailConfig({ ...emailConfig, service: e.target.value })}
            >
              <option value="smtp">SMTP (Recommended · 100% Free · Direct PDF Attachment)</option>
              <option value="emailjs">EmailJS (REST API Service)</option>
            </select>
          </label>

          {emailConfig.service === "smtp" ? (
            <div className="three-columns" style={{ marginTop: "14px" }}>
              <label className="form-field">
                SMTP Host
                <input
                  type="text"
                  value={emailConfig.smtp_host || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, smtp_host: e.target.value })}
                  placeholder="smtp.gmail.com"
                />
              </label>
              <label className="form-field">
                SMTP Port
                <input
                  type="number"
                  value={emailConfig.smtp_port || 587}
                  onChange={(e) => setEmailConfig({ ...emailConfig, smtp_port: Number(e.target.value) })}
                  placeholder="587"
                />
              </label>
              <label className="form-field">
                Username / Email
                <input
                  type="text"
                  value={emailConfig.smtp_user || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, smtp_user: e.target.value })}
                  placeholder="your-email@gmail.com"
                />
              </label>
              <label className="form-field" style={{ gridColumn: "span 2" }}>
                App Password / Password
                <input
                  type="password"
                  value={emailConfig.smtp_pass || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, smtp_pass: e.target.value })}
                  placeholder={emailConfig.has_smtp_pass ? "••••••••" : "Gmail App Password / SMTP password"}
                />
              </label>
            </div>
          ) : (
            <div className="two-columns" style={{ marginTop: "14px" }}>
              <label className="form-field">
                EmailJS Service ID
                <input
                  type="text"
                  value={emailConfig.emailjs_service_id || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, emailjs_service_id: e.target.value })}
                  placeholder="service_xxx"
                />
              </label>
              <label className="form-field">
                EmailJS Template ID
                <input
                  type="text"
                  value={emailConfig.emailjs_template_id || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, emailjs_template_id: e.target.value })}
                  placeholder="template_xxx"
                />
              </label>
              <label className="form-field">
                EmailJS Public Key
                <input
                  type="text"
                  value={emailConfig.emailjs_public_key || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, emailjs_public_key: e.target.value })}
                  placeholder="Public Key"
                />
              </label>
              <label className="form-field">
                EmailJS Private Key (Access Token)
                <input
                  type="password"
                  value={emailConfig.emailjs_private_key || ""}
                  onChange={(e) => setEmailConfig({ ...emailConfig, emailjs_private_key: e.target.value })}
                  placeholder="Private Key (for server-side PDF attachments)"
                />
              </label>
            </div>
          )}

          <div className="button-row" style={{ marginTop: "16px" }}>
            <button className="ui-button primary" type="submit" disabled={savingEmail}>
              <Save size={16} />
              {savingEmail ? "Saving…" : "Save Email Settings"}
            </button>
            <button
              className="ui-button"
              type="button"
              onClick={handleSendEmail}
              disabled={sendingEmail}
            >
              <Send size={16} />
              Test & Send PDF Now
            </button>
          </div>
          {emailConfig.last_status && (
            <p className="field-help" style={{ marginTop: "10px" }}>
              Status: <strong>{emailConfig.last_status}</strong>
            </p>
          )}
        </form>
      )}

      {/* 1. OVERALL SUMMARY */}
      <div className="section-title" style={{ marginTop: "12px", marginBottom: "8px" }}>
        <h2>1. Overall Summary</h2>
      </div>

      <div className="dashboard-metrics">
        <article className="operator-card">
          <span className="metric-icon-box blue">
            <Users size={22} />
          </span>
          <span className="metric-label">Total Detections</span>
          <strong className="dashboard-number" style={{ color: "#0B2545" }}>
            {summary?.total_inspections ?? 0}
          </strong>
        </article>

        <article className="operator-card" style={{ background: "#F0FDF4", borderColor: "#BBF7D0" }}>
          <span className="metric-icon-box green">
            <CheckCircle2 size={22} />
          </span>
          <span className="metric-label" style={{ color: "#166534" }}>Compliant Detections</span>
          <strong className="dashboard-number" style={{ color: "#16A34A" }}>
            {summary?.compliant_count ?? 0}
          </strong>
        </article>

        <article className="operator-card" style={{ background: "#FEF2F2", borderColor: "#FECACA" }}>
          <span className="metric-icon-box red">
            <AlertTriangle size={22} />
          </span>
          <span className="metric-label" style={{ color: "#991B1B" }}>Non-Compliant Detections</span>
          <strong className="dashboard-number" style={{ color: "#DC2626" }}>
            {summary?.violation_count ?? 0}
          </strong>
        </article>

        <article className="operator-card donut-card" style={{ background: "#FFFBEB", borderColor: "#FDE68A" }}>
          <span className="metric-label" style={{ color: "#92400E", marginBottom: "4px" }}>Violation Rate</span>
          <DonutChart percentage={summary?.violation_rate ?? 0.0} />
        </article>
      </div>

      {/* 2 & 3: VIOLATIONS BY TYPE & HOURLY NON-COMPLIANCE TREND (SIDE BY SIDE) */}
      <div className="dashboard-charts-grid">
        {/* Section 2: Violations by Type Pie Chart */}
        <article className="operator-card">
          <div className="section-title">
            <AlertTriangle size={20} color="#DC2626" />
            <h2>2. Violations by Type (Non-Compliance)</h2>
          </div>
          <p className="field-help">
            Proportional breakdown of observed safety violations by PPE category.
          </p>

          <div className="pie-container">
            <ViolationPieChart
              items={summary?.violations_by_type || []}
              totalViolations={summary?.violation_count || 0}
            />

            <table className="pie-legend-table">
              <tbody>
                {(summary?.violations_by_type || []).map((v) => (
                  <tr key={v.key}>
                    <td>
                      <span className="pie-legend-label">
                        <span className="legend-dot" style={{ backgroundColor: v.color }} />
                        {v.label}
                      </span>
                    </td>
                    <td className="pie-legend-val">
                      {v.count} ({v.percentage}%)
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        {/* Section 3: Hourly Non-Compliance Trend Bar Chart */}
        <article className="operator-card">
          <div className="section-title">
            <Clock size={20} color="var(--color-blue-primary)" />
            <h2>3. Hourly Non-Compliance Trend</h2>
          </div>
          <p className="field-help">
            Hourly violation frequency across shift duration with peak hour highlight.
          </p>

          <HourlyTrendChart
            hourly={summary?.hourly || []}
            peakViolations={summary?.peak_violations || 0}
          />
        </article>
      </div>

      {/* Equipment Compliance Breakdown */}
      <article className="operator-card">
        <div className="section-title">
          <ShieldCheck size={22} />
          <h2>Equipment Compliance Breakdown ({selectedDate})</h2>
        </div>
        <p className="field-help">
          Real detection statistics compiled from camera evaluations on this date.
        </p>

        {summary ? (
          <div className="sample-bars">
            {[
              [
                "Safety Goggles",
                summary.categories?.goggles?.rate ?? 100,
                summary.categories?.goggles?.compliant ?? 0,
                summary.categories?.goggles?.violations ?? 0,
              ],
              [
                "Wrist Protection (Clear Wrists)",
                summary.categories?.wrist?.rate ?? 100,
                summary.categories?.wrist?.compliant ?? 0,
                summary.categories?.wrist?.violations ?? 0,
              ],
              [
                "Approved Safety Shoes",
                summary.categories?.shoes?.rate ?? 100,
                summary.categories?.shoes?.compliant ?? 0,
                summary.categories?.shoes?.violations ?? 0,
              ],
            ].map(([label, rate, comp, viol]) => (
              <div key={label}>
                <span>
                  <strong>{label}</strong>
                  <span>
                    <strong style={{ color: viol > 0 ? "var(--ui-danger)" : "var(--ui-action)" }}>
                      {rate}% compliant
                    </strong>{" "}
                    ({comp} compliant · {viol} violations)
                  </span>
                </span>
                <meter min="0" max="100" value={rate}>
                  {rate}%
                </meter>
              </div>
            ))}
          </div>
        ) : (
          <p className="field-help">Loading equipment metrics…</p>
        )}
      </article>

      {/* Recent Violation Incidents Log */}
      <article className="operator-card">
        <div className="section-title">
          <AlertTriangle size={22} />
          <h2>Safety Violations Recorded ({selectedDate})</h2>
        </div>
        <p className="field-help">
          Timestamped inspection records flagging non-compliant PPE on the selected date.
        </p>

        {summary?.recent_violations && summary.recent_violations.length > 0 ? (
          <div className="violations-table-wrap">
            <table className="violations-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Camera View</th>
                  <th>Worker</th>
                  <th>Violation Observed</th>
                  <th>Evidence Snapshot</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {summary.recent_violations.map((v) => (
                  <tr key={v.id}>
                    <td><strong>{v.time}</strong></td>
                    <td style={{ textTransform: "capitalize" }}>{v.view} View</td>
                    <td>
                      {v.worker_id != null ? (
                        <span className="badge-pill" style={{ background: "#E0F2FE", color: "#0369A1", borderColor: "#BAE6FD" }}>
                          Worker #{v.worker_id}
                        </span>
                      ) : (
                        <span style={{ color: "#94A3B8" }}>—</span>
                      )}
                    </td>
                    <td>
                      <span className="badge-pill danger">
                        <XCircle size={12} />
                        {(v.violations || []).map((x) => x.replace("no_goggles", "No Goggles").replace("not_empty_wrist", "Wrist Accessory").replace("NOT_SAFETY_SHOE", "Other Shoe")).join(", ") || "Violation"}
                      </span>
                    </td>
                    <td>
                      {v.image_filename ? (
                        <a
                          href={`/api/violations/images/${encodeURIComponent(v.image_filename)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ui-button compact"
                          style={{
                            padding: "3px 8px",
                            fontSize: "12px",
                            textDecoration: "none",
                            display: "inline-flex",
                            gap: "5px",
                            alignItems: "center",
                          }}
                          title="Open captured violation evidence frame in high-res"
                        >
                          <Camera size={13} />
                          View Snapshot
                        </a>
                      ) : (
                        <span style={{ color: "#94A3B8" }}>—</span>
                      )}
                    </td>
                    <td>
                      <span className="badge-pill danger">Action Required</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="field-help" style={{ padding: "16px 0" }}>
            No PPE violations logged on {selectedDate}. All evaluated personnel met plant safety requirements.
          </p>
        )}
      </article>
    </section>
  );
}

const controlLabels = {
  brightness: "Brightness",
  contrast: "Contrast",
  saturation: "Saturation",
  exposure: "Exposure",
  auto_exposure: "Auto exposure",
  focus: "Focus",
  autofocus: "Auto focus",
  gain: "Gain",
};
const automaticControl = (name) => name === "auto_exposure" || name === "autofocus";
const cameraValue = (value) =>
  value === null
    ? "off"
    : typeof value === "number"
      ? `index:${value}`
      : `device:${value}`;
const parseCamera = (value) =>
  value === "off"
    ? null
    : value.startsWith("index:")
      ? Number(value.slice(6))
      : value.slice(7);
const newProfile = () => ({ width: 1920, height: 1080, fps: 30, controls: {} });
export function SettingsPage({
  connected,
  settings,
  onSaved,
  catalog,
  status,
  onDirty,
  notify,
}) {
  const [draft, setDraft] = useState(() => structuredClone(settings)),
    [selected, setSelected] = useState("upper"),
    [busy, setBusy] = useState(false),
    [capabilityResult, setCapabilityResult] = useState(null),
    [capabilityRevision, setCapabilityRevision] = useState(0),
    [error, setError] = useState("");
  const dirty = JSON.stringify(draft) !== JSON.stringify(settings);
  useEffect(() => onDirty(dirty), [dirty, onDirty]);
  const update = (change) =>
    setDraft((prev) => {
      const next = structuredClone(prev);
      change(next);
      return next;
    });
  const camera = draft[selected].camera,
    key = String(camera),
    profile = draft.profiles[key] || newProfile();
  const hardware = Object.values(status?.views || {}).find(
    (v) => v.camera === camera,
  );
  const capabilities = capabilityResult?.camera === camera ? capabilityResult.data : null;
  const capabilityLoading = connected && camera !== null &&
    (capabilityResult?.camera !== camera || capabilityResult?.revision !== capabilityRevision);
  const capabilityError = capabilityResult?.camera === camera ? capabilityResult.error : "";
  useEffect(() => {
    if (!connected || camera === null) return;
    const controller = new AbortController();
    request(`/api/camera-capabilities?camera=${encodeURIComponent(camera)}`, {
      signal: AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]),
    }).then(data => setCapabilityResult({ camera, data, revision: capabilityRevision }))
      .catch(err => {
        if (!controller.signal.aborted)
          setCapabilityResult({ camera, data: null, error: err.message, revision: capabilityRevision });
      });
    return () => controller.abort();
  }, [camera, connected, status?.running, capabilityRevision]);
  const currentControl = (name) => {
    const spec = capabilities?.controls?.[name];
    return spec && spec.current != null && spec.current >= spec.min && spec.current <= spec.max
      ? spec.current : spec?.default;
  };
  const setProfile = (prop, value) =>
    update((next) => {
      next.profiles[key] = {
        ...(next.profiles[key] || newProfile()),
        [prop]: value,
      };
    });
  const setControl = (name, enabled, value = 0) => {
    const controls = { ...profile.controls };
    if (enabled) {
      controls[name] = value;
      const automatic = { exposure: "auto_exposure", focus: "autofocus" }[name];
      if (automatic && capabilities?.controls?.[automatic]?.options.includes(0))
        controls[automatic] = 0;
      if (automaticControl(name) && value === 1)
        delete controls[name === "auto_exposure" ? "exposure" : "focus"];
    }
    else delete controls[name];
    setProfile("controls", controls);
  };
  const reset = () => {
    if (!capabilities?.available || !Object.keys(capabilities.controls).length) {
      setError(
        "Stop detection and reopen Settings to read this camera's supported defaults.",
      );
      return;
    }
    setError("");
    update(next => {
      next.profiles[key] = { ...profile, device_id: capabilities.device_id,
        controls: Object.fromEntries(Object.entries(capabilities.controls).map(([name, spec]) => [name, spec.default])) };
    });
    notify(
      "Camera defaults restored in the draft. Save settings, then start detection to apply.",
    );
  };
  const save = async (event) => {
    event.preventDefault();
    if (!validateSettings(event.currentTarget, setError)) return;
    setBusy(true);
    setError("");
    try {
      if (!connected)
        throw new Error(
          "The application service is unavailable. Settings have not been saved.",
        );
      const result = await request("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      setDraft(structuredClone(result.settings));
      onSaved(structuredClone(result.settings));
      onDirty(false);
      notify("Settings saved. Start detection to apply camera settings.");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  const known = [...catalog];
  for (const view of ["upper", "lower"])
    if (
      typeof draft[view].camera === "string" &&
      !known.some((c) => c.id === draft[view].camera)
    )
      known.push({
        id: draft[view].camera,
        name: "Saved camera (not currently listed)",
      });
  return (
    <section className="operator-page">
      <PageHeader title="Settings">
        Assign camera views and save the picture settings used when each device
        connects.
      </PageHeader>
      {!connected && (
        <div className="ui-notice">
          Camera settings are unavailable while the application reconnects. Your
          unsaved edits are kept.
        </div>
      )}
      <article className="operator-card inference-panel" aria-labelledby="inference-title">
        <div className="section-title">
          <Video size={22} aria-hidden="true" />
          <h2 id="inference-title">Inference & hardware</h2>
        </div>
        <p className="inference-device">{connected ? status?.gpu?.name || "GPU unavailable" : "Service unavailable"}</p>
        <p className="field-help">
          {connected && status?.gpu?.ready ? "CUDA inference ready" : status?.gpu?.error || "Connect to see hardware and inference status."}
        </p>
        <div className="two-columns">
          {["upper", "lower"].map(view => {
            const engine = connected ? status?.views?.[view] : null;
            return (
              <div key={view}>
                <h3>{view === "upper" ? "Upper-body model" : "Footwear model"}</h3>
                <dl className="inference-details">
                  <div><dt>Model</dt><dd>{engine?.model_ready ? "Ready" : "Unavailable"}</dd></div>
                  <div><dt>Inference</dt><dd>{engine?.inference_ms != null ? `${engine.inference_ms} ms` : "—"}</dd></div>
                  <div><dt>Processing rate</dt><dd>{engine?.processing_fps != null ? `${engine.processing_fps} FPS` : "—"}</dd></div>
                </dl>
                {engine?.error && <p className="inline-error">{engine.error}</p>}
              </div>
            );
          })}
        </div>
      </article>
      <form noValidate onSubmit={save} aria-busy={busy}>
        <article className="operator-card">
          <div className="section-title">
            <Video size={22} />
            <h2>Camera assignments</h2>
          </div>
          <p className="field-help">
            Select separate cameras, use the same camera for both views, or
            disable either view.
          </p>
          <div className="two-columns">
            {["upper", "lower"].map((view) => (
              <label className="form-field" key={view}>
                {view === "upper"
                  ? "Upper body — goggles & wrists"
                  : "Lower body — safety shoes"}
                <select
                  aria-label={
                    view === "upper"
                      ? "Upper body — goggles & wrists"
                      : "Lower body — safety shoes"
                  }
                  value={cameraValue(draft[view].camera)}
                  onChange={(e) =>
                    update((next) => {
                      next[view].camera = parseCamera(e.target.value);
                    })
                  }
                >
                  <option value="off">Disabled</option>
                  {known.map((c) => (
                    <option key={c.id ?? c.index} value={c.id ? `device:${c.id}` : `index:${c.index}`}>
                      {c.name}
                      {c.index !== undefined ? ` · USB ${c.index}` : ""}
                    </option>
                  ))}
                  {Array.from({ length: 16 }, (_, i) => (
                    <option key={i} value={`index:${i}`}>
                      Manual camera index {i}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          {draft.upper.camera !== null &&
            draft.upper.camera === draft.lower.camera && (
              <p className="ui-notice">
                Both views share one camera. The device opens once; each view
                applies its own detection.
              </p>
            )}
        </article>
        <article className="operator-card">
          <div className="section-title">
            <Video size={22} />
            <h2>Camera picture</h2>
          </div>
          <label className="form-field">
            Edit camera assigned to
            <select
              aria-label="Edit camera assigned to"
              value={selected}
              onChange={(e) => setSelected(e.target.value)}
            >
              <option value="upper">Upper-body view</option>
              <option value="lower">Lower-body view</option>
            </select>
          </label>
          <fieldset disabled={camera === null}>
            <legend className="sr-only">Capture profile</legend>
            <div className="three-columns">
              {[
                ["width", "Width", 320, 3840],
                ["height", "Height", 240, 2160],
                ["fps", "Frames per second", 5, 60],
              ].map(([prop, label, min, max]) => (
                <label className="form-field" key={prop}>
                  {label}
                  <input
                    type="number"
                    min={min}
                    max={max}
                    required
                    aria-label={label}
                    value={profile[prop]}
                    onChange={(e) =>
                      setProfile(
                        prop,
                        e.target.value === "" ? "" : Number(e.target.value),
                      )
                    }
                  />
                </label>
              ))}
            </div>
            <p className="field-help">
              Controls depend on the USB driver. Enable a custom value to
              request it. Unsupported values cannot be forced. A shared camera
              has one picture profile. Enabling manual exposure or focus turns
              its automatic mode off. Save, then start detection to apply.
            </p>
            <p className="field-help" role="status">
              {capabilityLoading ? "Reading camera capabilities…" : capabilityError || capabilities?.message || "Select a camera to read supported controls."}
            </p>
            <button className="ui-button" type="button"
              disabled={!connected || !!status?.running || capabilityLoading}
              onClick={() => setCapabilityRevision(value => value+1)}>
              Refresh camera controls
            </button>
            {profile.device_id && capabilities?.device_id && profile.device_id !== capabilities.device_id && (
              <p className="inline-error">The camera at this index has changed. Restore camera defaults before saving.</p>
            )}
            {Object.keys(profile.controls).some(name => capabilities && !capabilities.controls[name]) && (
              <p className="inline-error">Saved controls are unsupported by this camera. Restore camera defaults before saving.</p>
            )}
            <div className="camera-controls">
              {Object.entries(controlLabels).filter(([name]) => capabilities?.controls?.[name]).map(([name, label]) => (
                <div className="camera-control" key={name}>
                  <label className="checkbox-row">
                    <input
                      type="checkbox"
                      checked={name in profile.controls}
                      disabled={capabilityLoading}
                      onChange={(e) =>
                        setControl(
                          name,
                          e.target.checked,
                          currentControl(name) ?? (automaticControl(name) ? 1 : 0),
                        )
                      }
                    />
                    {label}
                  </label>
                  {automaticControl(name) ? (
                    <select
                      aria-label={`${label} custom value`}
                      disabled={!(name in profile.controls)}
                      value={profile.controls[name] ?? 1}
                      onChange={(e) => setControl(name, true, Number(e.target.value))}
                    >
                      {capabilities.controls[name].options.map(value => (
                        <option key={value} value={value}>{value ? "On" : "Off"}</option>
                      ))}
                    </select>
                  ) : <input
                    aria-label={`${label} custom value`}
                    type="number"
                    step={capabilities.controls[name].step}
                    min={capabilities.controls[name].min}
                    max={capabilities.controls[name].max}
                    disabled={!(name in profile.controls)}
                    value={profile.controls[name] ?? currentControl(name) ?? ""}
                    onChange={(e) =>
                      setControl(
                        name,
                        true,
                        e.target.value === "" ? "" : Number(e.target.value),
                      )
                    }
                    required
                  />}
                  <span className="field-help">
                    {!automaticControl(name) && `Range: ${capabilities.controls[name].min}–${capabilities.controls[name].max} · Step: ${capabilities.controls[name].step} · `}
                    {hardware?.controls?.[name]
                      ? `Read-back: ${hardware.controls[name].value ?? "Unavailable"}${hardware.controls[name].accepted === false ? " · Request rejected" : ""}`
                      : `Current: ${currentControl(name) ?? "Unavailable"}`}
                  </span>
                </div>
              ))}
            </div>
            <button className="ui-button" type="button" onClick={reset} disabled={!capabilities?.available || capabilityLoading}>
              <RotateCcw size={16} />
              Restore camera defaults
            </button>
            <p className="field-help">
              Uses this camera driver's reported defaults. Save settings and
              restart detection to apply supported controls.
            </p>
          </fieldset>
        </article>
        <article className="operator-card">
          <details>
            <summary>Advanced detection settings</summary>
            <p className="field-help">
              Thresholds require calibration with trained weights. SAHI
              increases wrist detail and processing time.
            </p>
            <div className="two-columns">
              {["upper", "lower"].map((view) => (
                <fieldset key={view}>
                  <legend>
                    {view === "upper" ? "Upper body" : "Lower body"}
                  </legend>
                  <label className="form-field">
                    Confidence threshold
                    <input
                      type="number"
                      min="0.05"
                      max="0.95"
                      step="0.01"
                      value={draft[view].confidence}
                      onChange={(e) =>
                        update((next) => {
                          next[view].confidence =
                            e.target.value === "" ? "" : Number(e.target.value);
                        })
                      }
                      required
                    />
                  </label>
                  <label className="form-field">
                    Inference image size
                    <select
                      value={draft[view].imgsz}
                      onChange={(e) =>
                        update((next) => {
                          next[view].imgsz = Number(e.target.value);
                        })
                      }
                    >
                      {[...new Set([640, 960, 1280, draft[view].imgsz])]
                        .sort((a, b) => a - b)
                        .map((n) => (
                          <option key={n}>{n}</option>
                        ))}
                    </select>
                  </label>
                </fieldset>
              ))}
            </div>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={draft.upper.sahi_wrist}
                onChange={(e) =>
                  update((next) => {
                    next.upper.sahi_wrist = e.target.checked;
                  })
                }
              />
              Use SAHI slicing for wrist detection
            </label>
            <label className="form-field">
              Wrist slice size
              <select
                disabled={!draft.upper.sahi_wrist}
                value={draft.upper.sahi_slice}
                onChange={(e) =>
                  update((next) => {
                    next.upper.sahi_slice = Number(e.target.value);
                  })
                }
              >
                {[...new Set([640, 960, 1280, draft.upper.sahi_slice])]
                  .sort((a, b) => a - b)
                  .map((n) => (
                    <option key={n}>{n}</option>
                  ))}
              </select>
            </label>
          </details>
        </article>
        <div className="settings-save">
          <div>
            <strong>{dirty ? "Unsaved changes" : "All changes saved"}</strong>
            <span>
              {connected
                ? "Saved camera settings apply when detection starts."
                : "Waiting for the application service"}
            </span>
          </div>
          <button
            className="ui-button primary"
            disabled={busy || !connected || !!status?.running}
            type="submit"
          >
            <Save size={18} />
            {busy ? "Saving…" : "Save settings"}
          </button>
        </div>
        {status?.running && (
          <p className="ui-notice">
            Stop detection from Live before saving camera settings.
          </p>
        )}
        <p className="inline-error" id="settings-error" role="alert">
          {error}
        </p>
      </form>
    </section>
  );
}
