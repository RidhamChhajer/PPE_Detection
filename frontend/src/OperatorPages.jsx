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

export function Dashboard() {
  return (
    <section className="operator-page">
      <PageHeader title="Sample dashboard">
        An illustrative overview for presentation. These figures are synthetic.
      </PageHeader>
      <div className="ui-notice">
        SAMPLE DATA · This dashboard does not count workers or retain inspection
        records.
      </div>
      <div className="dashboard-metrics">
        {[
          [Users, "Personnel", 342, "blue"],
          [ShieldCheck, "Compliant", 298, "green"],
          [AlertTriangle, "Needs attention", 44, "red"],
          [Video, "Cameras", 2, "blue"],
        ].map(([Icon, label, value, tone]) => (
          <article className="operator-card" key={label}>
            <span className={`metric-icon-box ${tone}`}>
              <Icon size={22} />
            </span>
            <span className="metric-label">Sample {label.toLowerCase()}</span>
            <strong className="dashboard-number">{value}</strong>
          </article>
        ))}
      </div>
      <article className="operator-card">
        <div className="section-title">
          <ShieldCheck size={22} />
          <h2>Sample PPE observations</h2>
        </div>
        <p className="field-help">
          Example rates for the display design; unrelated to model accuracy.
        </p>
        <div className="sample-bars">
          {[
            ["Safety goggles", 87],
            ["Clear wrists", 93],
            ["Approved safety shoes", 96],
          ].map(([label, value]) => (
            <div key={label}>
              <span>
                {label}
                <strong>{value}% sample</strong>
              </span>
              <meter min="0" max="100" value={value}>
                {value}% sample
              </meter>
            </div>
          ))}
        </div>
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
