import { useEffect, useRef, useState } from "react";
import { Video, Maximize, Minimize } from "lucide-react";

export default function LiveCamera({ view, running, message }) {
  const surface = useRef(null);
  const [phase, setPhase] = useState("waiting");
  const [expanded, setExpanded] = useState(false);
  const [expandError, setExpandError] = useState("");

  useEffect(() => {
    const changed = () => setExpanded(document.fullscreenElement === surface.current?.parentElement);
    document.addEventListener("fullscreenchange", changed);
    return () => document.removeEventListener("fullscreenchange", changed);
  }, []);

  const expand = async () => {
    setExpandError("");
    try {
      if (expanded) await document.exitFullscreen();
      else await surface.current.parentElement.requestFullscreen();
    } catch {
      setExpandError("Fullscreen is unavailable. Try opening the app in Edge or Chrome.");
    }
  };

  useEffect(() => {
    if (!running) return;
    const session = new AbortController();
    let timer;
    let previous;

    async function drawLatest() {
      const start = performance.now();
      let delay = 50;
      try {
        const response = await fetch(`/api/frame/${view}`, {
          cache: "no-store",
          signal: AbortSignal.any([session.signal, AbortSignal.timeout(1500)]),
        });
        if (response.status === 204) {
          setPhase("waiting");
          previous = undefined;
          delay = 250;
          return;
        }
        if (!response.ok) throw new Error("Preview request failed");
        const sequence = response.headers.get("X-Frame-Sequence");
        if (sequence && sequence === previous) return;
        const bitmap = await createImageBitmap(await response.blob());
        try {
          if (session.signal.aborted) return;
          const canvas = surface.current;
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
          }
          canvas.getContext("2d").drawImage(bitmap, 0, 0);
          canvas.dataset.frame = sequence || "";
          previous = sequence;
          setPhase("live");
        } finally {
          bitmap.close();
        }
      } catch {
        if (!session.signal.aborted) {
          previous = undefined;
          setPhase("retrying");
          delay = 500;
        }
      } finally {
        if (!session.signal.aborted) {
          timer = setTimeout(drawLatest, Math.max(0, delay - (performance.now() - start)));
        }
      }
    }

    drawLatest();
    return () => {
      session.abort();
      clearTimeout(timer);
    };
  }, [view, running]);

  const visible = running && phase === "live";
  return (
    <>
      <button
        type="button"
        className="camera-expand"
        aria-label={`${expanded ? "Exit fullscreen for" : "Expand"} ${view} camera`}
        aria-pressed={expanded}
        title={expanded ? "Exit fullscreen (Esc)" : "Expand camera"}
        onClick={expand}
      >
        {expanded ? <Minimize size={19} aria-hidden="true" /> : <Maximize size={19} aria-hidden="true" />}
      </button>
      {expandError && <p className="camera-expand-error" role="status">{expandError}</p>}
      <canvas
        ref={surface}
        role="img"
        aria-label={`${view} live camera with detections`}
        data-ready={visible}
        hidden={!visible}
      />
      {!visible && (
        <div className="feed-empty" role="status">
          <Video size={32} aria-hidden="true" />
          <strong>
            {!running ? "Monitoring stopped" : phase === "retrying" ? "Preview reconnecting" : "Waiting for camera"}
          </strong>
          <span>
            {!running
              ? "Start detection to open the selected camera."
              : phase === "retrying"
                ? "No preview response. Retrying automatically…"
                : message || "No current camera frames. Check the selected device and camera access."}
          </span>
        </div>
      )}
    </>
  );
}
