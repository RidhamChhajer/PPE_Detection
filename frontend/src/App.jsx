import { useCallback, useEffect, useRef, useState } from "react";
import {
  Video,
  PanelLeftClose,
  PanelLeftOpen,
  ImagePlay,
  LayoutDashboard,
  Settings,
} from "lucide-react";
import { ShieldCheckLogo } from "./components/icons";
import Live from "./Live";
import { TestMedia, Dashboard, SettingsPage } from "./OperatorPages";
import { request, defaults } from "./api";
import "./App.css";
import "./Workspace.css";
const pages = {
  live: "Live",
  test: "Test media",
  dashboard: "Sample dashboard",
  settings: "Settings",
};
const route = () =>
  Object.hasOwn(pages, location.hash.slice(1))
    ? location.hash.slice(1)
    : "live";
export default function App() {
  const [page, setPage] = useState(route),
    [expanded, setExpanded] = useState(false),
    [connected, setConnected] = useState(false),
    [connecting, setConnecting] = useState(false),
    [status, setStatus] = useState(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState(defaults),
    [catalog, setCatalog] = useState([]),
    [dirty, setDirty] = useState(false),
    [toast, setToast] = useState("");
  const [settingsKey, setSettingsKey] = useState(0);
  const dialog = useRef(null),
    pending = useRef(null),
    previousFocus = useRef(null),
    main = useRef(null),
    toastTimer = useRef(null),
    connectController = useRef(null);
  const notify = useCallback((message) => {
    setToast(message);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 6000);
  }, []);
  const commit = useCallback((next) => {
    setPage(next);
    history.replaceState(null, "", `#${next}`);
  }, []);
  const navigate = useCallback(
    (next) => {
      if (next === page) return;
      if (dirty) {
        pending.current = next;
        previousFocus.current = document.activeElement;
        dialog.current.showModal();
        dialog.current.querySelector("button")?.focus();
        history.replaceState(null, "", `#${page}`);
      } else commit(next);
    },
    [dirty, page, commit],
  );
  useEffect(() => {
    const change = () => navigate(route());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, [navigate]);
  useEffect(() => {
    document.title = `${pages[page]} — AI Based PPE Monitoring & Warning`;
    main.current?.querySelector("h1")?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [page]);
  useEffect(() => {
    const unload = (e) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", unload);
    return () => window.removeEventListener("beforeunload", unload);
  }, [dirty]);
  useEffect(
    () => () => {
      clearTimeout(toastTimer.current);
      connectController.current?.abort();
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
        const result = await request("/api/status", {
          signal: AbortSignal.any([active.signal, AbortSignal.timeout(10000)]),
        });
        if (!disposed) {
          setStatus(result);
          setError("");
        }
      } catch {
        if (!disposed) {
          setStatus(null);
          setError(
            "The application service is unavailable. Reconnecting automatically.",
          );
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
  const connect = useCallback(async () => {
    setConnecting(true);
    setError("");
    const controller = new AbortController();
    connectController.current = controller;
    try {
      const signal = AbortSignal.any([
        controller.signal,
        AbortSignal.timeout(10000),
      ]);
      const [state, saved, devices] = await Promise.all([
        request("/api/status", { signal }),
        request("/api/settings", { signal }),
        request("/api/cameras", { signal }),
      ]);
      if (controller.signal.aborted || connectController.current !== controller)
        return;
      setStatus(state);
      setSettings(saved);
      setCatalog(devices);
      setConnected(true);
      setSettingsKey((k) => k + 1);
    } catch (err) {
      if (controller.signal.aborted || connectController.current !== controller)
        return;
      setError(
        `Could not connect. Start the local detection service on port 8876. ${err.message}`,
      );
      notify("The application service is unavailable. Your edits are kept.");
    } finally {
      if (connectController.current === controller) setConnecting(false);
    }
  }, [notify]);
  useEffect(() => {
    const startup = setTimeout(connect, 0);
    return () => {
      clearTimeout(startup);
      connectController.current?.abort();
    };
  }, [connect]);
  const toggle = async () => {
    setBusy(true);
    try {
      setStatus(
        await request(`/api/live/${status?.running ? "stop" : "start"}`, {
          method: "POST",
        }),
      );
    } catch (err) {
      notify(err.message);
    } finally {
      setBusy(false);
    }
  };
  const closeDialog = () => {
    dialog.current.close();
    pending.current = null;
    previousFocus.current?.focus();
  };
  return (
    <div className={`workspace ${expanded ? "nav-expanded" : ""}`}>
      <a
        className="skip-link"
        href="#main-content"
        onClick={(e) => {
          e.preventDefault();
          main.current?.focus();
        }}
      >
        Skip to content
      </a>
      <aside className="sidebar" aria-label="Application sidebar">
        <div className="sidebar-brand">
          <ShieldCheckLogo size={34} />
          {expanded && <strong>AI Based PPE Monitoring & Warning</strong>}
        </div>
        <button
          className="nav-toggle"
          aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? (
            <PanelLeftClose size={21} />
          ) : (
            <PanelLeftOpen size={21} />
          )}
        </button>
        <nav aria-label="Main navigation">
          {[
            [Video, "live"],
            [ImagePlay, "test"],
            [LayoutDashboard, "dashboard"],
            [Settings, "settings"],
          ].map(([Icon, id]) => (
            <a
              key={id}
              href={`#${id}`}
              aria-label={pages[id]}
              title={pages[id]}
              aria-current={page === id ? "page" : undefined}
              onClick={(e) => {
                e.preventDefault();
                navigate(id);
              }}
            >
              <Icon size={22} />
              {expanded && <span>{pages[id]}</span>}
            </a>
          ))}
        </nav>
      </aside>
      <main id="main-content" ref={main} tabIndex={-1}>
        {page === "live" ? (
          <Live
            connected={connected}
            connecting={connecting}
            onConnect={connect}
            status={status}
            catalog={catalog}
            error={error}
            busy={busy}
            onToggle={toggle}
          />
        ) : (
          <>
            {error && (
              <div
                className="page-connection-error ui-notice error"
                role="alert"
              >
                {error}
              </div>
            )}
            {page === "test" ? (
              <TestMedia
                key={String(connected)}
                connected={connected && !!status}
              />
            ) : page === "dashboard" ? (
              <Dashboard />
            ) : (
              <SettingsPage
                key={settingsKey}
                connected={connected && !!status}
                settings={settings}
                onSaved={setSettings}
                catalog={catalog}
                status={status}
                onDirty={setDirty}
                notify={notify}
              />
            )}
          </>
        )}
      </main>
      <div
        className="ui-toast"
        role="status"
        aria-live="polite"
        hidden={!toast}
      >
        {toast}
      </div>
      <dialog
        ref={dialog}
        className="unsaved-dialog"
        aria-labelledby="unsaved-title"
        onCancel={() => {
          pending.current = null;
          previousFocus.current?.focus();
        }}
      >
        <h2 id="unsaved-title">Keep your settings changes?</h2>
        <p>
          Your edits have not been saved. Stay to review them, or discard them
          and continue.
        </p>
        <div className="button-row">
          <button className="ui-button primary" onClick={closeDialog}>
            Stay in Settings
          </button>
          <button
            className="ui-button"
            onClick={() => {
              const next = pending.current;
              setDirty(false);
              setSettingsKey((k) => k + 1);
              closeDialog();
              commit(next);
            }}
          >
            Discard changes
          </button>
        </div>
      </dialog>
    </div>
  );
}
