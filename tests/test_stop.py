import urllib.error

import pytest

from tools import stop


def test_stop_releases_active_work_before_terminating_service(monkeypatch):
    calls = []

    def request(path, method="GET"):
        calls.append((path, method))
        return {"application": "ppe-guard"} if path == "/api/health" else {"state": "running"}

    monkeypatch.setattr(stop, "request", request)
    monkeypatch.setattr(stop.subprocess, "run", lambda *a, **k: calls.append("terminate"))
    stop.main()
    assert calls == [
        ("/api/health", "GET"),
        ("/api/test", "GET"),
        ("/api/test", "DELETE"),
        ("/api/live/stop", "POST"),
        "terminate",
    ]


@pytest.mark.parametrize("unreachable", [True, False])
def test_stop_never_terminates_an_unverified_service(monkeypatch, unreachable):
    def request(*_):
        if unreachable:
            raise urllib.error.URLError("not running")
        return {"application": "something-else"}

    monkeypatch.setattr(stop, "request", request)
    monkeypatch.setattr(
        stop.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Unsafe kill"))
    )
    if unreachable:
        stop.main()
    else:
        with pytest.raises(RuntimeError, match="not PPE Detection System"):
            stop.main()


def test_full_exit_can_close_verified_service_when_camera_stop_is_blocked(monkeypatch):
    terminated = []

    def request(path, method="GET"):
        if path == "/api/live/stop":
            raise urllib.error.HTTPError(stop.URL + path, 409, "Camera busy", {}, None)
        return {"application": "ppe-guard", "state": "idle"}

    monkeypatch.setattr(stop, "request", request)
    monkeypatch.setattr(stop.subprocess, "run", lambda *a, **k: terminated.append(True))
    stop.main()
    assert terminated == [True]
