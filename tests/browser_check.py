"""Production React UI + real HTTP/API checks using isolated simulated hardware."""

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import uvicorn
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    from app.cameras import CameraPool
    from app.config import ConfigStore
    from app.runtime import Runtime
    from app.server import create_app
    from tests.test_core import FakeCapture, FakeDetector
    from tests.test_media_export import video_bytes

    output = ROOT / "TEMPORARY"
    output.mkdir(exist_ok=True)
    checks, errors = [], []
    with tempfile.TemporaryDirectory(prefix="frontend-check-", dir=output) as scratch:
        scratch = Path(scratch)
        store, detector = ConfigStore(scratch / "settings.json"), FakeDetector()
        runtime = Runtime(store, detector, CameraPool(lambda _: FakeCapture()))
        app = create_app(
            store=store,
            detector=detector,
            runtime=runtime,
            initialize=False,
            scratch_root=scratch,
            capability_provider=lambda _: {
                "available": True,
                "device_id": "a" * 64,
                "message": "Simulated camera controls",
                "controls": {
                    "brightness": dict(min=0, max=255, step=1, default=128, current=128),
                    "auto_exposure": dict(
                        min=0, max=1, step=1, default=1, current=1, options=[0, 1]
                    ),
                },
            },
        )
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        with patch(
            "app.server.camera_catalog",
            return_value=[{"id": "test-usb-a", "index": 0, "name": "Test USB A"}],
        ):
            worker.start()
            try:
                for _ in range(100):
                    if server.started:
                        break
                    time.sleep(0.05)
                assert server.started, "Isolated HTTP server did not start"
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        channel="msedge", headless=True, downloads_path=str(scratch / "downloads")
                    )
                    context = browser.new_context(
                        viewport={"width": 1080, "height": 1920},
                        reduced_motion="reduce",
                        accept_downloads=True,
                    )
                    page = context.new_page()
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.add_init_script(
                        "window.policyErrors=[]; addEventListener('securitypolicyviolation', e=>window.policyErrors.push(e.violatedDirective+': '+e.blockedURI))"
                    )
                    try:
                        page.goto(url, wait_until="domcontentloaded")
                        expect(
                            page.get_by_role("button", name="Start detection", exact=True)
                        ).to_be_enabled()
                        expect(page).to_have_title("Live — AI Based PPE Monitoring & Warning")
                        expect(
                            page.get_by_role(
                                "heading", name="AI Based PPE Monitoring & Warning", exact=True
                            )
                        ).to_be_visible()
                        expect(page.locator(".brand-subtitle")).to_have_text(
                            "Safety First,No Harm to ANYONE"
                        )
                        assert page.locator(".live-toolbar").count() == 0
                        expect(page.locator(".kiosk-screen")).not_to_contain_text("test double")
                        expect(page.locator(".metric-value")).to_have_text(["—"] * 4)
                        expect(page.locator(".metric-label")).to_have_text(
                            [
                                "Total workers detected",
                                "Violations",
                                "Violation percentage",
                                "Workers currently in camera",
                            ]
                        )
                        for logo, width in (
                            (".goggles-logo", 1254),
                            (".vit-logo", 1575),
                            (".bekaert-logo", 3415),
                        ):
                            for image in page.locator(logo).all():
                                expect(image).to_have_js_property("naturalWidth", width)
                        footer = page.locator(".project-footer")
                        expect(footer.locator("li")).to_have_text(
                            [
                                "Ridham Chhajer",
                                "Kedar Naygaonkar",
                                "Kedar Pethkar",
                                "Sanskruti Chaure",
                                "Sahil Nehate",
                                "Om Parmar",
                            ]
                        )
                        assert footer.bounding_box()["height"] <= 110
                        vit_box = page.locator(".institute-credit").bounding_box()
                        bekaert_box = page.locator(".industry-credit").bounding_box()
                        assert vit_box["x"] + vit_box["width"] < bekaert_box["x"]
                        footer.screenshot(path=str(output / "branding-footer.png"))
                        assert page.locator(".sidebar-bottom, .bounding-box").count() == 0
                        assert page.locator(".camera-viewport canvas:visible").count() == 0
                        expect(page.get_by_text("NOT VISIBLE", exact=True)).to_have_count(3)
                        expect(page.get_by_text("IN PROGRESS", exact=True)).to_have_count(1)
                        expect(page.locator(".ppe-status-card")).to_have_count(4)
                        expect(page.locator(".ppe-status-card.neutral")).to_have_count(4)
                        page.locator(".ppe-status-grid").screenshot(
                            path=str(output / "ppe-status-cards.png")
                        )
                        assert (
                            abs(
                                page.locator(".kiosk-screen").bounding_box()["width"]
                                - page.locator("main").bounding_box()["width"]
                            )
                            < 1
                        )
                        assert page.evaluate(
                            "document.fonts.check('800 26px \"Plus Jakarta Sans\"')"
                        )
                        page.get_by_role("button", name="Expand sidebar").click()
                        expect(
                            page.get_by_role("button", name="Collapse sidebar")
                        ).to_have_attribute("aria-expanded", "true")
                        page.get_by_role("button", name="Collapse sidebar").click()
                        checks.append(
                            "Compiled UI, local fonts, neutral stopped state, full-width layout and collapsible sidebar"
                        )
                        page.get_by_role("link", name="Settings", exact=True).click()
                        expect(
                            page.get_by_role("heading", name="Inference & hardware", exact=True)
                        ).to_be_visible()
                        expect(page.locator(".inference-device")).to_have_text("test double")
                        expect(page.locator(".inference-panel")).to_contain_text(
                            "CUDA inference ready"
                        )
                        expect(page.get_by_label("Brightness custom value")).to_be_visible()
                        assert page.get_by_role("checkbox", name="Gain", exact=True).count() == 0
                        page.get_by_role("checkbox", name="Brightness", exact=True).check()
                        brightness = page.get_by_label("Brightness custom value")
                        expect(brightness).to_have_attribute("min", "0")
                        expect(brightness).to_have_attribute("max", "255")
                        brightness.fill("545")
                        page.get_by_role("button", name="Save settings").click()
                        expect(page.locator("#settings-error")).to_contain_text("Brightness")
                        brightness.fill("140")
                        page.get_by_role("checkbox", name="Auto exposure", exact=True).check()
                        page.get_by_label("Auto exposure custom value").select_option("1")
                        checks.append(
                            "Capability API shapes, unsupported controls hidden, brightness limits enforced and automatic mode selection"
                        )
                        page.get_by_label("Lower body — safety shoes", exact=True).select_option(
                            "index:0"
                        )
                        page.get_by_text("Advanced detection settings", exact=True).click()
                        page.get_by_label("Use SAHI slicing for wrist detection").uncheck()
                        page.get_by_role("button", name="Save settings").click()
                        expect(
                            page.get_by_role("status").filter(has_text="Settings saved.")
                        ).to_be_visible()
                        saved = page.request.get(url + "/api/settings").json()
                        assert saved["upper"]["camera"] == saved["lower"]["camera"] == 0
                        assert not saved["upper"]["sahi_wrist"]
                        page.get_by_label("Width", exact=True).fill("100")
                        page.get_by_role("button", name="Save settings").click()
                        expect(page.get_by_label("Width", exact=True)).to_have_attribute(
                            "aria-invalid", "true"
                        )
                        assert (
                            page.evaluate("document.activeElement.getAttribute('aria-label')")
                            == "Width"
                        )
                        page.get_by_label("Width", exact=True).fill("1280")
                        page.get_by_role("link", name="Live", exact=True).click()
                        expect(page.get_by_role("dialog")).to_be_visible()
                        assert (
                            page.evaluate("document.activeElement.textContent")
                            == "Stay in Settings"
                        )
                        page.keyboard.press("Escape")
                        expect(page.get_by_role("dialog")).to_be_hidden()
                        assert page.get_by_label("Width", exact=True).input_value() == "1280"
                        page.get_by_role("link", name="Live", exact=True).click()
                        page.get_by_role("button", name="Discard changes").click()
                        checks.append(
                            "Actual settings API saves shared camera/SAHI; validation, focus, unsaved modal, Escape and discard"
                        )
                        page.get_by_role("button", name="Start detection", exact=True).click()
                        expect(page.get_by_text("IN PROGRESS", exact=True)).to_have_count(1)
                        expect(
                            page.get_by_role("button", name="Stop detection", exact=True)
                        ).to_be_visible()
                        expect(page.locator(".camera-viewport canvas:visible")).to_have_count(2)
                        expect(page.locator(".metric-value")).to_have_text(["—"] * 4)
                        for feed in page.locator(".camera-viewport canvas:visible").all():
                            expect(feed).to_have_attribute("data-ready", "true")

                        for feed in page.locator(".camera-viewport canvas:visible").all():
                            assert feed.evaluate("node => node.width > 0 && node.height > 0")
                        panels = [
                            node.bounding_box() for node in page.locator(".camera-viewport").all()
                        ]
                        for width, height in [(1280, 720), (640, 480), (360, 640)]:
                            sample = np.zeros((height, width, 3), np.uint8)
                            cv2.rectangle(sample, (0, 0), (width - 1, height - 1), (0, 255, 0), 12)
                            ok, encoded = cv2.imencode(".jpg", sample)
                            assert ok
                            page.route(
                                "**/api/frame/*",
                                lambda route,
                                request,
                                body=encoded.tobytes(),
                                sequence=f"layout-{width}": route.fulfill(
                                    status=200,
                                    content_type="image/jpeg",
                                    body=body,
                                    headers={"X-Frame-Sequence": sequence},
                                ),
                            )
                            for index, feed in enumerate(
                                page.locator(".camera-viewport canvas").all()
                            ):
                                expect(feed).to_have_js_property("width", width)
                                expect(feed).to_have_js_property("height", height)
                                assert (
                                    feed.evaluate("node => getComputedStyle(node).objectFit")
                                    == "contain"
                                )
                                panel = page.locator(".camera-viewport").nth(index).bounding_box()
                                assert abs(panel["width"] - panels[index]["width"]) < 1
                                assert abs(panel["height"] - panels[index]["height"]) < 1
                                # Green source-frame edges must remain visible after letterboxing.
                                shot = cv2.imdecode(
                                    np.frombuffer(feed.screenshot(), np.uint8), cv2.IMREAD_COLOR
                                )
                                h, w = shot.shape[:2]
                                scale = min(w / width, h / height)
                                left = round((w - width * scale) / 2 + 3 * scale)
                                top = round((h - height * scale) / 2 + 3 * scale)
                                assert shot[h // 2, left, 1] > 180
                                assert shot[top, w // 2, 1] > 180
                            page.unroute("**/api/frame/*")
                        page.get_by_role("button", name="Expand upper camera", exact=True).click()
                        page.wait_for_function(
                            "document.fullscreenElement?.classList.contains('camera-viewport')"
                        )
                        expect(
                            page.get_by_role("button", name="Exit fullscreen for upper camera")
                        ).to_have_attribute("aria-pressed", "true")
                        page.get_by_role("button", name="Exit fullscreen for upper camera").click()
                        page.wait_for_function("!document.fullscreenElement")
                        page.evaluate(
                            "() => { document.querySelector('.lower-camera .camera-viewport').requestFullscreen = () => Promise.reject(new Error('unavailable')); }"
                        )
                        page.get_by_role("button", name="Expand lower camera", exact=True).click()
                        expect(page.locator(".camera-expand-error")).to_contain_text(
                            "Fullscreen is unavailable"
                        )
                        page.evaluate(
                            "delete document.querySelector('.lower-camera .camera-viewport').requestFullscreen"
                        )
                        assert (
                            abs(
                                page.locator(".camera-viewport").first.bounding_box()["height"]
                                - panels[0]["height"]
                            )
                            < 1
                        )
                        checks.append(
                            "Camera panels keep their dimensions across landscape, 4:3 and portrait feeds; full-frame edges stay visible; fullscreen expands and restores the same feed"
                        )
                        upper_feed = page.locator(".upper-camera canvas")
                        page.evaluate(
                            "window.stablePreview = document.querySelector('.upper-camera canvas')"
                        )
                        for failure in (204, 200, 503):
                            page.route(
                                "**/api/frame/upper",
                                lambda route, request, code=failure: route.fulfill(
                                    status=code,
                                    content_type="image/jpeg",
                                    body=b"" if code == 204 else b"invalid-jpeg",
                                ),
                            )
                            expect(upper_feed).to_be_hidden()
                            expect(page.locator(".lower-camera canvas")).to_be_visible()
                            page.unroute("**/api/frame/upper")
                            expect(upper_feed).to_be_visible(timeout=5000)
                            assert page.evaluate(
                                "window.stablePreview === document.querySelector('.upper-camera canvas')"
                            )
                        checks.append(
                            "No-frame, corrupt JPEG and failed HTTP responses clear the preview and recover independently"
                        )
                        expect(page.locator(".ppe-status-card").first).to_contain_text("DETECTED")
                        page.screenshot(path=str(output / "live-portrait.png"), full_page=True)
                        for observation, tone in (("detected", "green"), ("violation", "red")):

                            def status_colours(route, request, value=observation):
                                response = route.fetch()
                                body = response.json()
                                body["views"]["upper"]["summary"].update(goggles=value, wrist=value)
                                body["views"]["lower"]["summary"].update(shoes=value)
                                route.fulfill(response=response, json=body)

                            page.route("**/api/status", status_colours)
                            expect(page.locator(f".ppe-status-card.{tone}")).to_have_count(3)
                            expect(page.locator(".ppe-status-card.neutral")).to_have_count(1)
                            ratios = page.locator(f".ppe-status-card.{tone}").evaluate_all(r"""cards => {
                              const luminance = color => {
                                const rgb = color.match(/[\d.]+/g).slice(0,3).map(x => {
                                  const c = Number(x) / 255;
                                  return c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4;
                                });
                                return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
                              };
                              const contrast = (a,b) => {
                                const x=luminance(a), y=luminance(b);
                                return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);
                              };
                              return cards.flatMap(card => {
                                const pill=getComputedStyle(card.querySelector('.ppe-pill-badge'));
                                const result=getComputedStyle(card.querySelector('.detection-result-text'));
                                return [contrast(pill.color,pill.backgroundColor),
                                  contrast(result.color,getComputedStyle(card).backgroundColor)];
                              });
                            }""")
                            assert min(ratios) >= 4.5, ratios
                            page.locator(".ppe-status-grid").screenshot(
                                path=str(output / f"ppe-cards-{tone}.png")
                            )
                            page.unroute("**/api/status", status_colours)
                        checks.append(
                            "Green/red card backgrounds preserve 4.5:1 text contrast; earplugs remain neutral"
                        )
                        original_predict = detector.predict
                        resume_predictions = threading.Event()
                        prediction_waiting = threading.Event()

                        def delayed_predict(*args):
                            prediction_waiting.set()
                            resume_predictions.wait(timeout=10)
                            return original_predict(*args)

                        detector.predict = delayed_predict
                        try:
                            assert prediction_waiting.wait(timeout=3)
                            page.evaluate(
                                "window.stableFeed = document.querySelector('.upper-camera canvas')"
                            )
                            with runtime.lock:
                                for state in runtime.states.values():
                                    state["timestamp"] = time.monotonic() - 5
                            expect(page.get_by_text("NOT VISIBLE", exact=True)).to_have_count(3)
                            expect(page.locator(".camera-viewport canvas:visible")).to_have_count(2)
                            expect(page.get_by_text("UPDATING", exact=True)).to_have_count(2)
                            assert page.evaluate(
                                "window.stableFeed === document.querySelector('.upper-camera canvas')"
                            )
                            assert page.get_by_text("Camera unavailable", exact=True).count() == 0
                            for feed in page.locator(".camera-viewport canvas:visible").all():
                                expect(feed).to_have_attribute("data-ready", "true")
                            old_frame = upper_feed.get_attribute("data-frame")
                            expect(upper_feed).not_to_have_attribute("data-frame", old_frame)
                            page.screenshot(
                                path=str(output / "live-slow-inference.png"), full_page=True
                            )
                        finally:
                            detector.predict = original_predict
                            resume_predictions.set()
                        expect(page.locator(".ppe-status-card").first).to_contain_text("DETECTED")
                        checks.append(
                            "Slow inference keeps advancing frame canvases, clears stale evidence, and recovers without reconnecting"
                        )
                        page.get_by_role("button", name="Stop detection", exact=True).click()
                        expect(page.get_by_text("NOT VISIBLE", exact=True)).to_have_count(3)
                        expect(page.locator(".camera-viewport canvas:visible")).to_have_count(0)
                        checks.append(
                            "Finite JPEG requests and live lifecycle with simulated capture/detector; stop clears evidence"
                        )
                        page.get_by_role("link", name="Test media", exact=True).click()
                        assert (
                            page.get_by_label("Detection mode", exact=True).input_value() == "both"
                        )
                        page.get_by_role("button", name="Run detection").click()
                        expect(page.locator("#media-error")).to_contain_text("Choose an image")
                        page.locator("#test-file").set_input_files(
                            {"name": "bad.txt", "mimeType": "text/plain", "buffer": b"invalid"}
                        )
                        expect(page.locator("#media-error")).to_contain_text("Choose a JPEG")
                        ok, image = cv2.imencode(".png", np.zeros((120, 160, 3), np.uint8))
                        assert ok
                        page.locator("#test-file").set_input_files(
                            {
                                "name": "image.png",
                                "mimeType": "image/png",
                                "buffer": image.tobytes(),
                            }
                        )
                        expect(page.get_by_alt_text("Selected image — unprocessed")).to_be_visible()
                        page.get_by_role("button", name="Run detection").click()
                        expect(page.locator(".result-caption")).to_contain_text(
                            "Test complete", timeout=15000
                        )
                        assert page.request.get(url + "/api/test").json()["view"] == "both"
                        page.get_by_role("button", name="Stop & remove").click()
                        payload = video_bytes(scratch / "input.avi")
                        page.locator("#test-file").set_input_files(
                            {"name": "clip.avi", "mimeType": "video/x-msvideo", "buffer": payload}
                        )
                        page.get_by_role("button", name="Run detection").click()
                        link = page.get_by_role("link", name="Download detection video")
                        expect(link).to_be_visible(timeout=15000)
                        player = page.get_by_label("Processed detection video")
                        expect(player).to_be_visible()
                        page.wait_for_function(
                            "document.querySelector('video[aria-label=\"Processed detection video\"]').readyState >= 2"
                        )
                        player.evaluate(
                            "async video => { video.muted = true; await video.play(); }"
                        )
                        page.wait_for_function(
                            "document.querySelector('video[aria-label=\"Processed detection video\"]').currentTime > 0"
                        )
                        player.evaluate("video => video.pause()")
                        with page.expect_download() as download_info:
                            link.click()
                        download = download_info.value
                        assert download.failure() is None, download.failure()
                        assert download.suggested_filename == "ppe-detection.mp4"
                        downloaded = scratch / "result.mp4"
                        download.save_as(downloaded)
                        cap = cv2.VideoCapture(str(downloaded))
                        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 3
                        assert cap.read()[0]
                        cap.release()
                        page.get_by_role("button", name="Stop & remove").click()
                        expect(link).to_have_count(0)
                        assert not list((scratch / "exports").glob("ppe-*.mp4"))
                        checks.append(
                            "Actual combined image/video processing, annotated MP4 encoding, browser download transfer and cleanup (fake inference only)"
                        )
                        page.get_by_role("link", name="Live", exact=True).click()
                        page.route("**/api/status", lambda route: route.abort())
                        expect(
                            page.get_by_text("The application service is unavailable.", exact=False)
                        ).to_be_visible(timeout=15000)
                        expect(page.get_by_text("NOT VISIBLE", exact=True)).to_have_count(3)
                        page.unroute("**/api/status")
                        expect(
                            page.get_by_text("The application service is unavailable.", exact=False)
                        ).to_have_count(0, timeout=15000)
                        checks.append(
                            "Backend loss clears evidence and recovers without sample fallback"
                        )
                        for width, height in [(925, 900), (1366, 768), (390, 844)]:
                            page.set_viewport_size({"width": width, "height": height})
                            for name in ["Live", "Test media", "Sample dashboard", "Settings"]:
                                page.get_by_role("link", name=name, exact=True).click()
                                if name == "Live":
                                    upper = page.locator(".upper-camera").bounding_box()
                                    lower = page.locator(".lower-camera").bounding_box()
                                    assert lower["y"] >= upper["y"] + upper["height"]
                                    assert abs(upper["x"] - lower["x"]) < 1
                                    controls = [
                                        page.locator(selector).bounding_box()
                                        for selector in (
                                            ".time-badge-card",
                                            ".system-status-pill",
                                            ".detection-toggle",
                                        )
                                    ]
                                    centers = [box["y"] + box["height"] / 2 for box in controls]
                                    assert max(centers) - min(centers) < 2, (width, controls)
                                    assert controls[0]["x"] < controls[1]["x"] < controls[2]["x"]
                                    expect(page.locator(".metric-value")).to_have_text(["—"] * 4)
                                    assert page.locator(".project-footer").bounding_box()[
                                        "height"
                                    ] <= (180 if width < 650 else 110), (
                                        width,
                                        page.locator(".project-footer").bounding_box(),
                                    )
                                    for selector in (".vit-logo", ".bekaert-logo"):
                                        assert page.locator(selector).evaluate(
                                            "node => Math.abs(node.getBoundingClientRect().width / node.getBoundingClientRect().height - node.naturalWidth / node.naturalHeight) < 0.02"
                                        )
                                    page.locator(".ppe-header").screenshot(
                                        path=str(output / f"header-{width}.png")
                                    )
                                    page.locator(".project-footer").screenshot(
                                        path=str(output / f"branding-footer-{width}.png")
                                    )
                                assert page.evaluate(
                                    "document.documentElement.scrollWidth <= innerWidth"
                                ), (width, name)
                            page.evaluate("window.scrollTo(0,0)")
                            page.screenshot(
                                path=str(output / f"settings-{width}.png"), full_page=True
                            )
                        for width, height in [(1708, 960), (1093, 614), (683, 384)]:
                            page.set_viewport_size({"width": width, "height": height})
                            page.get_by_role("link", name="Live", exact=True).click()
                            assert page.evaluate(
                                "document.documentElement.scrollWidth <= innerWidth"
                            )
                            expect(
                                page.get_by_role("button", name="Expand upper camera")
                            ).to_be_visible()
                        checks.append(
                            "Camera 1 stays above Camera 2 on desktop and mobile; 80%, 125% and 200% equivalent CSS viewport sizes reflow without horizontal overflow"
                        )
                        page.get_by_role("link", name="Settings", exact=True).click()
                        page.get_by_label("Upper body — goggles & wrists", exact=True).focus()
                        page.keyboard.press("Alt+ArrowDown")
                        page.keyboard.press("Escape")
                        assert page.evaluate("document.activeElement.tagName") == "SELECT"
                        page.keyboard.press("Tab")
                        assert (
                            page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
                            != "none"
                        )
                        assert (
                            page.evaluate(
                                "getComputedStyle(document.documentElement).scrollbarColor"
                            )
                            != "auto"
                        )
                        assert not page.evaluate("window.policyErrors"), page.evaluate(
                            "window.policyErrors"
                        )
                        assert not errors, errors
                        checks.append(
                            "All routes at laptop/mobile sizes; native select, visible keyboard focus, scrollbars and no CSP violations"
                        )
                    finally:
                        context.close()
                        browser.close()
            finally:
                server.should_exit = True
                worker.join(timeout=10)
                listener.close()
    report = {
        "result": "passed",
        "checks": checks,
        "javascript_errors": errors,
        "scope": "Production compiled frontend and real API/codec/download code; isolated fake detector/cameras. Operator settings, real cameras and training weights are not altered.",
    }
    (output / "browser-evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
