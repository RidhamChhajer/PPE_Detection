import io
import time
from datetime import datetime
from fastapi.testclient import TestClient

from app.audit import AuditStore
from app.config import ConfigStore
from app.reporting import generate_daily_pdf
from app.server import create_app
from tests.test_core import FakeDetector


def test_audit_store_records_and_aggregates_inspections(tmp_path):
    db_file = tmp_path / "test_audit.db"
    store = AuditStore(db_file)
    today = datetime.now().strftime("%Y-%m-%d")

    # Record compliant detection
    store.record("upper", {"goggles": "detected", "wrist": "detected", "shoes": "unknown"}, [{"class": "goggles", "confidence": 0.88}])
    # Record violation detection
    store.record("upper", {"goggles": "violation", "wrist": "detected", "shoes": "unknown"}, [{"class": "no_goggles", "confidence": 0.74}])

    summary = store.get_daily_summary(today)
    assert summary["date"] == today
    assert summary["total_inspections"] >= 2
    assert summary["violation_count"] >= 1
    assert summary["categories"]["goggles"]["violations"] >= 1


def test_pdf_generation_returns_valid_pdf_stream(tmp_path):
    db_file = tmp_path / "test_audit2.db"
    store = AuditStore(db_file)
    today = datetime.now().strftime("%Y-%m-%d")
    summary = store.get_daily_summary(today)

    pdf_bytes = generate_daily_pdf(today, summary)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


def test_dashboard_api_endpoints(tmp_path):
    app = create_app(
        store=ConfigStore(tmp_path / "settings.json"),
        detector=FakeDetector(),
        initialize=False,
        scratch_root=tmp_path / "scratch",
    )
    with TestClient(app) as client:
        # Dates endpoint
        res = client.get("/api/dashboard/dates")
        assert res.status_code == 200
        assert "dates" in res.json()
        assert len(res.json()["dates"]) > 0

        # Summary endpoint
        res = client.get("/api/dashboard/summary")
        assert res.status_code == 200
        data = res.json()
        assert "total_inspections" in data
        assert "compliance_rate" in data
        assert "categories" in data
        assert "hourly" in data

        # PDF download endpoint
        res = client.get("/api/reports/daily/pdf")
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        assert res.content.startswith(b"%PDF")

        # Email config endpoints
        res = client.get("/api/reports/email-config")
        assert res.status_code == 200
        cfg = res.json()
        assert "schedule_time" in cfg
        assert isinstance(cfg["schedule_time"], str) and len(cfg["schedule_time"]) == 5
