import asyncio
import base64
import json
import smtplib
import threading
import time
from datetime import datetime, timedelta
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import ROOT

CONFIG_FILE = ROOT / "config" / "email_config.json"

DEFAULT_CONFIG = {
    "enabled": True,
    "service": "smtp",  # "smtp" or "emailjs"
    "schedule_time": "05:30",
    "recipients": ["safety-officer@factory.internal"],
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_pass": "",
    "smtp_use_tls": True,
    "emailjs_service_id": "",
    "emailjs_template_id": "",
    "emailjs_public_key": "",
    "emailjs_private_key": "",
    "last_sent_at": None,
    "last_status": "Ready | Scheduled daily at 05:30 AM",
}


def parse_schedule_time(sched_input: str) -> tuple[int, int, str]:
    """
    Parses arbitrary human or HTML5 time input into (hour, minute, "HH:MM").
    Supports 24h ('05:30', '17:45'), 12h with AM/PM ('7:30 PM', '7:30pm', '7am'),
    and dot-delimited times ('07.30').
    """
    raw = str(sched_input or "05:30").strip().lower()
    is_pm = "pm" in raw
    is_am = "am" in raw
    cleaned = raw.replace("am", "").replace("pm", "").replace(".", ":").strip()
    parts = cleaned.split(":")
    try:
        if len(parts) >= 2:
            h = int(parts[0].strip())
            m = int(parts[1].strip())
        elif len(parts) == 1:
            h = int(parts[0].strip())
            m = 0
        else:
            h, m = 5, 30

        if is_pm and h < 12:
            h += 12
        elif is_am and h == 12:
            h = 0

        h = max(0, min(23, h))
        m = max(0, min(59, m))
        norm = f"{h:02d}:{m:02d}"
        return h, m, norm
    except Exception:
        return 5, 30, "05:30"


def compute_next_schedule(sched_str: str, now: datetime = None, grace_seconds: int = 45) -> tuple[datetime, float, str, str]:
    """
    Calculates next trigger datetime, wait seconds, normalized "HH:MM", and human status.
    If the schedule time was within the last `grace_seconds` (e.g. operator saved current minute),
    fires in ~3 seconds rather than jumping 24 hours to tomorrow.
    """
    if now is None:
        now = datetime.now()
    target_h, target_m, norm_time = parse_schedule_time(sched_str)
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    diff = (target - now).total_seconds()

    if -grace_seconds <= diff < 0:
        # Operator configured the current minute right now: trigger immediately in ~3s
        target = now + timedelta(seconds=3)
        wait_seconds = 3.0
        status = f"Scheduled for {norm_time} (firing in ~3 seconds)"
    elif diff < -grace_seconds:
        # Time has already passed today -> schedule for tomorrow
        target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        hours_rem = wait_seconds / 3600.0
        status = f"Scheduled for {norm_time} tomorrow (in ~{hours_rem:.1f} hours)"
    else:
        wait_seconds = max(0.5, diff)
        mins_rem = max(1, int(round(wait_seconds / 60.0)))
        if wait_seconds < 60:
            status = f"Scheduled for {norm_time} (in {int(wait_seconds)}s)"
        elif mins_rem < 60:
            status = f"Scheduled for {norm_time} (in ~{mins_rem} min{'s' if mins_rem != 1 else ''})"
        else:
            hours_rem = wait_seconds / 3600.0
            status = f"Scheduled for {norm_time} (in ~{hours_rem:.1f} hours)"

    return target, wait_seconds, norm_time, status


class MailerService:
    def __init__(self, config_path=None):
        self.config_path = config_path or CONFIG_FILE
        self.lock = threading.RLock()
        self._last_mtime = 0
        self.config = self._load_config()
        self.task = None
        self.schedule_event = None

    def _check_disk_updates(self):
        """Detect external edits to email_config.json on disk (e.g., from IDE or external script)."""
        try:
            if not self.config_path.exists():
                return
            mtime = self.config_path.stat().st_mtime
            if self._last_mtime > 0 and mtime > self._last_mtime:
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                with self.lock:
                    self.config.update(loaded)
                    self._last_mtime = mtime
                if self.schedule_event is not None:
                    self.schedule_event.set()
            else:
                self._last_mtime = mtime
        except Exception:
            pass

    def _load_config(self) -> dict:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            try:
                self._last_mtime = self.config_path.stat().st_mtime
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
                merged = dict(DEFAULT_CONFIG)
                merged.update(loaded)
                return merged
            except Exception:
                pass
        self._save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)

    def _save_config(self, cfg: dict):
        self.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        try:
            self._last_mtime = self.config_path.stat().st_mtime
        except Exception:
            pass

    def _mask_config(self, cfg: dict) -> dict:
        masked = dict(cfg)
        if masked.get("smtp_pass"):
            masked["has_smtp_pass"] = True
            masked["smtp_pass"] = "••••••••"
        else:
            masked["has_smtp_pass"] = False
        return masked

    def get_config(self) -> dict:
        self._check_disk_updates()
        with self.lock:
            return self._mask_config(self.config)

    def update_config(self, updates: dict) -> dict:
        with self.lock:
            # Don't overwrite password if masked
            if updates.get("smtp_pass") == "••••••••":
                updates.pop("smtp_pass", None)

            # Normalize schedule time if provided
            if "schedule_time" in updates:
                _, _, norm_time = parse_schedule_time(updates["schedule_time"])
                updates["schedule_time"] = norm_time

            self.config.update(updates)

            # Immediately compute next scheduled status so API response has real-time feedback
            sched = self.config.get("schedule_time", "05:30")
            target_dt, _, norm_time, status_text = compute_next_schedule(sched)
            self.config["schedule_time"] = norm_time
            self.config["next_scheduled_at"] = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            self.config["last_status"] = status_text

            self._save_config(self.config)
            result = self._mask_config(self.config)

        # Signal background scheduler to wake up immediately and adapt to new time
        if self.schedule_event is not None:
            try:
                self.schedule_event.set()
            except Exception:
                pass

        return result

    def send_pdf_email(
        self,
        pdf_bytes: bytes,
        date_str: str,
        recipients: list[str] = None,
        custom_config: dict = None,
    ) -> tuple[bool, str]:
        """Dispatch the PDF report via SMTP or EmailJS."""
        cfg = custom_config or self.config
        recips = recipients or cfg.get("recipients", [])
        recips = [r.strip() for r in recips if r.strip()]

        if not recips:
            return False, "No recipient email addresses specified."

        service = cfg.get("service", "smtp").lower()

        if service == "smtp":
            return self._send_smtp(pdf_bytes, date_str, recips, cfg)
        elif service == "emailjs":
            return self._send_emailjs(pdf_bytes, date_str, recips, cfg)
        else:
            return False, f"Unknown email service: {service}"

    def _send_smtp(
        self, pdf_bytes: bytes, date_str: str, recipients: list[str], cfg: dict
    ) -> tuple[bool, str]:
        host = cfg.get("smtp_host", "smtp.gmail.com")
        port = int(cfg.get("smtp_port", 587))
        user = cfg.get("smtp_user", "").strip()
        pwd = cfg.get("smtp_pass", "").strip()
        if "gmail.com" in host.lower():
            pwd = pwd.replace(" ", "")
        use_tls = cfg.get("smtp_use_tls", True)

        if not user or not pwd:
            return (
                False,
                "SMTP username and password / App Password are required to send emails.",
            )

        msg = MIMEMultipart()
        msg["From"] = f"AI PPE Monitoring System <{user}>"
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = f"Daily PPE Compliance Audit Report - {date_str}"

        body_text = (
            f"AI Based PPE Monitoring & Warning System\n"
            f"Daily Safety Compliance Audit Report\n\n"
            f"Audit Date: {date_str}\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Facility: Bekaert Manufacturing Plant & VIT Pune\n\n"
            f"Please find the attached PDF containing key safety metrics, equipment compliance rates, "
            f"hourly trends, and safety violation incidents.\n\n"
            f"---\n"
            f"Sent automatically by the AI PPE Monitoring Service."
        )
        msg.attach(MIMEText(body_text, "plain"))

        part = MIMEApplication(pdf_bytes, _subtype="pdf")
        filename = f"ppe-daily-report-{date_str}.pdf"
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)

        try:
            with smtplib.SMTP(host, port, timeout=20) as server:
                if use_tls:
                    server.starttls()
                server.login(user, pwd)
                server.send_message(msg)
            return True, f"Email sent successfully via SMTP to {len(recipients)} recipient(s)."
        except Exception as exc:
            return False, f"SMTP delivery failed: {exc}"

    def _send_emailjs(
        self, pdf_bytes: bytes, date_str: str, recipients: list[str], cfg: dict
    ) -> tuple[bool, str]:
        service_id = cfg.get("emailjs_service_id", "").strip()
        template_id = cfg.get("emailjs_template_id", "").strip()
        public_key = cfg.get("emailjs_public_key", "").strip()
        private_key = cfg.get("emailjs_private_key", "").strip()

        if not service_id or not template_id or not public_key:
            return (
                False,
                "EmailJS Service ID, Template ID, and Public Key are required.",
            )

        # Base64 encode the PDF attachment for EmailJS REST API
        pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

        payload = {
            "service_id": service_id,
            "template_id": template_id,
            "user_id": public_key,
            "template_params": {
                "to_email": ", ".join(recipients),
                "date": date_str,
                "subject": f"PPE Daily Compliance Report - {date_str}",
                "report_summary": f"Daily PPE inspection report for {date_str}.",
                "content": f"Automated shift report generated on {date_str}.",
            },
        }

        # EmailJS allows accessToken (private key) when called from backend
        if private_key:
            payload["accessToken"] = private_key

        # Attach PDF if template is configured for attachment
        payload["template_params"]["attachment"] = f"data:application/pdf;base64,{pdf_b64}"

        data = json.dumps(payload).encode("utf-8")
        req = Request(
            "https://api.emailjs.com/api/v1.0/email/send",
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "PPE-Monitoring-System/2.0",
            },
        )

        try:
            with urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    return (
                        True,
                        f"Email sent successfully via EmailJS to {len(recipients)} recipient(s).",
                    )
                return False, f"EmailJS responded with HTTP {resp.status}"
        except HTTPError as err:
            err_msg = err.read().decode("utf-8", errors="replace")
            return False, f"EmailJS API Error ({err.code}): {err_msg}"
        except URLError as err:
            return False, f"Network error connecting to EmailJS: {err.reason}"
        except Exception as exc:
            return False, f"EmailJS dispatch failed: {exc}"


mailer_service = MailerService()


async def daily_scheduler_loop(audit_store, reporting_module):
    """Background coroutine that dynamically fires at the configured schedule time to dispatch the daily PDF."""
    if mailer_service.schedule_event is None:
        mailer_service.schedule_event = asyncio.Event()

    print("[MailerScheduler] Dynamic email scheduler initialized.")
    while True:
        try:
            mailer_service._check_disk_updates()
            cfg = mailer_service.get_config()
            sched_str = cfg.get("schedule_time", "05:30")

            target_dt, wait_seconds, norm_time, status_text = compute_next_schedule(sched_str)
            next_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")

            with mailer_service.lock:
                mailer_service.config["schedule_time"] = norm_time
                mailer_service.config["next_scheduled_at"] = next_str
                mailer_service.config["last_status"] = status_text
                mailer_service._save_config(mailer_service.config)

            print(
                f"[MailerScheduler] Armed for {norm_time} at {next_str} ({wait_seconds:.0f}s / {wait_seconds / 3600.0:.2f}h remaining)."
            )

            # Wait until target time OR wake up immediately if operator modifies settings!
            try:
                await asyncio.wait_for(mailer_service.schedule_event.wait(), timeout=wait_seconds)
                # Event triggered! Clear event and loop back immediately to adapt to new time!
                mailer_service.schedule_event.clear()
                print("[MailerScheduler] Schedule settings updated! Recalculating schedule immediately...")
                continue
            except (asyncio.TimeoutError, TimeoutError):
                # Scheduled delivery time reached!
                pass

            # Target time reached!
            # Build rolling 24h window: (schedule_time yesterday) -> (schedule_time today)
            now = datetime.now()
            sched_h, sched_m, norm_time_parsed = parse_schedule_time(norm_time)
            window_end = now.replace(hour=sched_h, minute=sched_m, second=0, microsecond=0)
            if window_end > now:
                window_end -= timedelta(days=1)
            window_start = window_end - timedelta(hours=24)
            report_date = window_end.strftime("%Y-%m-%d")

            print(
                f"[MailerScheduler] Target time {norm_time} reached! "
                f"Rolling 24h window: {window_start.strftime('%Y-%m-%d %H:%M')} -> "
                f"{window_end.strftime('%Y-%m-%d %H:%M')}"
            )

            cfg = mailer_service.get_config()
            if cfg.get("enabled") and cfg.get("recipients"):
                # Use rolling 24h summary (exact window, not calendar day)
                summary = audit_store.get_rolling_24h_summary(sched_h, sched_m)
                pdf_bytes = reporting_module.generate_daily_pdf(report_date, summary)
                ok, message = mailer_service.send_pdf_email(pdf_bytes, report_date)

                now_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with mailer_service.lock:
                    mailer_service.config["last_sent_at"] = now_stamp
                    mailer_service.config["last_status"] = (
                        f"Delivered at {now_stamp}: {message}" if ok else f"Failed at {now_stamp}: {message}"
                    )
                    mailer_service._save_config(mailer_service.config)

                print(f"[MailerScheduler] Automated delivery result: {message}")

                if ok:
                    # Purge data older than today and yesterday to save disk space.
                    # Keep only the 2 most recent calendar days.
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                    try:
                        audit_store.purge_old_data(keep_dates=[today_str, yesterday_str])
                        print(f"[MailerScheduler] Old data purged. Keeping: {today_str}, {yesterday_str}")
                    except Exception as purge_exc:
                        print(f"[MailerScheduler] Purge failed (non-critical): {purge_exc}")
            else:
                print("[MailerScheduler] Skipping automated delivery: email service is disabled or has no recipients.")

            # Pause to ensure we cross beyond the current second
            await asyncio.sleep(5)

        except asyncio.CancelledError:
            print("[MailerScheduler] Background scheduler task cancelled.")
            break
        except Exception as exc:
            print(f"[MailerScheduler] Unexpected error in scheduler: {exc}")
            await asyncio.sleep(10)
