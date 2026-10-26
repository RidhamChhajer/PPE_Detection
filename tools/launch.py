import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8876"


def ready():
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=1) as response:
            return json.load(response).get("application") == "ppe-guard"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main():
    if not ready():
        scratch = ROOT / "TEMPORARY"
        scratch.mkdir(exist_ok=True)
        with (scratch / "server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8876",
                    "--no-access-log",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        for _ in range(120):
            if ready():
                break
            if process.poll() is not None:
                raise RuntimeError(
                    "Application could not start. See TEMPORARY/server.log. Port 8876 may be in use."
                )
            time.sleep(0.5)
        else:
            raise RuntimeError("Startup timed out. See TEMPORARY/server.log.")
    webbrowser.open(URL)
    print("PPE Detection System is open. Use Stop monitoring to release USB cameras.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(exc)
        input("Press Enter to close.")
        sys.exit(1)
