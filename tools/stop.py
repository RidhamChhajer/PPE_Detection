"""Stop the local PPE Detection System service without targeting unrelated Python programs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8876"

STOP_SERVER = r"""
$ErrorActionPreference = 'Stop'
$listener = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort 8876 -State Listen
$serverId = @($listener.OwningProcess | Select-Object -Unique)
if ($serverId.Count -ne 1) { throw 'Could not identify one PPE Detection System service.' }
$server = Get-CimInstance Win32_Process -Filter "ProcessId = $($serverId[0])"
if ($server.CommandLine -notmatch '-m\s+uvicorn\s+app\.server:app(?:\s|$)' -or
    $server.CommandLine -notmatch '--port\s+8876(?:\s|$)') {
    throw 'Port 8876 belongs to an unexpected process. Nothing was terminated.'
}
& "$env:SystemRoot\System32\taskkill.exe" /PID $serverId[0] /T /F
if ($LASTEXITCODE -ne 0) { throw 'The service could not be stopped.' }
"""


def request(path, method="GET"):
    with urllib.request.urlopen(
        urllib.request.Request(URL + path, method=method), timeout=30
    ) as response:
        return json.load(response)


def main():
    try:
        health = request("/api/health")
    except urllib.error.URLError:
        print("PPE Detection System is not reachable on port 8876. No process was terminated.")
        return
    if health.get("application") != "ppe-guard":
        raise RuntimeError("Port 8876 is not PPE Detection System. No process was terminated.")
    if request("/api/test").get("state") == "running":
        request("/api/test", "DELETE")
    try:
        request("/api/live/stop", "POST")
    except urllib.error.HTTPError as exc:
        if exc.code != 409:
            raise
        print("A camera or GPU operation is stuck; closing the verified application process.")
    subprocess.run(
        [
            str(
                Path(os.environ.get("SystemRoot", "C:/Windows"))
                / "System32/WindowsPowerShell/v1.0/powershell.exe"
            ),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            STOP_SERVER,
        ],
        check=True,
    )
    print("PPE Detection System stopped. Start PPE Detection System.cmd opens it again.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print("Could not close PPE Detection System:", exc)
        sys.exit(1)
