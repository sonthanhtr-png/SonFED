from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path

from PIL import Image, ImageDraw
import pystray

ROOT = Path(__file__).resolve().parent
URL = "http://localhost:8501"
process: subprocess.Popen | None = None


def make_icon() -> Image.Image:
    image = Image.new("RGB", (64, 64), "#111827")
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 12, 52, 52), fill="#f5c542")
    draw.text((24, 22), "S", fill="#111827")
    return image


def start_streamlit() -> None:
    global process
    if process and process.poll() is None:
        return
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.address", "127.0.0.1"], cwd=ROOT)


def open_dashboard(icon=None, item=None) -> None:
    start_streamlit()
    webbrowser.open(URL)


def stop_background(icon=None, item=None) -> None:
    global process
    if process and process.poll() is None:
        process.terminate()
    process = None


def quit_app(icon, item) -> None:
    stop_background()
    icon.stop()


def main() -> None:
    start_streamlit()
    icon = pystray.Icon(
        "SonFED",
        make_icon(),
        "SonFED",
        menu=pystray.Menu(
            pystray.MenuItem("Mở dashboard", open_dashboard),
            pystray.MenuItem("Tắt chạy nền", stop_background),
            pystray.MenuItem("Thoát", quit_app),
        ),
    )
    icon.run()


if __name__ == "__main__":
    main()
