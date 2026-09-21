import asyncio
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse
)


WEB_HOST = "0.0.0.0"
WEB_PORT = 8000


class AnnotatedFrameBroker:
    """
    Holds the latest annotated JPEG and writes screenshots to disk.
    """

    def __init__(
        self,
        save_dir,
        archive_interval=1.0,
        max_archives=200
    ):
        self.save_dir = Path(save_dir)
        self.archive_dir = self.save_dir / "annotated"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.archive_interval = archive_interval
        self.max_archives = max_archives

        self.lock = threading.Lock()
        self.jpeg = None
        self.last_archive = 0.0
        self.last_archive_name = None
        self.frame_count = 0

    def publish(self, frame):
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if not ok:
            return

        jpeg = buffer.tobytes()

        with self.lock:
            self.jpeg = jpeg
            self.frame_count += 1

        latest_path = self.save_dir / "latest.jpg"
        latest_path.write_bytes(jpeg)

        now = time.monotonic()

        if now - self.last_archive >= self.archive_interval:
            self._archive(jpeg)

    def save_now(self):
        with self.lock:
            jpeg = self.jpeg

        if jpeg is None:
            return None

        return self._archive(jpeg)

    def _archive(self, jpeg):
        stamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )[:-3]

        filename = f"{stamp}.jpg"
        path = self.archive_dir / filename
        path.write_bytes(jpeg)

        self.last_archive = time.monotonic()
        self.last_archive_name = filename

        archives = sorted(self.archive_dir.glob("*.jpg"))

        for old in archives[:-self.max_archives]:
            try:
                old.unlink()
            except Exception:
                pass

        return filename

    def get_jpeg(self):
        with self.lock:
            return self.jpeg

    def list_archives(self, limit=24):
        files = sorted(
            self.archive_dir.glob("*.jpg"),
            reverse=True
        )

        return [
            {
                "name": path.name,
                "url": f"/captures/annotated/{path.name}"
            }
            for path in files[:limit]
        ]


def local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def viewer_urls():
    return {
        "local": f"http://127.0.0.1:{WEB_PORT}",
        "lan": f"http://{local_ip()}:{WEB_PORT}"
    }


def create_app(broker, state, camera):
    app = FastAPI(title="Jeb Annotated Feed")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PAGE

    @app.get("/api/latest.jpg")
    def latest_jpg():
        jpeg = broker.get_jpeg()

        if jpeg is None:
            return Response(status_code=204)

        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/stream")
    async def stream():
        async def generate():
            while True:
                jpeg = broker.get_jpeg()

                if jpeg is not None:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )

                await asyncio.sleep(0.05)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/status")
    def status():
        with state.lock:
            payload = {
                "control_mode": state.control_mode,
                "mission": state.mission,
                "ball_detected": state.ball_detected,
                "object_class": state.object_class,
                "object_direction": state.object_direction,
                "ball_confidence": state.ball_confidence,
                "balloon_detected": state.balloon_detected,
                "balloon_class": state.balloon_class,
                "source": getattr(camera, "active_mode", None),
                "monitor": None,
                "last_archive": broker.last_archive_name,
                "frame_count": broker.frame_count
            }

        if (
            getattr(camera, "active_mode", None) == "screen"
            and hasattr(camera.camera, "monitor_index")
        ):
            payload["monitor"] = camera.camera.monitor_index

        return JSONResponse(payload)

    @app.get("/api/captures")
    def captures():
        return JSONResponse(broker.list_archives())

    @app.post("/api/save")
    def save():
        filename = broker.save_now()

        if filename is None:
            return JSONResponse(
                {"saved": False, "error": "No annotated frame yet."},
                status_code=409
            )

        return JSONResponse({
            "saved": True,
            "name": filename,
            "url": f"/captures/annotated/{filename}"
        })

    @app.get("/api/monitors")
    def monitors():
        if (
            getattr(camera, "active_mode", None) != "screen"
            or not hasattr(camera.camera, "list_monitors")
        ):
            return JSONResponse([])

        return JSONResponse(camera.camera.list_monitors())

    @app.post("/api/monitor/{monitor_id}")
    def set_monitor(monitor_id: int):
        if (
            getattr(camera, "active_mode", None) != "screen"
            or not hasattr(camera.camera, "set_monitor")
        ):
            return JSONResponse(
                {"ok": False, "error": "Screen capture is not active."},
                status_code=409
            )

        camera.camera.set_monitor(monitor_id)

        return JSONResponse({
            "ok": True,
            "monitor": camera.camera.monitor_index
        })

    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/captures",
        StaticFiles(directory=str(broker.save_dir)),
        name="captures"
    )

    return app


def start_web_viewer(broker, state, camera):
    import uvicorn

    app = create_app(broker, state, camera)
    config = uvicorn.Config(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run,
        daemon=True
    )
    thread.start()

    urls = viewer_urls()

    print()
    print("==============================")
    print("     ANNOTATED WEB FEED")
    print("==============================")
    print(f"Local: {urls['local']}")
    print(f"LAN:   {urls['lan']}")
    print("==============================")
    print()

    return server, urls


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Jeb Annotated FPV</title>
  <style>
    :root {
      --bg: #101418;
      --panel: #1b2128;
      --line: #2d3844;
      --text: #e8eef4;
      --muted: #93a1b0;
      --accent: #3dd68c;
      --warn: #ffb020;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 20px;
    }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    .actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button, select {
      background: #24303b;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    main {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
      padding: 16px;
    }
    .feed {
      background: #000;
      border: 1px solid var(--line);
      border-radius: 12px;
      min-height: 360px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .feed img { width: 100%; display: block; }
    .side { display: flex; flex-direction: column; gap: 12px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      background: #24303b;
      color: var(--muted);
    }
    .chip.on { background: #163527; color: var(--accent); }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .grid img {
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--line);
    }
    .note { color: var(--warn); font-size: 12px; line-height: 1.4; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Jeb annotated FPV</h1>
      <div class="sub">Live screen capture with object overlays</div>
    </div>
    <div class="actions">
      <select id="monitor"></select>
      <button id="save">Save screenshot</button>
    </div>
  </header>
  <main>
    <section class="feed">
      <img id="live" src="/api/stream" alt="Annotated live feed">
    </section>
    <aside class="side">
      <div class="card">
        <div class="chips">
          <span class="chip" id="mode">MODE</span>
          <span class="chip" id="mission">MISSION</span>
          <span class="chip" id="source">SOURCE</span>
          <span class="chip" id="ball">BALL</span>
          <span class="chip" id="object">OBJECT</span>
        </div>
      </div>
      <div class="card note">
        If this page is on the same monitor you are capturing,
        open it on another screen so Jeb does not annotate itself.
      </div>
      <div class="card">
        <div class="sub" id="last-save">No screenshots yet</div>
        <div class="grid" id="gallery"></div>
      </div>
    </aside>
  </main>
  <script>
    async function refreshStatus() {
      const res = await fetch("/api/status");
      const data = await res.json();
      setChip("mode", data.control_mode, true);
      setChip("mission", data.mission, true);
      setChip("source", data.source || "none", !!data.source);
      setChip("ball", data.ball_detected ? "BALL YES" : "BALL NO", data.ball_detected);
      setChip("object", data.object_class || "NO OBJECT", !!data.object_class);
      if (data.last_archive) {
        document.getElementById("last-save").textContent =
          "Last saved: " + data.last_archive;
      }
    }

    function setChip(id, text, on) {
      const el = document.getElementById(id);
      el.textContent = text;
      el.classList.toggle("on", on);
    }

    async function refreshGallery() {
      const res = await fetch("/api/captures");
      const items = await res.json();
      const gallery = document.getElementById("gallery");
      gallery.innerHTML = items.map(item =>
        `<a href="${item.url}" target="_blank">
           <img src="${item.url}" alt="${item.name}">
         </a>`
      ).join("");
    }

    async function loadMonitors() {
      const res = await fetch("/api/monitors");
      const monitors = await res.json();
      const select = document.getElementById("monitor");
      select.innerHTML = monitors.map(monitor =>
        `<option value="${monitor.id}">${monitor.label} (${monitor.width}x${monitor.height})</option>`
      ).join("");
    }

    document.getElementById("monitor").addEventListener("change", async (event) => {
      await fetch("/api/monitor/" + event.target.value, { method: "POST" });
    });

    document.getElementById("save").addEventListener("click", async () => {
      await fetch("/api/save", { method: "POST" });
      refreshGallery();
      refreshStatus();
    });

    loadMonitors();
    refreshStatus();
    refreshGallery();
    setInterval(refreshStatus, 1000);
    setInterval(refreshGallery, 4000);
  </script>
</body>
</html>
"""
