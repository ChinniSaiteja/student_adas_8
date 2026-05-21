"""
FrameViewer — live MJPEG stream of the camera feed with browser-based display pages.

Opens a tiny HTTP server so you can watch the simulation in any browser
without needing a display server or GUI toolkit.

Usage::

    from adas.viewer import FrameViewer

    viewer = FrameViewer()   # default port 8080
    viewer.start()

    def process_image(self, image):
        viewer.push(image)   # call inside your callback

Or with automatic telemetry (call once after creating both objects)::

    self.register_viewer(self._viewer)   # in CarlaADASInterface subclass

Useful routes:

    /           operator/dashboard view
    /display    fullscreen presentation view
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import cv2        # type: ignore
import numpy as np  # type: ignore


def _make_placeholder(width: int = 640, height: int = 360) -> bytes:
    img = np.full((height, width, 3), 30, dtype=np.uint8)
    lines = ["Waiting for simulation stream...", "http://localhost:8080"]
    y0 = height // 2 - 20
    for i, line in enumerate(lines):
        size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = (width - size[0]) // 2
        cv2.putText(img, line, (x, y0 + i * 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HSHL ADAS Control Surface</title>
  <style>
    :root {
      --bg: #06090d;
      --panel: rgba(9, 16, 21, 0.72);
      --panel-strong: rgba(9, 16, 21, 0.9);
      --line: rgba(196, 231, 224, 0.14);
      --line-strong: rgba(196, 231, 224, 0.26);
      --text: #f3efe5;
      --muted: #a6b7b6;
      --soft: #768785;
      --accent: #9fe6d2;
      --accent-2: #ffc677;
      --warn: #ffd174;
      --alert: #ff8d7b;
      --shadow: 0 26px 60px rgba(0, 0, 0, 0.3);
      --radius: 26px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(86, 156, 154, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(226, 177, 101, 0.14), transparent 26%),
        linear-gradient(180deg, #0a1015 0%, var(--bg) 48%, #030507 100%);
      color: var(--text);
      font-family: "Aptos", "Segoe UI Variable Text", "Bahnschrift", "Trebuchet MS", sans-serif;
      padding: 24px;
    }
    .page {
      max-width: 1540px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .glass {
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      background-color: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px) saturate(120%);
    }
    .hero {
      border-radius: calc(var(--radius) + 4px);
      padding: 22px 24px;
      display: grid;
      grid-template-columns: minmax(260px, 1.3fr) minmax(320px, 1fr);
      gap: 16px;
      align-items: center;
    }
    .hero-copy {
      display: grid;
      gap: 12px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      width: fit-content;
      border-radius: 999px;
      border: 1px solid var(--line-strong);
      background: rgba(255,255,255,0.04);
      padding: 8px 12px;
      color: var(--accent);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }
    .eyebrow-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 18px rgba(159, 230, 210, 0.6);
      animation: pulse 1.8s ease-in-out infinite;
    }
    .hero h1 {
      font-size: clamp(2rem, 3vw, 3.8rem);
      line-height: 0.97;
      letter-spacing: -0.04em;
      max-width: 12ch;
    }
    .hero p {
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.58;
      max-width: 58ch;
    }
    .hero-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      font-size: 0.88rem;
      color: var(--text);
    }
    .status-panel {
      display: grid;
      grid-template-columns: repeat(2, minmax(160px, 1fr));
      gap: 12px;
    }
    .status-card {
      border-radius: 22px;
      padding: 16px 18px;
    }
    .status-label {
      color: var(--soft);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 0.72rem;
      margin-bottom: 8px;
    }
    .status-value {
      color: var(--text);
      font-size: 1.18rem;
      line-height: 1.15;
      letter-spacing: -0.03em;
      min-height: 1.2em;
    }
    .status-value.dim { color: var(--soft); }
    .status-value.warn { color: var(--warn); }
    .status-value.alert { color: var(--alert); }
    .hud-banner {
      border-radius: 999px;
      padding: 16px 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      min-height: 64px;
      transition: border-color 0.2s ease, color 0.2s ease, opacity 0.2s ease;
    }
    .hud-banner.idle { color: var(--soft); }
    .hud-banner.info { color: var(--text); border-color: var(--line-strong); }
    .hud-banner.warning { color: var(--warn); border-color: rgba(255, 209, 116, 0.34); }
    .hud-banner.alert { color: var(--alert); border-color: rgba(255, 141, 123, 0.34); }
    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.72fr);
      gap: 18px;
      align-items: start;
    }
    .stage {
      display: grid;
      gap: 16px;
    }
    .cam-wrap {
      position: relative;
      overflow: hidden;
      border-radius: 30px;
      min-height: 360px;
      background: #0a1116;
    }
    .cam-wrap::after {
      content: "";
      position: absolute;
      inset: 0;
      pointer-events: none;
      background:
        linear-gradient(180deg, rgba(0,0,0,0.28), rgba(0,0,0,0.05) 22%, rgba(0,0,0,0.1) 68%, rgba(0,0,0,0.34)),
        radial-gradient(circle at center, rgba(0,0,0,0) 54%, rgba(0,0,0,0.3) 100%);
    }
    .cam-wrap img {
      width: 100%;
      display: block;
      min-height: 360px;
      max-height: 72vh;
      object-fit: cover;
      filter: saturate(1.02) contrast(1.03);
    }
    #overlay {
      position: absolute; inset: 0;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: rgba(6, 11, 15, 0.9);
      transition: opacity 0.5s ease;
      z-index: 2;
    }
    #overlay.hidden { opacity: 0; pointer-events: none; }
    .spinner {
      width: 54px; height: 54px;
      border: 5px solid rgba(255,255,255,0.08);
      border-top-color: var(--accent);
      border-radius: 50%; animation: spin 1s linear infinite;
      margin-bottom: 16px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #status-msg { color: var(--muted); font-size: 0.95rem; letter-spacing: 0.03em; }
    .dot { display: inline-block; animation: pulse 1.4s ease-in-out infinite; }
    .dot:nth-child(2) { animation-delay: 0.18s; }
    .dot:nth-child(3) { animation-delay: 0.36s; }
    @keyframes pulse { 0%,80%,100% { opacity: 0.35; } 40% { opacity: 1; } }
    .stage-footer {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
    }
    .quick-card {
      border-radius: 20px;
      padding: 16px 18px;
    }
    .quick-label {
      color: var(--soft);
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-size: 0.7rem;
      margin-bottom: 8px;
    }
    .quick-value {
      font-size: 1.15rem;
      letter-spacing: -0.02em;
      line-height: 1.12;
    }
    .sidebar {
      border-radius: 30px;
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .sidebar-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding-bottom: 4px;
    }
    .sidebar-title strong {
      font-size: 1rem;
      letter-spacing: -0.02em;
    }
    .sidebar-title span {
      color: var(--soft);
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }
    .telemetry-card {
      border-radius: 22px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.05);
    }
    .mgroup { margin-bottom: 12px; }
    .mgroup:last-child { margin-bottom: 0; }
    .mlabel {
      font-size: 0.7rem; color: var(--soft);
      text-transform: uppercase; letter-spacing: 0.08em;
      margin-bottom: 6px;
    }
    .mval {
      font-size: 0.92rem; color: var(--muted);
      font-variant-numeric: tabular-nums; line-height: 1.6;
    }
    .mval.big {
      font-size: 2.1rem; color: var(--text); font-weight: 600;
      line-height: 1.1; letter-spacing: -0.01em;
    }
    .munit { font-size: 0.72rem; color: var(--soft); margin-left: 4px; }
    .mdim { color: #44514f; }
    .event-actor { color: var(--accent-2); font-size: 0.86rem; word-break: break-word; }
    .event-ago   { color: var(--soft); font-size: 0.76rem; margin-top: 3px; }
    .lane-type   { color: var(--warn); font-size: 0.88rem; }
    .sep { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 12px 0; }
    footer {
      color: var(--soft);
      font-size: 0.84rem;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      padding: 4px 2px 0;
    }
    footer a { color: var(--accent); text-decoration: none; }
    footer a:hover { color: #c9fff0; }
    @media (max-width: 1180px) {
      .hero,
      .workspace {
        grid-template-columns: 1fr;
      }
      .status-panel,
      .stage-footer {
        grid-template-columns: repeat(2, minmax(140px, 1fr));
      }
      .hero h1 { max-width: none; }
    }
    @media (max-width: 760px) {
      body { padding: 14px; }
      .hero,
      .sidebar,
      .cam-wrap,
      .status-card,
      .telemetry-card,
      .quick-card {
        border-radius: 22px;
      }
      .status-panel,
      .stage-footer {
        grid-template-columns: 1fr;
      }
      footer { flex-direction: column; align-items: flex-start; }
      .cam-wrap img { max-height: none; min-height: 260px; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero glass">
      <div class="hero-copy">
        <div class="eyebrow"><span class="eyebrow-dot"></span>HSHL ADAS Control Surface</div>
        <h1>Production-style visibility for the live lab stack.</h1>
        <p>The dashboard keeps rich telemetry and recovery context, but now uses the same visual language as the kiosk presentation so Machine B feels polished in both operator mode and public demo mode.</p>
        <div class="hero-pills">
          <span class="pill">Machine B operator dashboard</span>
          <span class="pill">Live stream with graceful fallback</span>
          <span class="pill">Unified runtime and telemetry status</span>
        </div>
      </div>
      <div class="status-panel">
        <div class="status-card glass">
          <div class="status-label">Mode</div>
          <div class="status-value" id="s-mode">&mdash;</div>
        </div>
        <div class="status-card glass">
          <div class="status-label">Map</div>
          <div class="status-value" id="s-map">&mdash;</div>
        </div>
        <div class="status-card glass">
          <div class="status-label">FPS</div>
          <div class="status-value" id="s-fps">&mdash;</div>
        </div>
        <div class="status-card glass">
          <div class="status-label">Recording</div>
          <div class="status-value" id="s-rec">&mdash;</div>
        </div>
        <div class="status-card glass">
          <div class="status-label">Wheel</div>
          <div class="status-value" id="s-wheel">&mdash;</div>
        </div>
        <div class="status-card glass">
          <div class="status-label">Stream</div>
          <div class="status-value" id="status-msg-inline">waiting</div>
        </div>
      </div>
    </section>

    <div class="hud-banner glass idle" id="hud-banner">Waiting for HUD events and runtime state...</div>

    <section class="workspace">
      <div class="stage">
        <div class="cam-wrap glass">
          <img id="feed" src="/stream" alt="camera stream">
          <div id="overlay">
            <div class="spinner"></div>
            <div id="status-msg">
              Waiting for simulation stream<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
            </div>
          </div>
        </div>
        <div class="stage-footer">
          <div class="quick-card glass">
            <div class="quick-label">Speed</div>
            <div class="quick-value" id="t-speed"><span class="mdim">&#8212;</span></div>
          </div>
          <div class="quick-card glass">
            <div class="quick-label">Frame Freshness</div>
            <div class="quick-value" id="frame-freshness">--</div>
          </div>
          <div class="quick-card glass">
            <div class="quick-label">Telemetry Freshness</div>
            <div class="quick-value" id="ui-freshness">--</div>
          </div>
          <div class="quick-card glass">
            <div class="quick-label">Presentation</div>
            <div class="quick-value"><a href="/display" style="color: var(--accent); text-decoration: none;">Open fullscreen view</a></div>
          </div>
        </div>
      </div>

      <aside class="sidebar glass">
        <div class="sidebar-title">
          <strong>Telemetry</strong>
          <span>Deep signal view</span>
        </div>

        <div class="telemetry-card">
          <div class="mgroup">
            <div class="mlabel">IMU</div>
            <div class="mval" id="t-imu-ax"><span class="mdim">ax &mdash;</span></div>
            <div class="mval" id="t-imu-ay"><span class="mdim">ay &mdash;</span></div>
            <div class="mval" id="t-imu-gz"><span class="mdim">gz &mdash;</span></div>
          </div>
        </div>

        <div class="telemetry-card">
          <div class="mgroup">
            <div class="mlabel">GNSS</div>
            <div class="mval" id="t-lat"><span class="mdim">&mdash;</span></div>
            <div class="mval" id="t-lon"><span class="mdim">&mdash;</span></div>
            <div class="mval" id="t-alt"><span class="mdim">alt &mdash;</span></div>
          </div>
        </div>

        <div class="telemetry-card">
          <div class="mgroup">
            <div class="mlabel">Last Collision</div>
            <div id="t-collision"><span class="mval mdim">none</span></div>
          </div>
          <hr class="sep">
          <div class="mgroup">
            <div class="mlabel">Lane Marking</div>
            <div id="t-lane"><span class="mval mdim">none</span></div>
          </div>
        </div>
      </aside>
    </section>

    <footer>
      <span>Operator view for live supervision, diagnostics, and presentation handoff.</span>
      <span><a href="/stream" target="_blank">Direct stream URL</a> · <a href="/display" target="_blank">Fullscreen kiosk</a></span>
    </footer>
  </div>

  <script>
    const overlay   = document.getElementById('overlay');
    const statusMsg = document.getElementById('status-msg');
    const statusMsgInline = document.getElementById('status-msg-inline');
    const hudBanner = document.getElementById('hud-banner');
    const feed = document.getElementById('feed');
    let   ready     = false;
    let   snapshotTimer = null;
    let   usingSnapshotMode = false;

    // ── helpers ────────────────────────────────────────────
    function dim(text) { return '<span class="mdim">' + text + '</span>'; }

    function fmtVal(v, digits, unit) {
      if (v === null || v === undefined) return dim('\u2014');
      const sign = v >= 0 ? '+' : '';
      return sign + v.toFixed(digits) + '\u202f' + unit;
    }

    function agoStr(ago_s) {
      if (ago_s === null || ago_s === undefined) return '';
      if (ago_s < 60)  return (ago_s | 0) + 's ago';
      return ((ago_s / 60) | 0) + 'm ago';
    }

    function freshnessStr(ago_s) {
      if (ago_s === null || ago_s === undefined) return '--';
      if (ago_s < 1) return '<1s';
      return ago_s.toFixed(1) + 's';
    }

    function setStatus(id, text, cssClass='') {
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = 'status-value' + (cssClass ? ' ' + cssClass : '');
    }

    function startSnapshotMode() {
      if (usingSnapshotMode) return;
      usingSnapshotMode = true;

      function tickSnapshot() {
        feed.src = '/frame.jpg?t=' + Date.now();
      }

      tickSnapshot();
      snapshotTimer = setInterval(tickSnapshot, 100);
    }

    function setupFeed() {
      feed.addEventListener('error', () => {
        startSnapshotMode();
      });

      // Start with MJPEG stream first, fallback to snapshots if browser rejects it.
      feed.src = '/stream';
    }

    // ── unified data poll ──────────────────────────────────
    function refresh() {
      fetch('/data')
        .then(r => r.json())
        .then(d => {

          // loading overlay
          if (d.stream_state === 'live' && !ready) {
            ready = true;
            overlay.classList.add('hidden');
          } else if (d.stream_state !== 'live' && ready) {
            ready = false;
            overlay.classList.remove('hidden');
          }
          statusMsg.textContent = d.status_message || 'Waiting for simulation stream...';
          statusMsgInline.textContent = d.stream_state || 'waiting';

          // speed
          const sp = document.getElementById('t-speed');
          sp.innerHTML = d.speed !== null
            ? d.speed.toFixed(1)
            : dim('\u2014');
          document.getElementById('frame-freshness').textContent = freshnessStr(d.frame_age_s);
          document.getElementById('ui-freshness').textContent = freshnessStr(d.ui_state_age_s);

          // runtime ui state
          const ui = d.ui_state;
          const uiFresh = d.ui_state_age_s === null || d.ui_state_age_s < 2.5;
          setStatus('s-mode', ui ? (ui.mode_text || ui.state || '\u2014') : '\u2014', ui ? (uiFresh ? '' : 'warn') : 'dim');
          setStatus('s-map', ui ? (ui.map_label || ui.map_name || '\u2014') : '\u2014', ui ? (uiFresh ? '' : 'warn') : 'dim');
          setStatus('s-fps', ui && ui.fps !== null && ui.fps !== undefined ? ui.fps.toFixed(1) : '\u2014', ui ? (uiFresh ? '' : 'warn') : 'dim');
          setStatus('s-rec', ui ? (ui.recording ? 'REC' : 'idle') : '\u2014', ui && ui.recording ? 'alert' : (ui ? (uiFresh ? '' : 'warn') : 'dim'));
          setStatus('s-wheel', ui ? (ui.wheel_connected ? 'connected' : 'missing') : '\u2014', ui ? (ui.wheel_connected ? (uiFresh ? '' : 'warn') : 'warn') : 'dim');

          // hud banner
          const hud = d.hud_event;
          if (d.stream_state !== 'live') {
            hudBanner.textContent = d.status_message || 'Waiting for simulation stream...';
            hudBanner.className = 'hud-banner warning';
          } else if (hud && hud.text) {
            hudBanner.textContent = hud.text;
            hudBanner.className = 'hud-banner ' + (hud.level || 'info');
          } else if (ui && ui.mode_text) {
            hudBanner.textContent = ui.mode_text;
            hudBanner.className = 'hud-banner info';
          } else {
            hudBanner.textContent = 'Waiting for HUD events and runtime state...';
            hudBanner.className = 'hud-banner idle';
          }

          // imu
          const imu = d.imu;
          document.getElementById('t-imu-ax').innerHTML =
            imu ? 'ax\u2009' + fmtVal(imu.ax, 2, 'm/s\u00b2') : dim('ax \u2014');
          document.getElementById('t-imu-ay').innerHTML =
            imu ? 'ay\u2009' + fmtVal(imu.ay, 2, 'm/s\u00b2') : dim('ay \u2014');
          document.getElementById('t-imu-gz').innerHTML =
            imu ? 'gz\u2009' + fmtVal(imu.gz, 3, 'rad/s')  : dim('gz \u2014');

          // gnss
          const g = d.gnss;
          document.getElementById('t-lat').innerHTML =
            g ? g.lat.toFixed(6) + '\u00b0N' : dim('\u2014');
          document.getElementById('t-lon').innerHTML =
            g ? g.lon.toFixed(6) + '\u00b0E' : dim('\u2014');
          document.getElementById('t-alt').innerHTML =
            g ? 'alt\u2009' + g.alt.toFixed(1) + '\u202fm' : dim('alt \u2014');

          // collision
          const col = d.collision;
          document.getElementById('t-collision').innerHTML = col
            ? '<div class="event-actor">' + col.actor + '</div>'
              + '<div class="event-ago">'
              + col.impulse_mag.toFixed(1) + '\u202fN\u00b7s &bull; '
              + agoStr(col.ago_s) + '</div>'
            : '<span class="mval mdim">none</span>';

          // lane
          const lane = d.lane;
          document.getElementById('t-lane').innerHTML = lane
            ? '<span class="lane-type">' + lane.types.join(', ') + '</span>'
              + '<div class="event-ago">' + agoStr(lane.ago_s) + '</div>'
            : '<span class="mval mdim">none</span>';
        })
        .catch(() => {});

      setTimeout(refresh, 500);
    }

    setupFeed();
    refresh();
  </script>
</body>
</html>
""".encode("utf-8")

_DISPLAY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HSHL ADAS Experience</title>
  <style>
    :root {
      --bg: #04070a;
      --surface: rgba(8, 14, 18, 0.36);
      --line: rgba(197, 229, 222, 0.16);
      --line-strong: rgba(197, 229, 222, 0.28);
      --text: #f6f3ea;
      --muted: #9eb0b1;
      --soft: #75898a;
      --accent: #9fe6d2;
      --warn: #ffcf73;
      --alert: #ff8d7b;
      --record: #ff4d4d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%; height: 100%; overflow: hidden;
      background:
        radial-gradient(circle at top left, rgba(97, 168, 168, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(226, 173, 90, 0.16), transparent 28%),
        linear-gradient(180deg, #091015 0%, var(--bg) 40%, #020304 100%);
      color: var(--text);
      font-family: "Aptos", "Segoe UI Variable Display", "Bahnschrift", "Trebuchet MS", sans-serif;
    }
    body { position: relative; }
    #feed {
      position: fixed; inset: 0;
      width: 100vw; height: 100vh; object-fit: cover; background: #020406;
      filter: saturate(1.03) contrast(1.05);
      transform: scale(1.01);
    }
    .film {
      position: fixed; inset: 0;
      background:
        radial-gradient(circle at center, rgba(0,0,0,0) 44%, rgba(0,0,0,0.38) 100%),
        linear-gradient(180deg, rgba(5, 9, 12, 0.86) 0%, rgba(5, 9, 12, 0.12) 28%, rgba(5, 9, 12, 0.18) 68%, rgba(5, 9, 12, 0.88) 100%),
        linear-gradient(90deg, rgba(4, 7, 9, 0.78) 0%, rgba(4, 7, 9, 0.18) 26%, rgba(4, 7, 9, 0.08) 74%, rgba(4, 7, 9, 0.72) 100%);
      pointer-events: none;
    }
    .glass {
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
      background-color: var(--surface);
      border: 1px solid var(--line);
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.2);
      backdrop-filter: blur(14px) saturate(110%);
    }
    .overlay {
      position: fixed;
      inset: 0;
      z-index: 2;
      pointer-events: none;
    }
    .lab-stack {
      position: fixed;
      top: 18px;
      left: 18px;
      width: min(240px, calc(100vw - 36px));
      display: grid;
      gap: 8px;
      pointer-events: auto;
    }
    .lab-chip,
    .mini-stat,
    .mode-bar,
    .speed-dock,
    .banner,
    .modal-card {
      border-radius: 18px;
    }
    .lab-chip {
      width: fit-content;
      padding: 8px 12px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--line-strong);
      background: rgba(255,255,255,0.04);
      color: var(--accent);
      font-size: 0.67rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 18px rgba(159, 230, 210, 0.65);
      animation: pulse 1.8s ease-in-out infinite;
    }
    .mini-stat {
      padding: 10px 12px;
      display: grid;
      gap: 4px;
      width: fit-content;
      min-width: 150px;
    }
    .mini-label {
      color: var(--soft);
      font-size: 0.62rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .mini-value {
      font-size: 0.96rem;
      line-height: 1.1;
      letter-spacing: -0.02em;
    }
    .link-row {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .warn { color: var(--warn); }
    .alert { color: var(--alert); }
    .accent { color: var(--accent); }
    .signal {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .signal-bar {
      width: 5px;
      border-radius: 999px;
      background: rgba(255,255,255,0.18);
    }
    .signal.live .signal-bar:nth-child(1) { height: 8px; background: var(--accent); }
    .signal.live .signal-bar:nth-child(2) { height: 12px; background: var(--accent); }
    .signal.live .signal-bar:nth-child(3) { height: 16px; background: var(--accent); }
    .signal.live .signal-bar:nth-child(4) { height: 20px; background: var(--accent); }
    .signal.stale .signal-bar:nth-child(-n+2) { background: var(--warn); }
    .signal.waiting .signal-bar:nth-child(1) { background: var(--soft); }
    .signal.error .signal-bar:nth-child(1) { background: var(--alert); }
    .mode-bar {
      position: fixed;
      top: 18px;
      left: 50%;
      transform: translateX(-50%);
      width: min(340px, calc(100vw - 110px));
      padding: 10px 14px;
      text-align: center;
      pointer-events: auto;
    }
    .mode-label {
      color: var(--soft);
      font-size: 0.58rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }
    .mode-value {
      font-size: 1.1rem;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }
    .banner {
      position: fixed;
      top: 78px;
      left: 50%;
      transform: translateX(-50%);
      width: min(520px, calc(100vw - 40px));
      padding: 9px 12px;
      text-align: center;
      pointer-events: none;
      transition: opacity 180ms ease, transform 220ms ease, border-color 220ms ease;
      animation: rise 900ms ease-out;
    }
    .banner.hidden {
      opacity: 0;
      transform: translateX(-50%) translateY(10px);
    }
    .banner-kicker {
      color: var(--soft);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      font-size: 0.62rem;
      margin-bottom: 3px;
    }
    .banner-text {
      font-size: clamp(0.92rem, 1.05vw, 1.08rem);
      line-height: 1.24;
      letter-spacing: -0.02em;
    }
    .speed-dock {
      position: fixed;
      right: 18px;
      bottom: 18px;
      min-width: 120px;
      padding: 10px 12px;
      pointer-events: auto;
      display: grid;
      justify-items: end;
      gap: 8px;
      animation: rise 1060ms ease-out;
    }
    .speed-value {
      font-size: 1.42rem;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }
    .speed-main {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .speed-unit {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      border: 1px solid var(--line-strong);
      background: rgba(255,255,255,0.03);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.95rem;
      letter-spacing: -0.01em;
      color: var(--text);
    }
    .record-row {
      display: flex;
      align-items: center;
      gap: 0;
    }
    .record-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: transparent;
      opacity: 0;
    }
    .record-dot.active {
      background: var(--record);
      opacity: 1;
      box-shadow: 0 0 14px rgba(255, 77, 77, 0.7);
      animation: blink 1s ease-in-out infinite;
    }
    .modal {
      position: fixed;
      inset: 0;
      z-index: 3;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      pointer-events: none;
    }
    .modal-card {
      width: min(640px, 92vw);
      border-radius: 28px;
      padding: 30px 32px;
      text-align: center;
      pointer-events: auto;
      transition: opacity 220ms ease, transform 220ms ease;
      animation: floatIn 520ms ease-out;
    }
    .modal-card.hidden {
      opacity: 0;
      transform: scale(0.98) translateY(10px);
      pointer-events: none;
    }
    .modal-title {
      font-size: clamp(1.6rem, 2.4vw, 2.25rem);
      letter-spacing: -0.04em;
      margin-bottom: 12px;
    }
    .loading-spinner {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      border: 5px solid rgba(255,255,255,0.14);
      border-top-color: var(--accent);
      margin: 0 auto 14px;
      animation: spin 1s linear infinite;
    }
    .modal-text {
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.6;
    }
    .modal-meta {
      margin-top: 18px;
      display: inline-flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .meta-chip {
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.04);
      padding: 9px 13px;
      font-size: 0.86rem;
      color: var(--text);
    }
    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.22; }
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @keyframes pulse {
      0%, 100% { opacity: 0.65; transform: scale(0.92); }
      50% { opacity: 1; transform: scale(1); }
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes floatIn {
      from { opacity: 0; transform: scale(0.97) translateY(18px); }
      to { opacity: 1; transform: scale(1) translateY(0); }
    }
    @media (max-width: 1200px) {
      .mode-bar {
        width: min(300px, calc(100vw - 90px));
      }
    }
    @media (max-width: 740px) {
      .lab-stack {
        top: 14px;
        left: 14px;
      }
      .mode-bar {
        top: 14px;
        width: min(240px, calc(100vw - 80px));
      }
      .banner {
        top: 70px;
        width: calc(100vw - 28px);
      }
      .speed-dock {
        right: 14px;
        bottom: 14px;
      }
      .modal-card { padding: 24px 22px; }
    }
  </style>
</head>
<body>
  <img id="feed" src="/stream" alt="simulation stream">
  <div class="film"></div>

  <div class="overlay">
    <div class="lab-stack">
      <div class="lab-chip glass">
        <div class="eyebrow"><span class="dot"></span>HSHL Autonomous Driving Lab</div>
      </div>
      <div class="mini-stat glass">
        <div class="mini-label">FPS</div>
        <div class="mini-value" id="fps">--</div>
      </div>
      <div class="mini-stat glass">
        <div class="mini-label">Operator Link</div>
        <div class="link-row"><span class="signal waiting" id="signal"><span class="signal-bar"></span><span class="signal-bar"></span><span class="signal-bar"></span><span class="signal-bar"></span></span><span class="mini-value" id="status">waiting</span></div>
      </div>
    </div>

    <section class="mode-bar glass">
      <div class="mode-label">Drive Mode</div>
      <div class="mode-value" id="mode">--</div>
    </section>

    <section class="banner glass" id="banner">
      <div class="banner-kicker" id="banner-kicker">Live status</div>
      <div class="banner-text" id="banner-text">Waiting for simulation stream...</div>
    </section>

    <section class="speed-dock glass">
      <div class="speed-main"><div class="speed-value" id="speed">--</div><div class="speed-unit">km/h</div></div>
      <div class="record-row"><span class="record-dot" id="record-dot"></span></div>
    </section>
  </div>

  <div class="modal">
    <div class="modal-card glass" id="center-card">
      <div class="loading-spinner"></div>
      <div class="modal-title" id="center-title">Connecting</div>
      <div class="modal-text" id="center-text">Waiting for live frames and runtime state from Machine A.</div>
      <div class="modal-meta">
        <span class="meta-chip">Machine B kiosk</span>
        <span class="meta-chip" id="center-chip-state">status waiting</span>
        <span class="meta-chip" id="center-chip-map">map --</span>
      </div>
    </div>
  </div>

  <script>
    const feed = document.getElementById('feed');
    const banner = document.getElementById('banner');
    const bannerKicker = document.getElementById('banner-kicker');
    const bannerText = document.getElementById('banner-text');
    const centerCard = document.getElementById('center-card');
    const centerTitle = document.getElementById('center-title');
    const centerText = document.getElementById('center-text');
    const centerChipState = document.getElementById('center-chip-state');
    const centerChipMap = document.getElementById('center-chip-map');
    const status = document.getElementById('status');
    const recordDot = document.getElementById('record-dot');
    const signal = document.getElementById('signal');
    let snapshotTimer = null;
    let usingSnapshotMode = false;

    function setText(id, text) {
      document.getElementById(id).textContent = text;
    }

    function startSnapshotMode() {
      if (usingSnapshotMode) return;
      usingSnapshotMode = true;

      function tickSnapshot() {
        feed.src = '/frame.jpg?t=' + Date.now();
      }

      tickSnapshot();
      snapshotTimer = setInterval(tickSnapshot, 100);
    }

    feed.addEventListener('error', () => {
      startSnapshotMode();
    });

    feed.src = '/stream';

    function setStreamVisual(state) {
      signal.className = 'signal ' + state;
    }
    document.addEventListener('dblclick', openFullscreen);
    document.addEventListener('keydown', (event) => {
      if (event.key.toLowerCase() === 'f') openFullscreen();
    });

    function openFullscreen() {
      if (document.fullscreenElement) return;
      document.documentElement.requestFullscreen?.().catch(() => {});
    }

    function refresh() {
      fetch('/data')
        .then(r => r.json())
        .then(d => {
          const ui = d.ui_state || {};
          const hudEvent = d.hud_event || null;
          const live = d.stream_state === 'live';
          const stale = d.stream_state === 'stale';
          const streamState = d.stream_state || 'waiting';
          const modeText = ui.mode_text || ui.state || '--';
          const mapText = ui.map_label || ui.map_name || '--';
          const speedText = d.speed !== null ? d.speed.toFixed(1) : '--';
          const fpsText = ui.fps !== undefined && ui.fps !== null ? ui.fps.toFixed(1) + ' fps' : '--';
          const linkText = ui.wheel_connected === true ? 'connected' : (streamState === 'live' ? 'live' : streamState);

          setText('mode', modeText);
          setText('speed', speedText);
          setText('fps', fpsText);
          setText('status', linkText);
          recordDot.className = ui.recording ? 'record-dot active' : 'record-dot';

          setStreamVisual(streamState);
          centerChipState.textContent = 'status ' + streamState;
          centerChipMap.textContent = 'map ' + mapText;

          if (!live) {
            banner.className = 'banner glass';
            bannerKicker.textContent = stale ? 'Signal degraded' : 'Connection in progress';
            bannerText.textContent = d.status_message || 'Waiting for simulation stream...';
            centerCard.className = 'modal-card glass';
            centerTitle.textContent = stale ? 'Connection Stale' : 'Connecting';
            centerText.textContent = d.status_message || 'Waiting for live frames and runtime state from Machine A.';
          } else {
            centerCard.className = 'modal-card glass hidden';
            if (hudEvent && hudEvent.text) {
              banner.className = 'banner glass';
              bannerKicker.textContent = (hudEvent.level || 'info').replace('_', ' ');
              bannerText.textContent = hudEvent.text;
              if (hudEvent.level === 'warning') {
                bannerText.className = 'banner-text warn';
              } else if (hudEvent.level === 'alert') {
                bannerText.className = 'banner-text alert';
              } else {
                bannerText.className = 'banner-text';
              }
            } else {
              banner.className = 'banner glass hidden';
              bannerKicker.textContent = 'Live status';
              bannerText.textContent = ui.mode_text || 'Live simulation stream';
              bannerText.className = 'banner-text';
            }
          }
        })
        .catch(() => {
          banner.className = 'banner glass';
          bannerKicker.textContent = 'Display fault';
          bannerText.textContent = 'Display connection error';
          bannerText.className = 'banner-text alert';
          centerCard.className = 'modal-card glass';
          centerTitle.textContent = 'Connection Error';
          centerText.textContent = 'Machine B could not refresh the local display endpoint.';
          status.textContent = 'error';
          setStreamVisual('error');
        });

      setTimeout(refresh, 500);
    }

    refresh();
  </script>
</body>
</html>
""".encode("utf-8")

_FRONTEND_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HSHL Frontend Viewer</title>
  <style>
    :root {
      --bg: #05070a;
      --panel: rgba(10, 15, 19, 0.68);
      --line: rgba(189, 234, 218, 0.18);
      --text: #f4efe2;
      --muted: #95a8a6;
      --accent: #9fe6d2;
      --warn: #ffd06f;
      --alert: #ff8f7c;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(98, 176, 176, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(214, 160, 90, 0.14), transparent 28%),
        linear-gradient(180deg, #091017 0%, var(--bg) 48%, #020304 100%);
      color: var(--text);
      font-family: "Aptos", "Segoe UI Variable Display", "Bahnschrift", sans-serif;
      padding: 20px;
    }
    .shell {
      max-width: 1700px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
    }
    .glass {
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
      background-color: var(--panel);
      border: 1px solid var(--line);
      backdrop-filter: blur(16px) saturate(120%);
      box-shadow: 0 24px 52px rgba(0,0,0,0.28);
      border-radius: 24px;
    }
    .top {
      padding: 18px 20px;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 14px;
      align-items: center;
    }
    .title {
      display: flex;
      align-items: center;
      gap: 10px;
      letter-spacing: 0.03em;
    }
    .pulse {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 14px rgba(159, 230, 210, 0.65);
      animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse { 0%,100%{opacity:0.55;transform:scale(0.9);} 50%{opacity:1;transform:scale(1);} }
    .title h1 {
      font-size: clamp(1.2rem, 2vw, 1.8rem);
      letter-spacing: -0.02em;
    }
    .controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .btn {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.05);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      font: inherit;
      cursor: pointer;
    }
    .btn:hover { border-color: rgba(255,255,255,0.38); }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(300px, 0.75fr);
      gap: 16px;
      align-items: start;
    }
    .stage {
      position: relative;
      overflow: hidden;
      min-height: 420px;
    }
    .stage img {
      width: 100%;
      min-height: 420px;
      max-height: 78vh;
      object-fit: cover;
      display: block;
      filter: saturate(1.03) contrast(1.04);
    }
    #overlay {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      background: rgba(6, 11, 15, 0.9);
      transition: opacity 220ms ease;
      padding: 20px;
    }
    #overlay.hidden { opacity: 0; pointer-events: none; }
    .overlay-box {
      max-width: 620px;
      display: grid;
      gap: 10px;
    }
    .overlay-title { font-size: clamp(1.4rem, 2.4vw, 2.2rem); }
    .overlay-text { color: var(--muted); line-height: 1.5; }
    .ribbon {
      position: absolute;
      left: 14px;
      right: 14px;
      top: 14px;
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      z-index: 2;
    }
    .chip {
      border-radius: 16px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: rgba(8, 14, 18, 0.78);
      backdrop-filter: blur(8px);
    }
    .chip-k {
      color: var(--muted);
      font-size: 0.66rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 5px;
    }
    .chip-v { font-size: 1.02rem; line-height: 1.1; }
    .side {
      padding: 16px;
      display: grid;
      gap: 12px;
    }
    .metric {
      border-radius: 16px;
      padding: 14px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.03);
    }
    .metric-k {
      color: var(--muted);
      font-size: 0.68rem;
      letter-spacing: 0.11em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }
    .metric-v { font-size: 1.15rem; }
    .warn { color: var(--warn); }
    .alert { color: var(--alert); }
    .footer {
      padding: 14px 18px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .footer a { color: var(--accent); text-decoration: none; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .ribbon { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="top glass">
      <div class="title"><span class="pulse"></span><h1>HSHL Frontend Viewer</h1></div>
      <div class="controls">
        <button id="fs" class="btn" type="button">Fullscreen</button>
        <button id="kiosk" class="btn" type="button">Kiosk</button>
        <button id="dash" class="btn" type="button">Dashboard</button>
      </div>
    </section>

    <section class="layout">
      <article class="stage glass">
        <div class="ribbon">
          <div class="chip"><div class="chip-k">Mode</div><div class="chip-v" id="mode">--</div></div>
          <div class="chip"><div class="chip-k">Map</div><div class="chip-v" id="map">--</div></div>
          <div class="chip"><div class="chip-k">Speed</div><div class="chip-v" id="speed">--</div></div>
          <div class="chip"><div class="chip-k">FPS</div><div class="chip-v" id="fps">--</div></div>
          <div class="chip"><div class="chip-k">Status</div><div class="chip-v" id="status">waiting</div></div>
        </div>
        <img id="feed" src="/stream" alt="simulation stream">
        <div id="overlay">
          <div class="overlay-box">
            <div class="overlay-title" id="ov-title">Connecting</div>
            <div class="overlay-text" id="ov-text">Waiting for live frames from Machine A.</div>
          </div>
        </div>
      </article>

      <aside class="side glass">
        <div class="metric"><div class="metric-k">Wheel</div><div class="metric-v" id="wheel">--</div></div>
        <div class="metric"><div class="metric-k">Recording</div><div class="metric-v" id="rec">--</div></div>
        <div class="metric"><div class="metric-k">Frame Age</div><div class="metric-v" id="f-age">--</div></div>
        <div class="metric"><div class="metric-k">Telemetry Age</div><div class="metric-v" id="u-age">--</div></div>
        <div class="metric"><div class="metric-k">HUD</div><div class="metric-v" id="hud">--</div></div>
      </aside>
    </section>

    <section class="footer glass">
      <span>Private Machine B viewer endpoint</span>
      <span><a href="/display">Open display</a> · <a href="/">Open dashboard</a></span>
    </section>
  </div>

  <script>
    const feed = document.getElementById('feed');
    const overlay = document.getElementById('overlay');
    const ovTitle = document.getElementById('ov-title');
    const ovText = document.getElementById('ov-text');
    let snapshotMode = false;

    function setText(id, text, cls='') {
      const el = document.getElementById(id);
      el.textContent = text;
      if (cls) {
        el.className = 'metric-v ' + cls;
      } else if (el.className.startsWith('metric-v')) {
        el.className = 'metric-v';
      }
    }

    function fmtAge(v) {
      if (v === null || v === undefined) return '--';
      if (v < 1) return '<1s';
      return v.toFixed(1) + 's';
    }

    function fallbackSnapshots() {
      if (snapshotMode) return;
      snapshotMode = true;
      setInterval(() => {
        feed.src = '/frame.jpg?t=' + Date.now();
      }, 100);
    }

    feed.addEventListener('error', fallbackSnapshots);

    document.getElementById('fs').addEventListener('click', () => {
      document.documentElement.requestFullscreen?.().catch(() => {});
    });
    document.getElementById('kiosk').addEventListener('click', () => {
      window.location.href = '/display';
    });
    document.getElementById('dash').addEventListener('click', () => {
      window.location.href = '/';
    });

    function refresh() {
      fetch('/data')
        .then(r => r.json())
        .then(d => {
          const ui = d.ui_state || {};
          const hudEvent = d.hud_event || null;
          const live = d.stream_state === 'live';
          const stale = d.stream_state === 'stale';

          document.getElementById('mode').textContent = ui.mode_text || ui.state || '--';
          document.getElementById('map').textContent = ui.map_label || ui.map_name || '--';
          document.getElementById('speed').textContent = d.speed !== null ? d.speed.toFixed(1) + ' km/h' : '--';
          document.getElementById('fps').textContent = ui.fps !== null && ui.fps !== undefined ? ui.fps.toFixed(1) : '--';
          document.getElementById('status').textContent = d.stream_state || 'waiting';

          setText('wheel', ui.wheel_connected === true ? 'connected' : (ui.wheel_connected === false ? 'missing' : '--'), ui.wheel_connected === false ? 'alert' : '');
          setText('rec', ui.recording ? 'REC ON' : 'IDLE', ui.recording ? 'warn' : '');
          setText('f-age', fmtAge(d.frame_age_s), stale ? 'warn' : '');
          setText('u-age', fmtAge(d.ui_state_age_s), stale ? 'warn' : '');
          setText('hud', hudEvent && hudEvent.text ? hudEvent.text : (ui.mode_text || '--'));

          if (live) {
            overlay.classList.add('hidden');
          } else {
            overlay.classList.remove('hidden');
            ovTitle.textContent = stale ? 'Signal Stale' : 'Connecting';
            ovText.textContent = d.status_message || 'Waiting for live frames from Machine A.';
          }
        })
        .catch(() => {
          overlay.classList.remove('hidden');
          ovTitle.textContent = 'Connection Error';
          ovText.textContent = 'Could not fetch viewer data from local endpoint.';
          document.getElementById('status').textContent = 'error';
        });

      setTimeout(refresh, 500);
    }

    refresh();
  </script>
</body>
</html>
""".encode("utf-8")


class FrameViewer:
    """
    Serve camera frames as a browser-viewable MJPEG stream with a live
    sensor telemetry sidebar.

    Parameters
    ----------
    port : int
        TCP port the HTTP server listens on (default 8080).
    fps  : int
        Maximum frames per second sent to the browser (default 15).
    """

    def __init__(self, port: int = 8080, fps: int = 15):
        self._port = port
        self._delay = 1.0 / max(1, fps)
        self._frame_bytes: bytes = _make_placeholder()
        self._has_real_frame: bool = False
        self._frame_ts: float | None = None
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None

        # Telemetry state (updated via push_telemetry)
        self._telemetry: dict = {
            "speed":     None,
            "imu":       None,
            "gnss":      None,
            "collision": None,
            "lane":      None,
            "ui_state":  None,
            "hud_event": None,
        }
        self._collision_ts: float | None = None
        self._lane_ts:      float | None = None
        self._ui_state_ts:  float | None = None
        self._hud_ts:       float | None = None
        self._tlock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        viewer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):  # silence per-request access logs
                pass

            def do_GET(self):
                if self.path in ("/", ""):
                    self._serve_page(_HTML)
                elif self.path == "/display":
                    self._serve_page(_DISPLAY_HTML)
                elif self.path == "/frontend":
                    self._serve_page(_FRONTEND_HTML)
                elif self.path == "/stream":
                    self._serve_stream()
                elif self.path.startswith("/frame.jpg"):
                    self._serve_frame()
                elif self.path in ("/data", "/status"):
                    self._serve_data()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_page(self, content: bytes):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(content)

            def _serve_data(self):
                now = time.time()
                with viewer._lock:
                    is_ready = viewer._has_real_frame
                    frame_ts = viewer._frame_ts
                with viewer._tlock:
                    t = viewer._telemetry
                    col_ts = viewer._collision_ts
                    lane_ts = viewer._lane_ts
                    ui_state_ts = viewer._ui_state_ts
                    frame_age_s = round(now - frame_ts, 1) if frame_ts is not None else None
                    ui_state_age_s = round(now - ui_state_ts, 1) if ui_state_ts is not None else None
                    if not is_ready:
                        stream_state = "waiting"
                        status_message = "Waiting for camera frames from Machine A..."
                    elif frame_age_s is not None and frame_age_s >= 2.5:
                        stream_state = "stale"
                        status_message = "Camera stream is stale. Check Machine A and the network link."
                    elif ui_state_age_s is not None and ui_state_age_s >= 2.5:
                        stream_state = "stale"
                        status_message = "Runtime state is stale. Images are present but UI updates are old."
                    else:
                        stream_state = "live"
                        status_message = "Live simulation stream"
                    payload = {
                        "ready": is_ready,
                        "stream_state": stream_state,
                        "status_message": status_message,
                        "frame_age_s": frame_age_s,
                        "ui_state_age_s": ui_state_age_s,
                        "speed": t["speed"],
                        "imu":   t["imu"],
                        "gnss":  t["gnss"],
                        "ui_state": t["ui_state"],
                        "hud_event": t["hud_event"],
                        "collision": None,
                        "lane":      None,
                    }
                    if t["collision"] is not None and col_ts is not None:
                        payload["collision"] = {
                            **t["collision"],
                            "ago_s": round(now - col_ts, 1),
                        }
                    if t["lane"] is not None and lane_ts is not None:
                        payload["lane"] = {
                            **t["lane"],
                            "ago_s": round(now - lane_ts, 1),
                        }
                    if t["hud_event"] is not None and viewer._hud_ts is not None:
                        payload["hud_event"] = {
                            **t["hud_event"],
                            "ago_s": round(now - viewer._hud_ts, 1),
                        }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(body)

            def _serve_stream(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()
                try:
                    while True:
                        with viewer._lock:
                            data = viewer._frame_bytes
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                        )
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        time.sleep(viewer._delay)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _serve_frame(self):
                with viewer._lock:
                    data = viewer._frame_bytes
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("0.0.0.0", self._port), _Handler)
        thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="FrameViewer"
        )
        thread.start()
        print(
            f"\n[FrameViewer] Dashboard  -> http://<machine-b-host>:{self._port}/\n"
          f"[FrameViewer] Fullscreen -> http://<machine-b-host>:{self._port}/display\n"
          f"[FrameViewer] Frontend   -> http://<machine-b-host>:{self._port}/frontend\n",
            flush=True,
        )

    def push(self, frame: np.ndarray) -> None:
        """
        Push a BGR frame (numpy array) to the live stream.

        Call this inside your ``process_image`` callback::

            def process_image(self, image):
                self._viewer.push(image)
                # … your own logic below …
        """
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        with self._lock:
            self._frame_bytes = jpeg.tobytes()
            self._has_real_frame = True
            self._frame_ts = time.time()

    def push_telemetry(self, **kwargs) -> None:
        """
        Update one or more telemetry values shown in the browser sidebar.

        Keyword arguments
        -----------------
        speed      : float          Vehicle speed in km/h.
        imu        : dict           Keys: ax, ay, az, gx, gy, gz (floats).
        gnss       : dict           Keys: lat, lon, alt (floats).
        collision  : dict           Keys: actor (str), impulse_mag (float).
        lane       : dict           Keys: types (list[str]).

        Called automatically by CarlaADASInterface when a viewer is registered
        via ``self.register_viewer(viewer)``.  Students can also call it
        manually to push custom data.

        Example::

            self._viewer.push_telemetry(speed=self.current_speed)
        """
        with self._tlock:
            if "speed" in kwargs:
                self._telemetry["speed"] = float(kwargs["speed"])
            if "imu" in kwargs:
                self._telemetry["imu"] = kwargs["imu"]
            if "gnss" in kwargs:
                self._telemetry["gnss"] = kwargs["gnss"]
            if "collision" in kwargs:
                self._telemetry["collision"] = kwargs["collision"]
                self._collision_ts = time.time()
            if "lane" in kwargs:
                self._telemetry["lane"] = kwargs["lane"]
                self._lane_ts = time.time()

    def push_ui_state(self, state: dict) -> None:
        with self._tlock:
            self._telemetry["ui_state"] = state
            self._ui_state_ts = time.time()

    def push_hud_event(self, event: dict) -> None:
        with self._tlock:
            self._telemetry["hud_event"] = event
            self._hud_ts = time.time()

    def stop(self) -> None:
        """Shut down the HTTP server (called automatically on node destroy)."""
        if self._server:
            self._server.shutdown()

