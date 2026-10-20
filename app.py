from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from mind_reader_engine import MindReaderLogisticRegression

app = FastAPI()
engine = MindReaderLogisticRegression(lr=0.38, l2_decay=0.015)


class MoveRequest(BaseModel):
    move: str
    dt_ms: float


def get_top_learned_weights(top_k: int = 7):
    weights = [float(v) for v in engine.w]
    pairs = [
        {"feature": engine.feature_names[i], "weight": round(weights[i], 3)}
        for i in range(engine.n_features)
        if abs(weights[i]) > 0.005
    ]
    pairs.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return pairs[:top_k]


@app.post("/api/step")
def play_step(req: MoveRequest):
    move = req.move.upper()
    if move not in ("L", "R"):
        move = "R"
    result = engine.step(move, dt_ms=req.dt_ms)
    next_lock = engine.predict(dt_ms=250.0)
    result["top_weights"] = get_top_learned_weights(7)
    result["next_lock_conf"] = next_lock["confidence"]
    result["next_z"] = next_lock["log_odds"]
    result["next_p"] = next_lock["prob_right"]
    return result


@app.post("/api/reset")
def reset_game():
    engine.reset()
    return {"status": "reset", "round": 0}


HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KINETIC // Online SGD Logistic Telemetry</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #07080a;
      --text-primary: #f4f1ea;
      --text-muted: rgba(244, 241, 234, 0.52);
      --text-dim: rgba(244, 241, 234, 0.30);
      --jade: #2dd4bf;
      --jade-glow: rgba(45, 212, 191, 0.32);
      --amber: #ff6b35;
      --amber-glow: rgba(255, 107, 53, 0.35);
      --lime: #d4ff33;
      --glass-surface: linear-gradient(135deg, rgba(255, 255, 255, 0.085) 0%, rgba(255, 255, 255, 0.02) 100%);
      --glass-recessed: rgba(6, 7, 10, 0.52);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background-color: var(--bg-base);
      color: var(--text-primary);
      font-family: 'Space Grotesk', sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
      position: relative;
      user-select: none;
      -webkit-font-smoothing: antialiased;
    }

    .mono { font-family: 'JetBrains Mono', monospace; }

    /* Ambient Mesh Light Orbs behind the glass */
    .ambient-canvas {
      position: fixed;
      inset: 0;
      z-index: 0;
      overflow: hidden;
      pointer-events: none;
    }
    .orb {
      position: absolute;
      border-radius: 50%;
      filter: blur(110px);
      opacity: 0.52;
      animation: floatOrb 22s infinite alternate ease-in-out;
    }
    .orb-amber {
      width: 520px; height: 520px;
      background: radial-gradient(circle, #ff6b35 0%, #9a3412 70%, transparent 100%);
      top: -12%; right: 6%;
    }
    .orb-jade {
      width: 560px; height: 560px;
      background: radial-gradient(circle, #14b8a6 0%, #115e59 70%, transparent 100%);
      bottom: -18%; left: 4%;
      animation-delay: -9s;
    }
    .orb-lime {
      width: 340px; height: 340px;
      background: radial-gradient(circle, rgba(212, 255, 51, 0.45) 0%, transparent 70%);
      top: 38%; left: 42%;
      animation-delay: -14s;
    }

    @keyframes floatOrb {
      0% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(-40px, 30px) scale(1.08); }
      100% { transform: translate(35px, -25px) scale(0.95); }
    }

    /* Tactile Optical Noise Overlay */
    .grain {
      position: fixed;
      inset: 0;
      z-index: 1;
      pointer-events: none;
      opacity: 0.04;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    }

    /* Authentic Multi-Layer Glassmorphism */
    .glass-panel {
      background: var(--glass-surface);
      backdrop-filter: blur(30px) saturate(185%);
      -webkit-backdrop-filter: blur(30px) saturate(185%);
      border-top: 1px solid rgba(255, 255, 255, 0.22);
      border-left: 1px solid rgba(255, 255, 255, 0.14);
      border-right: 1px solid rgba(255, 255, 255, 0.04);
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      box-shadow:
        inset 0 1px 0 0 rgba(255, 255, 255, 0.16),
        0 24px 48px -12px rgba(0, 0, 0, 0.72);
      border-radius: 22px;
    }

    .glass-well {
      background: var(--glass-recessed);
      border: 1px solid rgba(255, 255, 255, 0.07);
      box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.65);
      border-radius: 14px;
    }

    .shell {
      position: relative;
      z-index: 2;
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 24px 48px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Top Bar */
    .topbar {
      padding: 18px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .status-led {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: var(--lime);
      box-shadow: 0 0 14px var(--lime);
    }
    .brand-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .brand-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      letter-spacing: 0.03em;
    }
    .topbar-controls {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .lock-pill {
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      background: rgba(212, 255, 51, 0.08);
      border: 1px solid rgba(212, 255, 51, 0.28);
      color: var(--lime);
    }
    .btn-ghost {
      padding: 8px 16px;
      border-radius: 999px;
      font-family: 'Space Grotesk', sans-serif;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--text-primary);
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.15);
      cursor: pointer;
      transition: all 0.18s ease;
    }
    .btn-ghost:hover {
      background: rgba(255, 255, 255, 0.14);
      border-color: rgba(255, 255, 255, 0.3);
    }

    /* Telemetry Strip */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
    }
    @media (max-width: 820px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
    .kpi-card {
      padding: 18px 22px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 6px;
    }
    .kpi-label {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
    }
    .kpi-value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.1;
    }

    /* Main Workspace Grid */
    .workspace {
      display: grid;
      grid-template-columns: 1.45fr 1fr;
      gap: 20px;
    }
    @media (max-width: 960px) { .workspace { grid-template-columns: 1fr; } }

    .arena-panel {
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    /* Outcome Readout Banner */
    .readout-box {
      padding: 18px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      transition: all 0.22s ease;
    }
    .readout-box.state-intercepted {
      background: linear-gradient(90deg, rgba(255, 107, 53, 0.16), rgba(6, 7, 10, 0.55));
      border-color: rgba(255, 107, 53, 0.45);
    }
    .readout-box.state-evaded {
      background: linear-gradient(90deg, rgba(45, 212, 191, 0.16), rgba(6, 7, 10, 0.55));
      border-color: rgba(45, 212, 191, 0.45);
    }
    .readout-tag {
      font-size: 0.68rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .readout-headline {
      font-size: 1.15rem;
      font-weight: 600;
    }
    .readout-meta {
      text-align: right;
      font-size: 0.74rem;
      color: var(--text-muted);
    }

    /* Sigmoid Tug of War + Live Curve */
    .probability-section {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .prob-labels {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 0.78rem;
    }
    .dual-meter {
      height: 14px;
      border-radius: 999px;
      overflow: hidden;
      display: flex;
      position: relative;
      padding: 2px;
    }
    .bar-jade {
      background: linear-gradient(90deg, #0d9488, var(--jade));
      box-shadow: 0 0 16px var(--jade-glow);
      height: 100%;
      border-radius: 999px 0 0 999px;
      transition: width 0.22s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .bar-amber {
      background: linear-gradient(90deg, var(--amber), #fb923c);
      box-shadow: 0 0 16px var(--amber-glow);
      height: 100%;
      border-radius: 0 999px 999px 0;
      transition: width 0.22s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .center-tick {
      position: absolute;
      top: 0; bottom: 0; left: 50%;
      width: 2px;
      background: rgba(255, 255, 255, 0.75);
      transform: translateX(-50%);
    }

    /* Physical Glass Keycaps */
    .keycap-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .keycap {
      position: relative;
      padding: 26px 20px;
      border-radius: 18px;
      background: linear-gradient(160deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.02) 100%);
      border-top: 1px solid rgba(255,255,255,0.28);
      border-left: 1px solid rgba(255,255,255,0.16);
      border-right: 1px solid rgba(255,255,255,0.05);
      border-bottom: 4px solid rgba(0,0,0,0.65);
      box-shadow: 0 14px 28px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.2);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      transition: transform 0.08s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .keycap-left:hover, .keycap-left.pressed {
      border-color: var(--jade);
      box-shadow: 0 8px 24px var(--jade-glow), inset 0 0 18px rgba(45, 212, 191, 0.14);
    }
    .keycap-right:hover, .keycap-right.pressed {
      border-color: var(--amber);
      box-shadow: 0 8px 24px var(--amber-glow), inset 0 0 18px rgba(255, 107, 53, 0.14);
    }
    .keycap.pressed {
      transform: translateY(3px);
      border-bottom-width: 1px;
    }
    .keycap-dir {
      font-size: 1.55rem;
      font-weight: 700;
      letter-spacing: 0.06em;
    }
    .keycap-hint {
      font-size: 0.7rem;
      color: var(--text-muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    /* Attribution Chips */
    .chips-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      min-height: 32px;
    }
    .chip {
      padding: 6px 11px;
      border-radius: 8px;
      font-size: 0.72rem;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
    }
    .chip-jade {
      border-color: rgba(45, 212, 191, 0.4);
      color: var(--jade);
      background: rgba(45, 212, 191, 0.09);
    }
    .chip-amber {
      border-color: rgba(255, 107, 53, 0.4);
      color: #ff8c61;
      background: rgba(255, 107, 53, 0.09);
    }

    /* Right Telemetry Panel */
    .telemetry-panel {
      padding: 26px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 20px;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .section-title {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
    }

    /* Diverging Weight Bars */
    .weight-rows {
      display: flex;
      flex-direction: column;
      gap: 9px;
    }
    .weight-row {
      display: grid;
      grid-template-columns: 135px 1fr 54px;
      align-items: center;
      gap: 10px;
      font-size: 0.74rem;
    }
    .weight-track {
      height: 8px;
      background: rgba(0, 0, 0, 0.5);
      border-radius: 999px;
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .weight-zero {
      position: absolute;
      left: 50%; top: 0; bottom: 0;
      width: 1px;
      background: rgba(255, 255, 255, 0.28);
    }
    .weight-fill-left {
      position: absolute;
      top: 0; bottom: 0;
      right: 50%;
      background: var(--jade);
      border-radius: 999px 0 0 999px;
    }
    .weight-fill-right {
      position: absolute;
      top: 0; bottom: 0;
      left: 50%;
      background: var(--amber);
      border-radius: 0 999px 999px 0;
    }

    /* Sequence Tape */
    .tape-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      min-height: 28px;
    }
    .tape-node {
      width: 26px; height: 26px;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.7rem;
      font-weight: 600;
    }
    .tape-hit {
      background: rgba(255, 107, 53, 0.18);
      border: 1px solid rgba(255, 107, 53, 0.5);
      color: #ff8c61;
    }
    .tape-miss {
      background: rgba(45, 212, 191, 0.16);
      border: 1px solid rgba(45, 212, 191, 0.45);
      color: var(--jade);
    }
  </style>
</head>
<body>

  <!-- Background Refracted Light Field -->
  <div class="ambient-canvas">
    <div class="orb orb-amber"></div>
    <div class="orb orb-jade"></div>
    <div class="orb orb-lime"></div>
  </div>
  <div class="grain"></div>

  <main class="shell">
    <!-- Top Glass Bar -->
    <header class="glass-panel topbar">
      <div class="brand">
        <span class="status-led"></span>
        <div>
          <div class="brand-title">KINETIC // SGD TELEMETRY</div>
          <div class="brand-sub">Online Logistic Regression • Real-Time N-Gram & Motor Rhythm Interceptor</div>
        </div>
      </div>
      <div class="topbar-controls">
        <div id="lockBadge" class="lock-pill mono">PRE-COMMIT LOCK // TURN #001</div>
        <button onclick="resetGame()" class="btn-ghost">Zero Weights</button>
      </div>
    </header>

    <!-- KPI Telemetry Strip -->
    <section class="kpi-grid">
      <div class="glass-panel kpi-card">
        <span class="kpi-label">Sequence Step</span>
        <div id="roundNum" class="kpi-value mono">000 <span style="font-size:0.9rem;color:var(--text-dim)">/ 100</span></div>
      </div>
      <div class="glass-panel kpi-card">
        <span class="kpi-label">Model Intercept Rate</span>
        <div id="winRate" class="kpi-value mono" style="color:var(--lime)">50.0%</div>
      </div>
      <div class="glass-panel kpi-card">
        <span class="kpi-label">AI Intercepts</span>
        <div id="aiScore" class="kpi-value mono" style="color:var(--amber)">0</div>
      </div>
      <div class="glass-panel kpi-card">
        <span class="kpi-label">Human Evasions</span>
        <div id="humanScore" class="kpi-value mono" style="color:var(--jade)">0</div>
      </div>
    </section>

    <!-- Main Split Workspace -->
    <section class="workspace">
      <!-- Left Column: Interactive Duel Arena -->
      <div class="glass-panel arena-panel">

        <div id="turnBanner" class="glass-well readout-box">
          <div>
            <div id="bannerTag" class="readout-tag mono">AWAITING INPUT STREAM</div>
            <div id="bannerTitle" class="readout-headline">Use Left [←] and Right [→] keys to test unpredictability</div>
          </div>
          <div class="readout-meta mono">
            <div id="rhythmText">Δt: -- ms</div>
            <div id="gradText">∇L: 0.000</div>
          </div>
        </div>

        <!-- Sigmoid Probability Bar -->
        <div class="probability-section">
          <div class="prob-labels mono">
            <span id="probLeftText" style="color:var(--jade);font-weight:600">P(LEFT) 50.0%</span>
            <span id="logOddsText" style="color:var(--text-muted)">z = 0.000</span>
            <span id="probRightText" style="color:var(--amber);font-weight:600">P(RIGHT) 50.0%</span>
          </div>
          <div class="glass-well dual-meter">
            <div id="leftBar" class="bar-jade" style="width: 50%"></div>
            <div id="rightBar" class="bar-amber" style="width: 50%"></div>
            <div class="center-tick"></div>
          </div>
        </div>

        <!-- Tactile Frosted Keycaps -->
        <div class="keycap-grid">
          <button id="keyL" onclick="sendMove('L')" class="keycap keycap-left">
            <span class="keycap-dir" style="color:var(--jade)">← LEFT</span>
            <span class="keycap-hint mono">Key [←] or [A] • y = 0</span>
          </button>
          <button id="keyR" onclick="sendMove('R')" class="keycap keycap-right">
            <span class="keycap-dir" style="color:var(--amber)">RIGHT →</span>
            <span class="keycap-hint mono">Key [→] or [D] • y = 1</span>
          </button>
        </div>

        <!-- Active Feature Attributions -->
        <div>
          <div class="section-header">
            <span class="section-title">Active Log-Odds Decomposition (βᵢ · xᵢ)</span>
            <span class="mono" style="font-size:0.68rem;color:var(--text-dim)">PRE-KEYSTROKE STATE</span>
          </div>
          <div id="reasonChips" class="chips-wrap mono">
            <span class="chip" style="color:var(--text-dim)">Zero-state initialization — awaiting first transitions</span>
          </div>
        </div>

      </div>

      <!-- Right Column: Sigmoid Curve & Diverging Weight Spectrum -->
      <div class="glass-panel telemetry-panel">

        <!-- Live SVG Sigmoid Operating Point -->
        <div>
          <div class="section-header">
            <span class="section-title">Live Sigmoid Manifold σ(z) = 1 / (1 + e⁻ᶻ)</span>
            <span id="curveCoord" class="mono" style="font-size:0.7rem;color:var(--lime)">(z: 0.00, p: 0.50)</span>
          </div>
          <div class="glass-well" style="padding: 12px 16px;">
            <svg viewBox="0 0 320 110" width="100%" height="110" style="overflow:visible">
              <line x1="0" y1="55" x2="320" y2="55" stroke="rgba(255,255,255,0.1)" stroke-dasharray="3 3"/>
              <line x1="160" y1="5" x2="160" y2="105" stroke="rgba(255,255,255,0.14)"/>
              <path d="M 0 102 C 80 102, 110 95, 160 55 C 210 15, 240 8, 320 8" fill="none" stroke="rgba(244,241,234,0.38)" stroke-width="2"/>
              <circle id="sigmoidDot" cx="160" cy="55" r="6" fill="#d4ff33" style="filter: drop-shadow(0 0 8px #d4ff33); transition: all 0.22s ease;"/>
            </svg>
          </div>
        </div>

        <!-- Diverging Weight Spectrum -->
        <div>
          <div class="section-header">
            <span class="section-title">Learned Weight Spectrum (w)</span>
            <span class="mono" style="font-size:0.66rem;color:var(--text-dim)">← LEFT BIAS | RIGHT BIAS →</span>
          </div>
          <div id="weightsList" class="weight-rows mono">
            <div style="color:var(--text-dim);font-size:0.74rem;padding:14px 0;">No non-zero weights learned yet.</div>
          </div>
        </div>

        <!-- Sequence History Tape -->
        <div>
          <div class="section-header">
            <span class="section-title">Recent Keystroke Telemetry</span>
            <span class="mono" style="font-size:0.66rem;color:var(--text-dim)">AMBER = AI HIT • JADE = EVADED</span>
          </div>
          <div id="moveTape" class="tape-grid mono"></div>
        </div>

      </div>
    </section>
  </main>

  <script>
    let lastTapTime = performance.now();
    let aiWins = 0;
    let humanWins = 0;
    let busy = false;

    const keyL = document.getElementById('keyL');
    const keyR = document.getElementById('keyR');

    document.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        keyL.classList.add('pressed');
        sendMove('L');
      }
      if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        keyR.classList.add('pressed');
        sendMove('R');
      }
    });

    document.addEventListener('keyup', (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keyL.classList.remove('pressed');
      if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keyR.classList.remove('pressed');
    });

    async function sendMove(move) {
      if (busy) return;
      busy = true;
      const now = performance.now();
      const dt_ms = Math.min(2000, Math.max(30, now - lastTapTime));
      lastTapTime = now;

      const res = await fetch('/api/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move, dt_ms })
      });
      const d = await res.json();
      busy = false;
      updateUI(d, Math.round(dt_ms));
    }

    function updateUI(d, dt_ms) {
      if (d.ai_correct) aiWins++; else humanWins++;

      const rStr = String(d.round).padStart(3, '0');
      document.getElementById('roundNum').innerHTML = `${rStr} <span style="font-size:0.9rem;color:var(--text-dim)">/ 100</span>`;
      document.getElementById('winRate').textContent = (d.ai_win_rate * 100).toFixed(1) + '%';
      document.getElementById('aiScore').textContent = aiWins;
      document.getElementById('humanScore').textContent = humanWins;

      const nextConf = (d.next_lock_conf * 100).toFixed(0);
      const nextTurn = String(d.round + 1).padStart(3, '0');
      document.getElementById('lockBadge').textContent = `LOCKED // TURN #${nextTurn} (${nextConf}% CONF)`;

      const banner = document.getElementById('turnBanner');
      const tag = document.getElementById('bannerTag');
      const title = document.getElementById('bannerTitle');

      if (d.ai_correct) {
        banner.className = 'glass-well readout-box state-intercepted';
        tag.textContent = 'INTERCEPTED // MODEL MATCH';
        tag.style.color = 'var(--amber)';
        title.innerHTML = `Input [${d.actual_move}] matched pre-committed prediction [${d.ai_prediction}]`;
      } else {
        banner.className = 'glass-well readout-box state-evaded';
        tag.textContent = 'EVADED // PREDICTION MISS';
        tag.style.color = 'var(--jade)';
        title.innerHTML = `Input [${d.actual_move}] broke pre-committed prediction [${d.ai_prediction}]`;
      }

      document.getElementById('rhythmText').textContent = `Δt: ${dt_ms} ms`;
      document.getElementById('gradText').textContent = `p - y: ${d.error > 0 ? '+' : ''}${d.error}`;

      const pRight = (d.prob_right * 100).toFixed(1);
      const pLeft = (100 - d.prob_right * 100).toFixed(1);
      document.getElementById('probLeftText').textContent = `P(LEFT) ${pLeft}%`;
      document.getElementById('probRightText').textContent = `P(RIGHT) ${pRight}%`;
      document.getElementById('logOddsText').textContent = `z = ${d.log_odds > 0 ? '+' : ''}${d.log_odds}`;
      document.getElementById('leftBar').style.width = `${pLeft}%`;
      document.getElementById('rightBar').style.width = `${pRight}%`;

      // Update Sigmoid SVG Dot
      const zClamped = Math.max(-3.5, Math.min(3.5, d.log_odds));
      const cx = 160 + (zClamped / 3.5) * 145;
      const cy = 105 - (d.prob_right * 95);
      const dot = document.getElementById('sigmoidDot');
      dot.setAttribute('cx', cx.toFixed(1));
      dot.setAttribute('cy', cy.toFixed(1));
      dot.setAttribute('fill', d.prob_right >= 0.5 ? '#ff6b35' : '#2dd4bf');
      document.getElementById('curveCoord').textContent = `(z: ${d.log_odds > 0 ? '+' : ''}${d.log_odds}, p: ${d.prob_right.toFixed(2)})`;

      // Active attribution chips
      const chipsDiv = document.getElementById('reasonChips');
      if (!d.reasons || d.reasons.length === 0) {
        chipsDiv.innerHTML = `<span class="chip" style="color:var(--text-dim)">Cold start — collecting initial n-gram state</span>`;
      } else {
        chipsDiv.innerHTML = d.reasons.map(r => {
          const cls = r.pushes_toward === 'R' ? 'chip chip-amber' : 'chip chip-jade';
          const sign = r.log_odds > 0 ? '+' : '';
          return `<span class="${cls}">${r.feature} → ${r.pushes_toward} (${sign}${r.log_odds})</span>`;
        }).join('');
      }

      // Diverging weight spectrum bars
      const wList = document.getElementById('weightsList');
      if (d.top_weights && d.top_weights.length > 0) {
        wList.innerHTML = d.top_weights.map(w => {
          const isPos = w.weight > 0;
          const pct = Math.min(50, Math.abs(w.weight) * 38);
          const barHtml = isPos
            ? `<div class="weight-fill-right" style="width:${pct}%"></div>`
            : `<div class="weight-fill-left" style="width:${pct}%"></div>`;
          const valColor = isPos ? 'var(--amber)' : 'var(--jade)';
          return `
            <div class="weight-row">
              <span style="color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${w.feature}</span>
              <div class="weight-track">
                <div class="weight-zero"></div>
                ${barHtml}
              </div>
              <span style="text-align:right;color:${valColor};font-weight:600">${isPos ? '+' : ''}${w.weight}</span>
            </div>
          `;
        }).join('');
      }

      // Sequence tape
      const tape = document.getElementById('moveTape');
      const nodeCls = d.ai_correct ? 'tape-node tape-hit' : 'tape-node tape-miss';
      tape.insertAdjacentHTML('afterbegin', `<div class="${nodeCls}">${d.actual_move}</div>`);
      while (tape.children.length > 20) tape.lastElementChild.remove();
    }

    async function resetGame() {
      await fetch('/api/reset', { method: 'POST' });
      aiWins = 0; humanWins = 0;
      document.getElementById('roundNum').innerHTML = `000 <span style="font-size:0.9rem;color:var(--text-dim)">/ 100</span>`;
      document.getElementById('winRate').textContent = '50.0%';
      document.getElementById('aiScore').textContent = '0';
      document.getElementById('humanScore').textContent = '0';
      document.getElementById('moveTape').innerHTML = '';
      document.getElementById('weightsList').innerHTML = '<div style="color:var(--text-dim);font-size:0.74rem;padding:14px 0;">Weights zeroed. Press [←] or [→] to begin.</div>';
      document.getElementById('bannerTag').textContent = 'WEIGHTS ZEROED';
      document.getElementById('bannerTitle').textContent = 'All 35 logistic weights reset to 0.000';
      document.getElementById('sigmoidDot').setAttribute('cx', '160');
      document.getElementById('sigmoidDot').setAttribute('cy', '55');
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_UI


if __name__ == "__main__":
    print("KINETIC Glass UI running at http://localhost:8001")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")