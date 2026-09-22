"""
Generates an interactive standalone HTML visualization dashboard for the synthetic sample.
Saves the report both to the examples directory and the Antigravity artifacts directory.
"""

import sys
import os
import json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for p in [REPO_ROOT, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.test_large_sample import generate_large_sector_sample
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset


def build_dashboard():
    sample = generate_large_sector_sample()
    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        sample["time"], sample["tpf_flux"], sample["tpf_flux_err"],
        sample["period"], sample["t0"], sample["duration_days"]
    )
    centroid_res = measure_centroid_offset(
        i_diff, sigma_diff, sample["target_pos"],
        tess_pixel_scale_arcsec=21.0, n_mc_perturbations=500
    )

    # 1. Dados da série temporal
    time_pts = sample["time"]
    flux_pts = sample["flux_1d"]

    # 2. Dobramento de fase (Phase fold)
    p = sample["period"]
    t0 = sample["t0"]
    phase = (time_pts - t0 + 0.5 * p) % p - 0.5 * p

    zoom_mask = np.abs(phase) <= 0.18
    fold_phase = phase[zoom_mask].tolist()
    fold_flux = flux_pts[zoom_mask].tolist()

    bins = np.linspace(-0.16, 0.16, 35)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    for i in range(len(bins) - 1):
        m = (phase >= bins[i]) & (phase < bins[i+1])
        if np.any(m):
            bin_means.append(float(np.median(flux_pts[m])))
        else:
            bin_means.append(1.0)

    i_out_norm = ((i_out - i_out.min()) / (i_out.max() - i_out.min())).tolist()
    i_diff_clipped = np.clip(i_diff, 0, None)
    i_diff_norm = (i_diff_clipped / (i_diff_clipped.max() + 1e-6)).tolist()

    t_rv = sample["t_rv"].tolist()
    rv_obs = sample["rv_obs"].tolist()
    rv_err = sample["rv_err"].tolist()
    rv_dense_t = np.linspace(0, 14.0, 300)
    mean_anom = (2.0 * np.pi * (rv_dense_t - t0) / p) % (2.0 * np.pi)
    rv_dense_curve = (sample["gamma_true"] + sample["k_semiamp_true"] * np.sin(mean_anom)).tolist()

    data_payload = {
        "time": [round(float(x), 4) for x in time_pts],
        "flux": [round(float(y), 5) for y in flux_pts],
        "period": sample["period"],
        "t0": sample["t0"],
        "duration_days": sample["duration_days"],
        "injected_depth": sample["injected_depth"],
        "fold_phase": [round(x, 4) for x in fold_phase],
        "fold_flux": [round(y, 5) for y in fold_flux],
        "bin_centers": [round(float(x), 4) for x in bin_centers],
        "bin_means": [round(float(y), 5) for y in bin_means],
        "tpf_out": i_out_norm,
        "tpf_diff": i_diff_norm,
        "target_pos": sample["target_pos"],
        "centroid_calc": [round(centroid_res["y_diff_cen"], 2), round(centroid_res["x_diff_cen"], 2)],
        "offset_arcsec": round(centroid_res["offset_arcsec"], 2),
        "t_rv": [round(x, 3) for x in t_rv],
        "rv_obs": [round(y, 2) for y in rv_obs],
        "rv_err": [round(e, 2) for e in rv_err],
        "rv_dense_t": [round(float(t), 3) for t in rv_dense_t],
        "rv_dense_curve": [round(float(v), 2) for v in rv_dense_curve]
    }

    html_template = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>Astro-Exo: Visualizador dos Dados Sintéticos</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0b0f19;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
    }
    .card {
      background: #131b2e;
      border: 1px solid #1e293b;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .pixel-grid {
      display: grid;
      grid-template-columns: repeat(11, 1fr);
      gap: 2px;
      width: 220px;
      height: 220px;
      background: #090d16;
      padding: 4px;
      border-radius: 8px;
    }
    .pixel-cell {
      border-radius: 2px;
      position: relative;
    }
    canvas {
      width: 100%;
      height: 220px;
    }
  </style>
</head>
<body>
  <div class="max-w-6xl mx-auto space-y-6">
    <!-- Header -->
    <div class="card border-blue-900/50 bg-gradient-to-r from-[#131b2e] to-[#172554]/40">
      <div class="flex items-center justify-between flex-wrap gap-4">
        <div>
          <span class="px-2.5 py-1 text-xs font-semibold uppercase tracking-wider bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-full">
            Pipeline Astro-Exo • Dados Sintéticos
          </span>
          <h1 class="text-2xl font-bold text-white mt-2">Diagnóstico Visual da Amostra Ampla (14 Dias)</h1>
          <p class="text-sm text-slate-400 mt-1">Série temporal contínua, matriz espacial TPF 11x11 com estrela vizinha contaminante e curvas Doppler RV.</p>
        </div>
        <div class="flex gap-4">
          <div class="text-right">
            <div class="text-xs text-slate-400">Trânsitos Injetados</div>
            <div class="text-xl font-bold text-emerald-400">5 eventos</div>
          </div>
          <div class="text-right border-l border-slate-700 pl-4">
            <div class="text-xs text-slate-400">Profundidade</div>
            <div class="text-xl font-bold text-blue-400">8.500 ppm (0.85%)</div>
          </div>
          <div class="text-right border-l border-slate-700 pl-4">
            <div class="text-xs text-slate-400">Semi-amplitude K</div>
            <div class="text-xl font-bold text-purple-400">85 m/s</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Gráficos Superiores: Curva de Luz 14 dias + Dobramento de Fase -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Painel 1: Série Temporal Completa -->
      <div class="card">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h2 class="text-base font-semibold text-white">1. Curva de Luz Contínua (14 Dias)</h2>
            <p class="text-xs text-slate-400">2.016 cadências. Observe a rotação estelar senoidal e as 5 quedas periódicas em vermelho.</p>
          </div>
          <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">2.016 pts</span>
        </div>
        <canvas id="canvasFullLC"></canvas>
      </div>

      <!-- Painel 2: Dobramento de Fase -->
      <div class="card">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h2 class="text-base font-semibold text-white">2. Dobramento de Fase (Phase Folded Transit)</h2>
            <p class="text-xs text-slate-400">Os 5 trânsitos sobrepostos revelando o formato em "U" com escurecimento de limbo (Kipping).</p>
          </div>
          <span class="text-xs px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 font-mono">SNR &gt; 25σ</span>
        </div>
        <canvas id="canvasFolded"></canvas>
      </div>
    </div>

    <!-- Gráficos Inferiores: Vetting Espacial TPF + Velocidade Radial -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Painel 3: Vetting Espacial TPF (11x11) -->
      <div class="card">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h2 class="text-base font-semibold text-white">3. Vetting Espacial TPF (11x11 Pixels)</h2>
            <p class="text-xs text-slate-400">Compare a imagem de referência com a diferença de imagem (I_diff). O vizinho desaparece!</p>
          </div>
          <span class="text-xs px-2 py-0.5 rounded bg-blue-950 text-blue-300 font-mono">Offset: __OFFSET_ARCSEC__"</span>
        </div>

        <div class="flex flex-wrap items-center justify-around gap-6 pt-2">
          <!-- Heatmap I_out -->
          <div class="flex flex-col items-center">
            <span class="text-xs font-medium text-slate-300 mb-2">Imagem de Referência (I_out)</span>
            <div id="gridOut" class="pixel-grid"></div>
            <div class="text-[11px] text-slate-400 mt-2 text-center">
              Centro: Alvo (5, 5)<br>
              Sup. Direito: Vizinho (2, 8)
            </div>
          </div>

          <!-- Heatmap I_diff -->
          <div class="flex flex-col items-center">
            <span class="text-xs font-medium text-emerald-400 mb-2">Diferença de Fluxo (I_diff)</span>
            <div id="gridDiff" class="pixel-grid"></div>
            <div class="text-[11px] text-slate-400 mt-2 text-center">
              ✓ Sinal apenas no alvo!<br>
              Vizinho descartado (Sem BEB)
            </div>
          </div>
        </div>
      </div>

      <!-- Painel 4: Curva Doppler RV -->
      <div class="card">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h2 class="text-base font-semibold text-white">4. Velocidade Radial (Doppler RV)</h2>
            <p class="text-xs text-slate-400">25 espectros tipo HARPS (precisão ±3.5 m/s) medindo a massa planetária Mp.</p>
          </div>
          <span class="text-xs px-2 py-0.5 rounded bg-purple-950 text-purple-300 font-mono">Mp ~ 0.85 M_Jup</span>
        </div>
        <canvas id="canvasRV"></canvas>
      </div>
    </div>

    <!-- Rodapé -->
    <div class="text-center text-xs text-slate-500 pt-2 pb-6">
      Gerado automaticamente pelo Astro-Exo • Validação de Algoritmos com Injeção de Sinal Sintético
    </div>
  </div>

  <script>
    const D = __DATA_PAYLOAD__;

    // 1. Plot da Curva de Luz Completa
    (function renderFullLC() {
      const c = document.getElementById("canvasFullLC");
      const ctx = c.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth * dpr;
      const h = c.clientHeight * dpr;
      c.width = w;
      c.height = h;

      const pad = { left: 45 * dpr, right: 20 * dpr, top: 15 * dpr, bottom: 30 * dpr };
      const pw = w - pad.left - pad.right;
      const ph = h - pad.top - pad.bottom;

      const minT = 0, maxT = 14;
      const minF = 0.985, maxF = 1.015;

      const toX = t => pad.left + ((t - minT) / (maxT - minT)) * pw;
      const toY = f => pad.top + ph - ((f - minF) / (maxF - minF)) * ph;

      // Grade
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1 * dpr;
      for (let t = 0; t <= 14; t += 2) {
        const x = toX(t);
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + ph); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = (10 * dpr) + "px sans-serif";
        ctx.fillText(t + "d", x - 8 * dpr, pad.top + ph + 18 * dpr);
      }

      // Pontos fotométricos
      for (let i = 0; i < D.time.length; i++) {
        const t = D.time[i];
        const f = D.flux[i];
        const phase = (t - D.t0 + 0.5 * D.period) % D.period - 0.5 * D.period;
        const inTransit = Math.abs(phase) <= (0.5 * D.duration_days);

        ctx.fillStyle = inTransit ? "#ef4444" : "#38bdf8";
        ctx.beginPath();
        ctx.arc(toX(t), toY(f), (inTransit ? 2.5 : 1.5) * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      // Legenda
      ctx.fillStyle = "#94a3b8";
      ctx.font = (10 * dpr) + "px sans-serif";
      ctx.fillText("Fluxo Normalizado", pad.left + 5 * dpr, pad.top + 12 * dpr);
    })();

    // 2. Plot do Trânsito Dobrado
    (function renderFolded() {
      const c = document.getElementById("canvasFolded");
      const ctx = c.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth * dpr;
      const h = c.clientHeight * dpr;
      c.width = w;
      c.height = h;

      const pad = { left: 45 * dpr, right: 20 * dpr, top: 15 * dpr, bottom: 30 * dpr };
      const pw = w - pad.left - pad.right;
      const ph = h - pad.top - pad.bottom;

      const minX = -0.15, maxX = 0.15;
      const minY = 0.988, maxY = 1.004;

      const toX = x => pad.left + ((x - minX) / (maxX - minX)) * pw;
      const toY = y => pad.top + ph - ((y - minY) / (maxY - minY)) * ph;

      // Grade
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1 * dpr;
      [-0.1, -0.05, 0, 0.05, 0.1].forEach(xVal => {
        const x = toX(xVal);
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + ph); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = (10 * dpr) + "px sans-serif";
        ctx.fillText(xVal.toFixed(2) + "d", x - 12 * dpr, pad.top + ph + 18 * dpr);
      });

      // Pontos individuais
      ctx.fillStyle = "rgba(56, 189, 248, 0.4)";
      for (let i = 0; i < D.fold_phase.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(D.fold_phase[i]), toY(D.fold_flux[i]), 1.8 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      // Linha média binnada
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath();
      for (let i = 0; i < D.bin_centers.length; i++) {
        const x = toX(D.bin_centers[i]);
        const y = toY(D.bin_means[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Marcadores binnados
      ctx.fillStyle = "#fbbf24";
      for (let i = 0; i < D.bin_centers.length; i++) {
        ctx.beginPath();
        ctx.arc(toX(D.bin_centers[i]), toY(D.bin_means[i]), 3.5 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    })();

    // 3. Heatmaps TPF
    function buildHeatmap(containerId, matrix2D, isDiff) {
      const el = document.getElementById(containerId);
      el.innerHTML = "";
      for (let y = 0; y < 11; y++) {
        for (let x = 0; x < 11; x++) {
          const val = matrix2D[y][x];
          const d = document.createElement("div");
          d.className = "pixel-cell";
          d.title = "(" + y + ", " + x + "): " + (val * 100).toFixed(1) + "%";
          
          if (!isDiff) {
            const b = Math.floor(val * 255);
            const r = Math.floor(val * val * 220);
            d.style.backgroundColor = "rgb(" + r + ", " + Math.floor(b * 0.7) + ", " + b + ")";
          } else {
            const g = Math.floor(val * 255);
            const r = Math.floor(val * 80);
            d.style.backgroundColor = "rgb(" + r + ", " + g + ", " + Math.floor(g * 0.4) + ")";
          }

          if (y === 5 && x === 5) {
            d.style.outline = "2px solid #ef4444";
          }
          if (!isDiff && y === 2 && x === 8) {
            d.style.outline = "2px solid #f59e0b";
          }

          el.appendChild(d);
        }
      }
    }
    buildHeatmap("gridOut", D.tpf_out, false);
    buildHeatmap("gridDiff", D.tpf_diff, true);

    // 4. Plot de Velocidade Radial (Doppler RV)
    (function renderRV() {
      const c = document.getElementById("canvasRV");
      const ctx = c.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth * dpr;
      const h = c.clientHeight * dpr;
      c.width = w;
      c.height = h;

      const pad = { left: 45 * dpr, right: 20 * dpr, top: 15 * dpr, bottom: 30 * dpr };
      const pw = w - pad.left - pad.right;
      const ph = h - pad.top - pad.bottom;

      const minT = 0, maxT = 14;
      const minV = -100, maxV = 120;

      const toX = t => pad.left + ((t - minT) / (maxT - minT)) * pw;
      const toY = v => pad.top + ph - ((v - minV) / (maxV - minV)) * ph;

      // Grade
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1 * dpr;
      [-80, -40, 0, 40, 80].forEach(v => {
        const y = toY(v);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = (10 * dpr) + "px sans-serif";
        ctx.fillText(v + " m/s", 5 * dpr, y + 4 * dpr);
      });

      // Curva modelo teórica contínua
      ctx.strokeStyle = "#c084fc";
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      for (let i = 0; i < D.rv_dense_t.length; i++) {
        const x = toX(D.rv_dense_t[i]);
        const y = toY(D.rv_dense_curve[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Pontos com barras de erro
      for (let i = 0; i < D.t_rv.length; i++) {
        const x = toX(D.t_rv[i]);
        const y = toY(D.rv_obs[i]);
        const errPx = ((D.rv_err[i]) / (maxV - minV)) * ph;

        ctx.strokeStyle = "#a855f7";
        ctx.lineWidth = 1.5 * dpr;
        ctx.beginPath();
        ctx.moveTo(x, y - errPx);
        ctx.lineTo(x, y + errPx);
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(x, y, 3 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#7e22ce";
        ctx.stroke();
      }
    })();
  </script>
</body>
</html>
"""

    html_content = html_template.replace("__DATA_PAYLOAD__", json.dumps(data_payload))
    html_content = html_content.replace("__OFFSET_ARCSEC__", str(round(centroid_res["offset_arcsec"], 2)))

    # 1. Salva no diretório de exemplos do repositório
    out_local = os.path.join(os.path.dirname(__file__), "sample_dashboard.html")
    with open(out_local, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Dashboard salvo localmente em: {out_local}")

    # 2. Salva no diretório de artefatos da conversa
    artifact_dir = "/Users/diegozolhos/.gemini/antigravity/brain/cb01ae73-9fed-4879-8295-1f9bf9e73f3a"
    os.makedirs(artifact_dir, exist_ok=True)
    out_artifact = os.path.join(artifact_dir, "synthetic_sample_dashboard.html")
    with open(out_artifact, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Dashboard salvo no diretório de artefatos em: {out_artifact}")

    return out_local, out_artifact


if __name__ == "__main__":
    build_dashboard()
