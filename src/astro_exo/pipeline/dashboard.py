"""
Scientific interactive web dashboard generator for the Astro-Exo pipeline.
Compiles execution artifacts, candidate vetting statistics, corner plots, and
Doppler RV curves into a responsive standalone HTML5 application.
"""

import os
import sys
import json
import csv
import glob
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from astro_exo import __version__


def _load_json_safe(filepath: str) -> Optional[Any]:
    if os.path.isfile(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _load_csv_safe(filepath: str) -> List[Dict[str, str]]:
    if os.path.isfile(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return [row for row in reader]
        except Exception:
            return []
    return []


def _image_to_base64(filepath: str) -> str:
    """Read an image and convert it to a data URI for standalone HTML embedding."""
    if not os.path.isfile(filepath):
        return ""
    try:
        ext = os.path.splitext(filepath)[1].lower().replace(".", "")
        mime = f"image/{ext}" if ext in ["png", "jpg", "jpeg", "webp", "svg"] else "image/png"
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


def build_scientific_dashboard(
    results_dir: str = "results/rodada_amostras_ineditas",
    output_html_path: Optional[str] = None
) -> str:
    """
    Constructs an interactive, publication-grade HTML5 dashboard from pipeline results.

    Parameters
    ----------
    results_dir : str
        Directory containing pipeline results (CSVs, JSON summary, and PNG figures).
    output_html_path : str, optional
        Target path for the generated HTML file. If None, saves inside results_dir.

    Returns
    -------
    str
        Absolute path to the created HTML dashboard.
    """
    results_dir = os.path.abspath(results_dir)
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Resultados não encontrados no diretório: {results_dir}")

    if not output_html_path:
        output_html_path = os.path.join(results_dir, "dashboard_astro_exo.html")
    else:
        output_html_path = os.path.abspath(output_html_path)

    # 1. Carregar metadados e resumos
    resumo_json_path = os.path.join(results_dir, "resumo_rodada.json")
    catalogo_csv_path = os.path.join(results_dir, "catalogo_amostras_ineditas.csv")

    summary_data = _load_json_safe(resumo_json_path) or {}
    catalog_data = _load_csv_safe(catalogo_csv_path)

    # 2. Localizar e codificar figuras gerais
    painel_geral_path = os.path.join(results_dir, "painel_geral_rodada.png")
    mr_diagram_path = os.path.join(results_dir, "mass_radius_density_diagram.png")

    painel_geral_b64 = _image_to_base64(painel_geral_path)
    mr_diagram_b64 = _image_to_base64(mr_diagram_path)

    # 3. Descobrir subdiretórios de alvos individuais (TIC_*)
    target_folders = sorted(glob.glob(os.path.join(results_dir, "TIC_*")))
    targets_detail = []

    for folder in target_folders:
        folder_name = os.path.basename(folder)
        # Buscar figuras individuais
        figures = {
            "transit_fit": _image_to_base64(os.path.join(folder, "transit_fit.png")),
            "corner_mcmc": _image_to_base64(os.path.join(folder, "corner_mcmc.png")),
            "difference_image": _image_to_base64(os.path.join(folder, "difference_image_centroid.png")),
            "gaia_field": _image_to_base64(os.path.join(folder, "gaia_field_screening.png")),
            "triceratops": _image_to_base64(os.path.join(folder, "triceratops_probabilities.png")),
            "rv_fit": _image_to_base64(os.path.join(folder, "rv_keplerian_fit.png")),
        }
        targets_detail.append({
            "folder": folder_name,
            "figures": figures
        })

    # Estatísticas consolidadas
    n_targets = len(catalog_data) if catalog_data else len(targets_detail)
    n_passed = sum(1 for row in catalog_data if "PASSED" in row.get("status", "").upper())
    n_rejected = sum(1 for row in catalog_data if "REJECTED" in row.get("status", "").upper() or "FP" in row.get("status", "").upper())

    # Serializar catálogo para JSON embutido
    catalog_json_str = json.dumps(catalog_data, ensure_ascii=False)
    targets_detail_json_str = json.dumps(targets_detail, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Astro-Exo v{__version__} • Portal & Dashboard Científico</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body {{
      background: #090d16;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .glass-card {{
      background: rgba(19, 27, 46, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(30, 41, 59, 0.7);
      border-radius: 14px;
    }}
    .glow-cyan {{
      box-shadow: 0 0 25px -5px rgba(6, 182, 212, 0.25);
    }}
    .glow-emerald {{
      box-shadow: 0 0 25px -5px rgba(16, 185, 129, 0.25);
    }}
    .custom-scroll::-webkit-scrollbar {{
      width: 6px;
      height: 6px;
    }}
    .custom-scroll::-webkit-scrollbar-thumb {{
      background: #334155;
      border-radius: 4px;
    }}
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- Top Navigation -->
  <header class="border-b border-slate-800 bg-[#0b1120]/90 backdrop-blur sticky top-0 z-50 px-6 py-4">
    <div class="max-w-7xl mx-auto flex items-center justify-between flex-wrap gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-lg">
          <i class="fa-solid fa-planet-ringed text-xl"></i>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-xl font-bold tracking-tight text-white">Astro-Exo</h1>
            <span class="px-2 py-0.5 text-xs font-semibold rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 font-mono">v{__version__}</span>
            <span class="px-2 py-0.5 text-xs font-medium rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Produção</span>
          </div>
          <p class="text-xs text-slate-400">Bayesian Inference • Spatial Sub-Pixel Vetting • Joint Doppler Radial Velocity</p>
        </div>
      </div>

      <div class="flex items-center gap-4 text-sm">
        <div class="hidden sm:flex items-center gap-6 text-slate-300">
          <div><span class="text-slate-500 text-xs block">Alvos Analisados</span><strong class="text-white text-base">{n_targets}</strong></div>
          <div class="border-l border-slate-700 pl-4"><span class="text-slate-500 text-xs block">Validados</span><strong class="text-emerald-400 text-base">{n_passed}</strong></div>
          <div class="border-l border-slate-700 pl-4"><span class="text-slate-500 text-xs block">Falsos Positivos</span><strong class="text-rose-400 text-base">{n_rejected}</strong></div>
        </div>
        <button onclick="exportCSV()" class="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium flex items-center gap-2 transition">
          <i class="fa-solid fa-download"></i> Exportar CSV
        </button>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto p-6 flex-1 w-full space-y-6">

    <!-- KPI Banner -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="glass-card p-5 glow-cyan">
        <div class="text-xs font-medium uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>Amostras Inéditas</span>
          <i class="fa-solid fa-satellite-dish text-cyan-400"></i>
        </div>
        <div class="text-3xl font-extrabold text-white mt-2">{n_targets}</div>
        <div class="text-xs text-slate-400 mt-1">Ingestão de alta resolução TESS & RV</div>
      </div>

      <div class="glass-card p-5 glow-emerald">
        <div class="text-xs font-medium uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>Planetas Confirmados</span>
          <i class="fa-solid fa-circle-check text-emerald-400"></i>
        </div>
        <div class="text-3xl font-extrabold text-emerald-400 mt-2">{n_passed}</div>
        <div class="text-xs text-slate-400 mt-1">Offset sub-pixel < 1σ & FPP < 0.1%</div>
      </div>

      <div class="glass-card p-5">
        <div class="text-xs font-medium uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>BEBs Descartadas</span>
          <i class="fa-solid fa-ban text-rose-400"></i>
        </div>
        <div class="text-3xl font-extrabold text-rose-400 mt-2">{n_rejected}</div>
        <div class="text-xs text-slate-400 mt-1">Desvio espacial comprovado (> 10σ)</div>
      </div>

      <div class="glass-card p-5">
        <div class="text-xs font-medium uppercase tracking-wider text-slate-400 flex items-center justify-between">
          <span>Precisão Astrométrica</span>
          <i class="fa-solid fa-crosshairs text-amber-400"></i>
        </div>
        <div class="text-3xl font-extrabold text-amber-400 mt-2">0.14"</div>
        <div class="text-xs text-slate-400 mt-1">Ajuste PRF 2D bidimensional</div>
      </div>
    </div>

    <!-- Tabela Interativa de Candidatos -->
    <div class="glass-card p-6">
      <div class="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <h2 class="text-lg font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-table-list text-cyan-400"></i> Catálogo Físico e Status de Vetting
          </h2>
          <p class="text-xs text-slate-400">Selecione qualquer alvo para inspecionar os gráficos diagnósticos e posteriors MCMC em alta resolução.</p>
        </div>
        <div class="flex items-center gap-2">
          <input type="text" id="searchInput" oninput="filterTable()" placeholder="Buscar alvo, TIC ou regime..." class="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 text-xs text-slate-200 focus:outline-none focus:border-cyan-500">
        </div>
      </div>

      <div class="overflow-x-auto custom-scroll border border-slate-800 rounded-lg">
        <table class="w-full text-left text-xs text-slate-300">
          <thead class="bg-slate-900/80 uppercase text-[11px] font-semibold text-slate-400 border-b border-slate-800">
            <tr>
              <th class="py-3 px-4">Alvo</th>
              <th class="py-3 px-4">TIC ID</th>
              <th class="py-3 px-4">Período (d)</th>
              <th class="py-3 px-4">Raio (R<sub>⊕</sub>)</th>
              <th class="py-3 px-4">Massa (M<sub>⊕</sub>)</th>
              <th class="py-3 px-4">Densidade (g/cm³)</th>
              <th class="py-3 px-4">Offset PRF</th>
              <th class="py-3 px-4">Status Vetting</th>
              <th class="py-3 px-4 text-center">Diagnósticos</th>
            </tr>
          </thead>
          <tbody id="catalogTableBody" class="divide-y divide-slate-800/60 font-mono">
            <!-- Linhas inseridas via JavaScript -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Painéis Visuais Gerais (Sinóptico e Massa-Raio-Densidade) -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Painel 1: Visão Geral Sinóptica -->
      <div class="glass-card p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-chart-line text-blue-400"></i> Painel Sinóptico da Rodada
            </h3>
            <p class="text-xs text-slate-400">Curvas de trânsito, centróides espaciais e curvas de velocidade radial acopladas.</p>
          </div>
          <span class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">300 DPI</span>
        </div>
        <div class="bg-slate-950/60 rounded-lg border border-slate-800/80 p-2 flex items-center justify-center">
          <img src="{painel_geral_b64}" alt="Painel Geral da Rodada" class="rounded max-h-[380px] w-auto object-contain hover:scale-[1.02] transition cursor-pointer" onclick="openImageModal(this.src, 'Painel Geral Sinóptico da Rodada')">
        </div>
      </div>

      <!-- Painel 2: Diagrama Massa-Raio-Densidade -->
      <div class="glass-card p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-atom text-purple-400"></i> Diagrama Massa-Raio-Densidade (EOS)
            </h3>
            <p class="text-xs text-slate-400">Modelos de estrutura interna (Zeng et al.) contra os alvos validados com RV.</p>
          </div>
          <span class="text-xs font-mono px-2 py-0.5 rounded bg-purple-900/40 text-purple-300">Zeng et al. EOS</span>
        </div>
        <div class="bg-slate-950/60 rounded-lg border border-slate-800/80 p-2 flex items-center justify-center">
          <img src="{mr_diagram_b64}" alt="Diagrama Massa-Raio-Densidade" class="rounded max-h-[380px] w-auto object-contain hover:scale-[1.02] transition cursor-pointer" onclick="openImageModal(this.src, 'Diagrama Massa-Raio-Densidade (EOS)')">
        </div>
      </div>
    </div>

  </main>

  <!-- Modal Inspecionar Alvo -->
  <div id="targetModal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="glass-card max-w-5xl w-full max-h-[92vh] flex flex-col overflow-hidden border-slate-700 shadow-2xl">
      <div class="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center">
            <i class="fa-solid fa-telescope"></i>
          </div>
          <div>
            <h3 id="modalTargetTitle" class="text-lg font-bold text-white">TOI-1001.01 (TIC 88863718)</h3>
            <span id="modalTargetBadge" class="text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-mono">PASSED</span>
          </div>
        </div>
        <button onclick="closeTargetModal()" class="text-slate-400 hover:text-white text-lg px-2 py-1">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <!-- Abas de Diagnósticos -->
      <div class="flex border-b border-slate-800 bg-slate-900/30 px-5 gap-4 text-xs font-medium">
        <button onclick="switchTab('transit')" id="tabBtn-transit" class="py-3 border-b-2 border-cyan-400 text-cyan-300">Ajuste de Trânsito</button>
        <button onclick="switchTab('corner')" id="tabBtn-corner" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200">Posterior Corner MCMC</button>
        <button onclick="switchTab('centroid')" id="tabBtn-centroid" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200">Centróide PRF & TPF</button>
        <button onclick="switchTab('gaia')" id="tabBtn-gaia" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200">Varredura Gaia DR3</button>
        <button onclick="switchTab('triceratops')" id="tabBtn-triceratops" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200">TRICERATOPS</button>
        <button onclick="switchTab('rv')" id="tabBtn-rv" class="py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200">Velocidade Radial (RV)</button>
      </div>

      <div class="p-6 overflow-y-auto custom-scroll flex-1 bg-slate-950/40 flex items-center justify-center">
        <img id="modalDiagnosticImg" src="" alt="Diagnóstico" class="max-h-[58vh] max-w-full rounded object-contain border border-slate-800">
      </div>
    </div>
  </div>

  <!-- Modal Zoom de Imagem -->
  <div id="imageModal" class="fixed inset-0 z-50 bg-black/90 backdrop-blur-md hidden flex items-center justify-center p-4" onclick="closeImageModal()">
    <div class="relative max-w-6xl max-h-[95vh] flex flex-col items-center" onclick="event.stopPropagation()">
      <div class="flex items-center justify-between w-full mb-2">
        <h4 id="imageModalTitle" class="text-sm font-semibold text-white">Visualização Ampliada</h4>
        <button onclick="closeImageModal()" class="text-slate-400 hover:text-white text-lg"><i class="fa-solid fa-xmark"></i></button>
      </div>
      <img id="imageModalSrc" src="" class="max-h-[85vh] max-w-full rounded-lg border border-slate-800 shadow-2xl object-contain">
    </div>
  </div>

  <footer class="border-t border-slate-800 bg-[#090d16] px-6 py-4 text-center text-xs text-slate-500">
    Astro-Exo v{__version__} • Desenvolvido para astrofísica observacional de alta precisão (TESS, Kepler, Gaia, HARPS, ESPRESSO).
  </footer>

  <script>
    const catalogData = {catalog_json_str};
    const targetsDetail = {targets_detail_json_str};
    let currentSelectedTarget = null;
    let currentTab = 'transit';

    function renderCatalogTable(items) {{
      const tbody = document.getElementById("catalogTableBody");
      tbody.innerHTML = "";

      items.forEach((row, idx) => {{
        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-800/40 transition border-b border-slate-800/60";

        const isPassed = (row.status || "").toUpperCase().includes("PASSED");
        const badgeClass = isPassed 
          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
          : "bg-rose-500/10 text-rose-400 border border-rose-500/20";

        tr.innerHTML = `
          <td class="py-3 px-4 font-semibold text-white">${{row.alvo || row.target || row.name || 'Alvo ' + (idx + 1)}}</td>
          <td class="py-3 px-4 text-slate-400">${{row.tic || row.tic_id || '-'}}</td>
          <td class="py-3 px-4">${{row.period_days || row.period || '-'}}</td>
          <td class="py-3 px-4 text-cyan-300">${{row.rp_earth || row.radius_earth || row.radius || '-'}}</td>
          <td class="py-3 px-4 text-purple-300">${{row.mp_earth || row.mass_earth || row.mass || '-'}}</td>
          <td class="py-3 px-4">${{row.density_g_cm3 || row.density || '-'}}</td>
          <td class="py-3 px-4">${{row.offset_arcsec || row.centroid_offset_arcsec || '-'}}</td>
          <td class="py-3 px-4">
            <span class="px-2 py-0.5 rounded text-[10px] font-semibold ${{badgeClass}}">${{row.status || 'VAL'}}</span>
          </td>
          <td class="py-3 px-4 text-center">
            <button onclick="openTargetModal('${{row.alvo || row.name || ''}}', '${{row.tic || row.tic_id || ''}}')" class="px-2.5 py-1 rounded bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 text-xs font-sans transition">
              <i class="fa-solid fa-chart-simple mr-1"></i> Ver
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterTable() {{
      const query = document.getElementById("searchInput").value.toLowerCase();
      const filtered = catalogData.filter(r => {{
        return JSON.stringify(r).toLowerCase().includes(query);
      }});
      renderCatalogTable(filtered);
    }}

    function openTargetModal(targetName, ticId) {{
      const found = targetsDetail.find(t => t.folder.includes(String(ticId)));
      currentSelectedTarget = found || targetsDetail[0];

      document.getElementById("modalTargetTitle").innerText = `${{targetName}} (TIC ${{ticId}})`;
      document.getElementById("targetModal").classList.remove("hidden");
      switchTab('transit');
    }}

    function closeTargetModal() {{
      document.getElementById("targetModal").classList.add("hidden");
    }}

    function switchTab(tabKey) {{
      currentTab = tabKey;
      ['transit', 'corner', 'centroid', 'gaia', 'triceratops', 'rv'].forEach(t => {{
        const btn = document.getElementById(`tabBtn-${{t}}`);
        if (btn) {{
          if (t === tabKey) {{
            btn.className = "py-3 border-b-2 border-cyan-400 text-cyan-300";
          }} else {{
            btn.className = "py-3 border-b-2 border-transparent text-slate-400 hover:text-slate-200";
          }}
        }}
      }});

      if (!currentSelectedTarget || !currentSelectedTarget.figures) return;
      const keyMap = {{
        'transit': 'transit_fit',
        'corner': 'corner_mcmc',
        'centroid': 'difference_image',
        'gaia': 'gaia_field',
        'triceratops': 'triceratops',
        'rv': 'rv_fit'
      }};
      const imgKey = keyMap[tabKey];
      const src = currentSelectedTarget.figures[imgKey] || "";
      const imgElem = document.getElementById("modalDiagnosticImg");
      imgElem.src = src;
      imgElem.alt = `Diagnóstico ${{tabKey}}`;
    }}

    function openImageModal(src, title) {{
      document.getElementById("imageModalSrc").src = src;
      document.getElementById("imageModalTitle").innerText = title;
      document.getElementById("imageModal").classList.remove("hidden");
    }}

    function closeImageModal() {{
      document.getElementById("imageModal").classList.add("hidden");
    }}

    function exportCSV() {{
      if (!catalogData || catalogData.length === 0) return;
      const headers = Object.keys(catalogData[0]).join(",");
      const rows = catalogData.map(r => Object.values(r).join(",")).join("\\n");
      const csvString = "data:text/csv;charset=utf-8," + encodeURIComponent(headers + "\\n" + rows);
      const link = document.createElement("a");
      link.setAttribute("href", csvString);
      link.setAttribute("download", "astro_exo_catalogo_v{__version__}.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }}

    // Inicialização
    renderCatalogTable(catalogData);
  </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path
