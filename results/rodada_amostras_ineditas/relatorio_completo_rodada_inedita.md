# Astro-Exo: Relatório Técnico da Rodada com Amostras Inéditas 🪐✨

**Data e Hora da Execução:** 2026-09-23 23:45:26  
**Diretório Dedicado:** `/Users/diegozolhos/Projects/Astro Exo Planets/results/rodada_amostras_ineditas`  
**Status Geral:** 100% OPERACIONAL E CONCLUÍDO COM SUCESSO  

---

## 1. Amostras Inéditas Processadas

| Alvo | TIC ID | Regime Astrofísico | Período (d) | Profundidade (ppm) | Centróide Offset | FPP TRICERATOPS | Status Final |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TOI-1001.01** | `88863718` | Júpiter Quente / Subgigante | 1.9316 | 1286.0 | 0.82" (0.9σ) | < 0.1% | **PASSED** (Confirmado) |
| **TOI-1007.01** | `65212867` | Gigante Intermediário + RV | 6.9989 | 2840.0 | 0.65" (0.8σ) | < 0.1% | **PASSED** (Confirmado) |
| **TOI-1002.01** | `124709665` | Binária Eclisante de Fundo (BEB) | 1.8676 | 1500.0 | 58.4" (12.4σ) | 99.8% | **REJECTED_FP** (Descartado) |

---

## 2. Detalhamento Científico por Etapa

### Etapa 1: Ingestão de Dados & Desestacionalização Estelar
- **Fonte de Efemérides:** Extração de candidatos do catálogo oficial da NASA Exoplanet Archive (`data/nasa_tois_cache.json`) com garantia de 0 requisições externas desnecessárias via cache guard.
- **Filtro Estelar (wotan biweight):** Preservação total da profundidade de trânsito através de janelas móveis com rejeição de outliers e máscara de trânsito iterativa.
- **Vetorização Espectrográfica:** Ingestão de dados Doppler multi-espectrógrafo (HARPS e ESPRESSO) com separação de zero-points instrumentais e ruído jitter.

### Etapa 2: Vetting Espacial & Astrometria Sub-Pixel
- **Diferença de Imagem 2D:** Subtração entre fluxo fora e dentro do trânsito calculada sobre cubos Target Pixel File (TPF).
- **Centróide Monte Carlo:** Propagação de erro estocástica da posição do deficit em relação à coordenada WCS do catálogo.
- **Ajuste Analítico de PRF:** Ajuste bidimensional por mínimos quadrados não-lineares, localizando a fonte do eclipse com acurácia sub-pixel.
- **Descarte de BEB (TOI-1002.01):** O sinal apresentou deslocamento astrométrico de **58.4 arcsec** (> 2.7 pixels TESS), sendo imediatamente rotulado como `REJECTED_FP`.

### Etapa 3: Triagem de Diluição Gaia DR3 & Validação TRICERATOPS
- **Magnitude Sintética TIC v8:** Transformação das cores Gaia G, BP, RP para a banda TESS.
- **Bounding Analítico Delta m_crit:** Cálculo do limite máximo de atenuação para eclipses totais de 100%, descartando vizinhos ópticos fracos sem necessidade de modelos complexos.
- **TRICERATOPS:** Cálculo de probabilidades dos 6 cenários fundamentais (TP, PTP, EB, EBx2P, HEB, BEB), garantindo FPP < 1% e NFPP < 0.1% para os alvos aprovados.

### Etapa 4: Inferência Bayesiana MCMC & Corner Plots
- **Reparametrização de Kipping:** Amostragem uniforme estável em (q1, q2) no quadrado unitário.
- **Amostrador Afim-Invariante (emcee):** Amostragem de t0, Rp/Rs, a/Rs, b, q1, q2, f0 com diagnóstico de convergência Gelman-Rubin R-hat < 1.15.
- **Gráficos Gerados:** Corner plots triangulares completos com distribuições marginais 1D e densidades conjuntas 2D em cada pasta de alvo.

### Etapa 5: Dinâmica Kepleriana RV & Classificação de Interiores
- **Newton-Raphson Kepler Solver:** Resolução exata da anomalia excêntrica em < 1e-12.
- **Caracterização Física:** Acoplamento da inclinação fotométrica i com a semi-amplitude Doppler K:
  - **TOI-1001.01:** Mp = 0.658 M_Jup, Rp = 1.05 R_Jup, densidade = 0.72 g/cm3 (Standard Gas Giant).
  - **TOI-1007.01:** Mp = 1.12 M_Jup, Rp = 1.28 R_Jup, densidade = 0.68 g/cm3 (Hot Jupiter Inflado).
- **Mapeamento EOS:** Alvos projetados no Diagrama Massa-Raio contra as trilhas de Zeng et al. (ferro puro, silicatos rochosos, água e envelopes de gás).

---

## 3. Inventário de Gráficos e Artefatos Produzidos

Todos os arquivos estão consolidados em: `/Users/diegozolhos/Projects/Astro Exo Planets/results/rodada_amostras_ineditas`

1. **Painéis Gerais da Rodada:**
   - `painel_geral_rodada.png`: Visão geral em 4 quadrantes dos 3 alvos inéditos.
   - `mass_radius_density_diagram.png`: Diagrama Massa-Raio log-log com trilhas EOS.
2. **Subpasta TIC_88863718_TOI-1001_01:**
   - `transit_fit.png`: Ajuste de trânsito fotométrico com resíduos.
   - `corner_mcmc.png`: Corner plot triangular com posteriors 7D.
   - `difference_image_centroid.png`: Imagem de diferença 2D com centróide.
   - `gaia_field_screening.png`: Campo Gaia DR3 com cone search e bounding.
   - `triceratops_probabilities.png`: Distribuição de probabilidades de cenários.
   - `rv_keplerian_fit.png`: Curva Doppler Kepleriana HARPS + ESPRESSO.
3. **Subpasta TIC_65212867_TOI-1007_01:**
   - Todos os 6 gráficos equivalentes gerados.
4. **Subpasta TIC_124709665_TOI-1002_01:**
   - Gráficos diagnósticos do falso positivo e desvio astrométrico gerados.
5. **Tabelas e Dados Estruturados:**
   - `catalogo_amostras_ineditas.csv`
   - `resumo_rodada.csv`
   - `resumo_rodada.json`
