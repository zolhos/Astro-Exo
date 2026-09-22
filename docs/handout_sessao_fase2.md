# Handout da Sessão: Astro-Exo 🪐✨

**Da Ingestão Massiva do Catálogo NASA TOI à Validação de Fótons Reais do TESS com Vetting Espacial Sub-Pixel**

*Data da Sessão: 22 de Setembro de 2026*  
*Repositório: [github.com/zolhos/Astro-Exo](https://github.com/zolhos/Astro-Exo.git)*  
*Status do Pipeline: Fase 1 (Concluída) • Fase 2 (Concluída e Validada com Sucesso)*

---

## 1. Sumário Executivo

Nesta sessão de trabalho, o projeto **Astro-Exo** atingiu um marco científico e computacional decisivo: a transição completa de modelos sintéticos de triagem para a **ingestão, calibração, desestacionalização e validação astrométrica de dados observacionais 100% reais do satélite TESS (NASA/MIT/STScI)**.

O pipeline provou na prática a sua capacidade de operar em escala (lotes de até 400 candidatos catalogados), conectar-se de forma auditável aos repositórios astronômicos da NASA e, mais importante, **discriminar com precisão sub-pixel trânsitos planetários genuínos de falsos positivos astrofísicos** causados por estrelas binárias eclipsantes próximas (NEBs).

---

## 2. O Que Foi Realizado Nesta Sessão

### 2.1 Escalabilidade de Lotes Massivos & Rastreabilidade Auditável (Fase 1)
- **Lotes Processados**: 10, 50, 100, 200 e 400 candidatos do NASA Exoplanet Archive (TOI Table).
- **Integridade Criptográfica**: Cada lote possui seu respectivo manifesto com hashes SHA-256 (`manifest.json`), garantindo reprodutibilidade científica total.
- **Dashboards Interativos**: Geração de painéis em HTML5 Canvas + Tailwind CSS para análise visual de distribuições de período, raio, duração e impacto de centróide.

### 2.2 Ingestão de Fotometria Real do TESS via MAST REST API (Fase 2)
- **Módulo `mast_client.py`**: Desenvolvido cliente HTTP/REST com o arquivo oficial MAST CAOM (`mast.stsci.edu`), com streaming de download e cache persistente em `data/photometry/TIC_<id>/`.
- **Módulo `fits_reader.py`**: Leitor de alta velocidade para tabelas binárias FITS (`PDCSAP_FLUX`, `SAP_FLUX`, `QUALITY`), cubos espaciais 3D ($N_{cad} \times 11 \times 11$) de Target Pixel Files (TPF) e extração automática de soluções WCS (`CRVAL`, `CRPIX`, `CDELT`, `PCi_j`) do cabeçalho da extensão `APERTURE`.

### 2.3 Desestacionalização Temporal com `wotan`
- Aplicação do filtro robusto *biweight* com janela de $0.75\text{ dias}$ e tolerância a quebras de cadência de $0.5\text{ dias}$.
- Remoção da variabilidade estelar e ruído instrumental de baixa frequência, preservando com fidelidade analítica a profundidade e a morfologia em "U" dos trânsitos.

### 2.4 Vetting Espacial Sub-Pixel com Calibração Astrométrica WCS
- **Imagem de Diferença**: Subtração de fluxo $\Delta I = I_{\text{out}} - I_{\text{in}}$, isolando exclusivamente o déficit de fótons do trânsito.
- **Centróide com Monte Carlo**: Medição do primeiro momento espacial com 500 perturbações normais para propagação rigorosa de incerteza astrométrica ($1\sigma$).
- **Coordenada Absoluta de Catálogo**: Mapeamento das coordenadas celestes de catálogo ($\alpha, \delta \to (x, y)$ em pixels) via WCS, eliminando o viés do centróide de fluxo direto causado por estrelas vizinhas.

---

## 3. Resultados dos Testes em Fótons Observacionais Reais

Ao longo da sessão, foram analisados **9 conjuntos de dados observacionais reais** do TESS, totalizando mais de **150.000 cadências fotométricas de alta precisão**:

### Tabela Consolidada de Amostras Reais Processadas

| Alvo | TIC ID | Setor TESS | Período | Profundidade Observada | Deslocamento WCS | Diagnóstico Vetting | Interpretação Astrofísica |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **WASP-126b** | 25155310 | 27 | 3.2888 d | 6.196 ppm (0.62%) | **0.179 pix (3.75")** | **PASS** | Júpiter quente on-target validado |
| **WASP-46b** (TOI-101.01) | 231663901 | 27 | 1.4304 d | 16.079 ppm (1.61%) | **0.133 pix (2.79")** | **PASS** | Trânsito profundo confirmado na estrela |
| **WASP-62b** (TOI-102.01) | 149603524 | 27 | 4.4119 d | 12.967 ppm (1.30%) | **0.354 pix (7.44")** | **PASS** | Sinal límpido e centrado na estrela |
| **HATS-3b** (TOI-103.01) | 336732616 | 1 | 3.5479 d | 9.189 ppm (0.92%) | **0.313 pix (6.58")** | **PASS** | Trânsito on-target confirmado |
| **TOI-1050.01** | 66818296 | 38 | 3.7355 d | 15.192 ppm (1.52%) | **0.401 pix (8.43")** | **PASS** | Candidato validado espacialmente |
| **WASP-95b** (TOI-105.01) | 144065872 | 28 | 2.1847 d | 10.624 ppm (1.06%) | **0.099 pix (2.09")** | **PASS** | Alinhamento sub-pixel espetacular (< 0.1 pix) |
| **WASP-77b** (TOI-1049.01) | 16288184 | 39 | 2.1805 d | 13.631 ppm (1.36%) | **0.640 pix (13.45")** | **FAIL_NEB** | Contaminação pela companheira binária WASP-77B |
| **TOI-1009.01** | 107782586 | 34 | 1.9600 d | 1.167 ppm (0.12%) | **0.298 pix (6.26")** | **PASS** | Trânsito raso detectado e on-target |
| **TOI-1019.01** | 341420329 | 35 | 5.2341 d | 18.339 ppm (1.83%) | **0.779 pix (16.36")** | **FAIL_NEB** | Falso positivo: eclipse em binária vizinha |

### 🔬 O Maior Destaque Astrofísico da Sessão
O algoritmo de vetting espacial sub-pixel demonstrou sua capacidade crítica ao **desmascarar o candidato TOI-1019.01**: a curva 1D aparentava ser um trânsito perfeito de $1.83\%$, mas a imagem de diferença provou que o centro do eclipse estava deslocado em **$16.36''$ (quase 1 pixel inteiro do TESS)**, interceptando um falso positivo causado por uma binária eclipsante de fundo (NEB).

---

## 4. O Que Temos Pela Frente (Roadmap das Próximas Fases)

```
┌──────────────────────────────────────────────────────────────┐
│ STATUS ATUAL DO PROJETO ASTRO-EXO                            │
│  [✓] Fase 1: Ingestão de Catálogos & Execução Massiva em Lote │
│  [✓] Fase 2: Ingestão de Fotometria Real TESS & Vetting WCS  │
│  [ ] Fase 3: Ajuste Analítico & Modelagem Bayesiana MCMC/NUTS │
│  [ ] Fase 4: Triagem de Diluição Gaia DR3 & TRICERATOPS       │
│  [ ] Fase 5: Dinâmica Kepleriana Conjunta Trânsito + RV      │
│  [ ] Fase 6: Automação Total, Deploy & Publicação Científica  │
└──────────────────────────────────────────────────────────────┘
```

### 🎯 Fase 3: Ajuste Analítico Rápido & Modelagem Bayesiana (Próximo Passo Imediato)
1. **Modelagem Paramétrica de Trânsitos**:
   - Ajustar modelos analíticos de Mandel & Agol (2002) usando `batman` ou implementação analítica vetorizada com reparametrização de Kipping $(q_1, q_2)$.
   - Estimar os parâmetros físicos fundamentais: razão de raios ($R_p/R_\star$), semi-eixo maior normalizado ($a/R_\star$), parâmetro de impacto ($b$) e inclinação orbital ($i$).
2. **Inferência Bayesiana Posterior**:
   - Executar amostragem MCMC (`emcee`) e NUTS com aceleração GPU/CPU (`NumPyro` / `JAX`).
   - Obter distribuições posteriores e intervalos de credibilidade ($1\sigma$ e $3\sigma$) para cada parâmetro físico.
3. **Escalar o Download Concorrente do MAST**:
   - Pipeline assíncrono para baixar e processar dezenas de setores e candidatos reais em paralelo.

### 🎯 Fase 4: Triagem de Diluição Multiespectral Gaia DR3 & Validação TRICERATOPS
- **Cone Search Gaia DR3**: Consulta automática de todas as estrelas vizinhas em um raio de $2.5'$ ao redor do alvo.
- **Cálculo Analítico de Diluição ($\Delta m_{\text{crit}}$)**: Determinar matematicamente se alguma estrela vizinha teria brilho suficiente para produzir o sinal observado se fosse uma binária eclipsante.
- **TRICERATOPS**: Cálculo de probabilidades de falsos positivos (FPP e NFPP) com base em modelos de população galáctica TRILEGAL.

### 🎯 Fase 5: Dinâmica Kepleriana Conjunta (Trânsitos + Velocidades Radiais)
- Ingestão de dados espectroscópicos Doppler (HARPS, ESPRESSO, HIRES).
- Ajuste simultâneo de curva de luz + velocidade radial com prior foto-excêntrico de densidade estelar ($\rho_\star$).
- Determinação da massa verdadeira ($M_p$) e densidade volumétrica planetária bulk ($\rho_p$), classificando a composição interna (rochoso, oceânico, sub-Netuno, gigante gasoso).

### 🎯 Fase 6: Portal Web Interativo & Publicação
- Catálogo web público com as curvas reais, imagens de diferença interativas e relatórios de vetting para a comunidade astronômica.
- Preparação de artigo científico / paper de validação da metodologia.

---

## 5. Estrutura de Arquivos Criados na Sessão

```
Astro-Exo/
├── src/astro_exo/
│   ├── ingestion/
│   │   ├── mast_client.py              # Cliente REST do MAST com cache
│   │   ├── fits_reader.py              # Parser FITS e solução WCS
│   │   ├── mast_tess.py                # Wrapper de alto nível
│   │   └── detrending.py               # Algoritmo biweight do wotan
│   └── vetting/
│       └── difference_img.py           # vet_target_pixel_file (WCS + Monte Carlo)
├── examples/
│   ├── generate_phase2_report_figure.py# Gerador de figura científica
│   ├── phase2_wasp126b_photometry_vetting.png # Figura WASP-126b
│   ├── run_phase2_batch.py             # Script lote 1 (4 amostras)
│   ├── run_phase2_batch_2.py           # Script lote 2 (4 amostras)
│   ├── generate_phase2_dashboard.py    # Gerador de dashboard HTML
│   └── phase2_samples_dashboard.html   # Dashboard interativo
├── results/
│   ├── phase2_samples/                 # Resultados Lote 1 (WASP-46b, etc.)
│   └── phase2_samples_batch_2/         # Resultados Lote 2 (WASP-95b, etc.)
├── tests/
│   └── test_phase2_real_photometry.py  # Suíte de testes de integração FITS
└── docs/
    └── handout_sessao_fase2.md         # Este handout
```

---

*Handout elaborado pelo sistema autônomo Astro-Exo. Código e diagnósticos sincronizados no repositório GitHub.*
