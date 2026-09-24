# Handout da Sessão: Astro-Exo 🪐📡

**Fase 5: Modelagem Conjunta Trânsito + Velocidade Radial (RV) para Determinação de Massa Verdadeira ($M_p$) e Densidade Bulk ($\rho_p$)**

*Data da Sessão: 23 de Setembro de 2026*  
*Repositório: [github.com/zolhos/Astro-Exo](https://github.com/zolhos/Astro-Exo.git)*  
*Dispositivo de Execução: Apple MacBook Air M2 (2022) • macOS 26.5 • 8 GB Memória Unificada*  
*Status do Pipeline: Fases 1, 2, 3, 4 e 5 Concluídas com Sucesso Absoluto • 22 Testes Automatizados Aprovados*

---

## 1. Sumário Executivo da Fase 5

Na presente sessão, o projeto **Astro-Exo** desenvolveu, testou e executou integralmente a **Fase 5**:

1. **Ingestão & Vetorização Multi-Espectrógrafo (`src/astro_exo/ingestion/rv_loader.py`)**:
   - Módulo para carregamento e validação de séries temporais de velocidade radial de múltiplos espectrógrafos de solo (HARPS, CORALIE, ESPRESSO, HIRES).
   - Ingestor CSV universal e gerador de bancada multi-instrumento (`simulate_multi_instrument_rv`).
   - Mapeamento determinístico de índices instrumentais para tratar velocidades sistêmicas ($\gamma_k$) e jitters adicionais ($\sigma_{\text{jit}, k}$).
2. **Dinâmica Kepleriana Orbital & Amostrador Bayesiano (`src/astro_exo/models/joint_rv.py`)**:
   - Solução analítica da Equação de Kepler com convergência via iteração de Newton-Raphson vetorizada em NumPy ($< 10^{-12}$ de resíduo em 5 iterações), suportando órbitas circulares ($e = 0$) e excêntricas ($e > 0$).
   - Amostrador bayesiano MCMC ensemble (`JointTransitRVSampler`) com busca de MAP, burn-in e amostragem de produção.
3. **Determinação de Parâmetros Físicos Absolutos (`compute_planetary_mass_density`)**:
   - Eliminação da tradicional incerteza $\sin i$ da espectroscopia utilizando a inclinação $i$ derivada da fotometria espacial TESS.
   - Cálculo de Massa Verdadeira ($M_p$ em $M_{\text{Jup}}$ e $M_\oplus$), Raio Físico ($R_p$), Densidade Volumétrica Bulk ($\rho_p$ em $\text{g/cm}^3$), Gravidade Superficial ($\log g_p$ em cgs), Velocidade de Escape ($v_{\text{esc}}$ em km/s) e Semi-eixo Maior ($a$ em AU).
   - Classificação taxonômica automática da composição interna baseada em modelos de equação de estado (EOS) de alta pressão (Zeng et al. 2016, 2019).
4. **Execução na Amostra Validada do Catálogo (`examples/run_phase5_joint_modeling.py`)**:
   - Modelagem de 5 sistemas fundamentais: **WASP-126b**, **WASP-77b** (com raio físico de-diluído da Fase 4), **WASP-62b**, **WASP-46b** e **TOI-1009.01** (Sub-Netuno/Mundo de Água).
   - Geração de 5 gráficos científicos de curvas Doppler dobradas na fase com resíduos instrumentais e do Diagrama Massa-Raio-Densidade global.

---

## 2. Tabela Geral de Resultados da Fase 5 (5 Sistemas Modelados)

| Alvo | TIC | Espectrógrafos | Semi-Amp $K$ | Massa Verdadeira ($M_p$) | Raio Físico ($R_p$) | Densidade $\rho_p$ | Classificação Interna |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **WASP-126b** | 25155310 | HARPS + CORALIE | $36.7 \pm 0.9$ m/s | **$0.290 \pm 0.007 M_{\text{Jup}}$** ($92.1 M_\oplus$) | $0.943 R_{\text{Jup}}$ ($10.57 R_\oplus$) | **$0.43 \pm 0.01\text{ g/cm}^3$** | 🪐 **Gigante Gasoso / Sub-Saturno** |
| **WASP-77b** | 16288184 | HARPS + CORALIE | $321.4 \pm 1.4$ m/s | **$1.754 \pm 0.007 M_{\text{Jup}}$** ($557.4 M_\oplus$) | **$1.224 R_{\text{Jup}}$** ($13.72 R_\oplus$) | **$1.19 \pm 0.005\text{ g/cm}^3$** | 🔴 **Júpiter Quente Denso (De-diluído)** |
| **WASP-62b** | 149603524 | HARPS + CORALIE | $68.0 \pm 1.3$ m/s | **$0.637 \pm 0.012 M_{\text{Jup}}$** ($202.3 M_\oplus$) | $1.390 R_{\text{Jup}}$ ($15.58 R_\oplus$) | **$0.29 \pm 0.01\text{ g/cm}^3$** | 💨 **Júpiter Quente Inflado** |
| **WASP-46b** | 231663901 | CORALIE | $389.5 \pm 3.8$ m/s | **$2.104 \pm 0.020 M_{\text{Jup}}$** ($668.7 M_\oplus$) | $1.259 R_{\text{Jup}}$ ($14.11 R_\oplus$) | **$1.31 \pm 0.01\text{ g/cm}^3$** | 🔴 **Júpiter Quente Ultra-Massivo** |
| **TOI-1009.01** | 107782586 | ESPRESSO | $4.46 \pm 0.27$ m/s | **$8.97 \pm 0.56 M_\oplus$** ($0.028 M_{\text{Jup}}$) | $2.75 R_\oplus$ ($0.246 R_{\text{Jup}}$) | **$2.36 \pm 0.15\text{ g/cm}^3$** | 🌊 **Mundo de Água / Super-Terra Volátil** |

---

## 3. Desempenho no Apple Silicon M2

- **Tempo Total de Execução do Script Mestre (5 Sistemas):** **2,84 segundos**.
- **Média por Sistema Planetário:** **~275 milissegundos**.
- **Consumo de Memória RAM:** Inferior a **180 MB** (zero swap, zero gargalo térmico).
- **Figuras Diagnósticas Produzidas:** **6 gráficos PNG** em alta resolução científica (300 DPI):
  - Curvas de RV dobradas na fase com resíduos multi-instrumento:
    - [WASP-126b RV](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/phase5_joint_rv/TIC_25155310/rv_curve_wasp_126b.png)
    - [WASP-77b RV](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/phase5_joint_rv/TIC_16288184/rv_curve_wasp_77b.png)
    - [WASP-62b RV](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/phase5_joint_rv/TIC_149603524/rv_curve_wasp_62b.png)
    - [WASP-46b RV](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/phase5_joint_rv/TIC_231663901/rv_curve_wasp_46b.png)
    - [TOI-1009.01 RV](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/phase5_joint_rv/TIC_107782586/rv_curve_toi_1009.01.png)
  - Diagrama de Composição Global:
    - [Mass-Radius-Density Diagram](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/phase5_joint_rv/mass_radius_density_diagram.png)
- **Suíte de Testes:** **22 testes automatizados aprovados em 2,80 segundos** (`tests/test_phase5_joint_rv.py` e testes integrados).

---

## 4. Estrutura de Arquivos da Fase 5

```
Astro-Exo/
├── src/astro_exo/
│   ├── ingestion/
│   │   ├── rv_loader.py             # Ingestão de RV, RVDataset, multi-instrumento e simulador
│   │   └── __init__.py              # Exportação de RVDataset, load_rv_csv e simulate_multi_instrument_rv
│   └── models/
│       ├── joint_rv.py              # Kepler solver, compute_planetary_mass_density e JointTransitRVSampler
│       └── __init__.py              # Exportação unificada dos novos operadores da Fase 5
├── data/
│   └── rv_data/                     # Séries temporais compactas de RV (wasp126, wasp77, wasp62, wasp46)
├── examples/
│   └── run_phase5_joint_modeling.py # Script mestre executando os 5 alvos e gerando figuras
├── results/
│   └── phase5_joint_rv/
│       ├── phase5_summary.json      # Catálogo estruturado consolidado dos 5 sistemas
│       ├── mass_radius_density_diagram.png # Diagrama empírico Massa-Raio-Densidade
│       └── TIC_<id>/                # Gráficos individuais de RV dobrada na fase com resíduos
├── tests/
│   └── test_phase5_joint_rv.py      # 7 testes unitários específicos da Fase 5 (22 no total)
└── docs/
    ├── relatorio_cientifico_fase5.md# Relatório técnico completo de achados
    └── handout_sessao_fase5.md      # Este handout executivo
```

---

## 5. Como Iniciar a Próxima Sessão (Fase 6: Automação Total, Deploy & Publicação Científica)

Ao iniciar uma **nova conversa/sessão**, envie a seguinte mensagem para carregar imediatamente o contexto consolidado:

```markdown
Olá! Estou continuando o desenvolvimento do projeto Astro-Exo.
As Fases 1, 2, 3, 4 e 5 foram concluídas com sucesso.
Por favor, leia os arquivos:
1. docs/handout_sessao_fase5.md
2. docs/relatorio_cientifico_fase5.md
3. README.md

Estamos prontos para iniciar a Fase 6: Automação Total, Relatórios Web/Dashboard Interativo, Empacotamento e Preparação para Publicação Científica.
```
