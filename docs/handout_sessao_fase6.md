# Handout da Sessão: Astro-Exo 🪐🚀

**Sessão Pré-Fase Final: Rodada Completa de Testes com Amostras Inéditas, Auditoria de Módulos (Fases 1 a 5) e Roteiro de Construção da Fase Final**

*Data da Sessão: 24 de Setembro de 2026*  
*Repositório: [github.com/zolhos/Astro-Exo](https://github.com/zolhos/Astro-Exo.git)*  
*Dispositivo de Execução: Apple MacBook Air M2 (2022) • macOS 26.5 • 8 GB Memória Unificada*  
*Status do Pipeline: Fases 1 a 5 Concluídas e Validadas • 32 Testes Automatizados Aprovados (100% de Sucesso)*  
*Diretório de Artefatos Dedicado: `results/rodada_amostras_ineditas/`*

---

## 1. Sumário Executivo do que Foi Realizado nesta Sessão

Nesta sessão, realizamos uma bateria rigorosa de testes ponta a ponta em todos os módulos científicos do **Astro-Exo**, submetendo alvos inéditos (não vistos nos pilotos anteriores) ao pipeline completo:

1. **Execução da Rodada Completa com Amostras Inéditas (`examples/executar_rodada_amostras_ineditas.py`)**:
   - Ingestão fotométrica e detrending spline/biweight (Fase 2).
   - Inferência Bayesiana com parametrizações de Kipping e órbitas excêntricas ($e-\omega$) (Fase 3).
   - Vetting espacial sub-pixel via diferença de imagens em TPFs, ajuste PRF 2D, cone search Gaia DR3 (raio de 2.5') e probabilidades TRICERATOPS (Fase 4).
   - Modelagem de Velocidade Radial (RV) multi-instrumento e classificação da equação de estado (EOS) no Diagrama Massa-Raio-Densidade (Fase 5).

2. **Geração Completa de Gráficos e Artefatos Determinísticos**:
   - Isolamento em diretório limpo e organizado: [`results/rodada_amostras_ineditas/`](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/rodada_amostras_ineditas/).
   - 17 gráficos individuais em alta resolução (300 DPI), cobrindo:
     - Ajuste de trânsito fotométrico (`transit_fit.png`).
     - Posterior corner plot MCMC/NUTS (`corner_mcmc.png`).
     - Mapa de diferença de imagem e centróide sub-pixel (`difference_image_centroid.png`).
     - Varredura de contaminação Gaia DR3 (`gaia_field_screening.png`).
     - Probabilidades de hipóteses astrofísicas TRICERATOPS (`triceratops_probabilities.png`).
     - Curva Kepleriana de Velocidade Radial (`rv_keplerian_fit.png`).
   - 2 painéis sinópticos globais: `painel_geral_rodada.png` e `mass_radius_density_diagram.png`.
   - Tabelas estruturadas: `catalogo_amostras_ineditas.csv`, `resumo_rodada.csv` e `resumo_rodada.json`.
   - Relatório detalhado: [`relatorio_completo_rodada_inedita.md`](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/results/rodada_amostras_ineditas/relatorio_completo_rodada_inedita.md).

3. **Resolução de Robustez & Compatibilidade no Código-Fonte**:
   - **`src/astro_exo/pipeline/runner.py`**: Suporte unificado à extração de posteriors tanto de dicionários puros quanto de objetos `arviz.InferenceData` do JAX/NumPyro NUTS.
   - **`src/astro_exo/models/gp_noise.py`**: Fallback exato com matriz densa com ruído de jitter quando `celerite2` não estiver compilado no ambiente local.
   - **`src/astro_exo/ingestion/nasa_archive.py`**: Cache offline com failover transparente (`data/nasa_tois_cache.json`) em caso de ausência de rede.
   - **`src/astro_exo/models/joint_rv.py`**: Alias formal `run_mcmc = run_rv_mcmc` para unificação da API.

4. **Expansão da Suíte de Testes Automatizados**:
   - Criação de [`tests/test_all_modules_novel_samples.py`](file:///Users/diegozolhos/Projects/Astro%20Exo%20Planets/tests/test_all_modules_novel_samples.py).
   - Total de **32 testes automatizados** passando em **7.4 segundos** com 100% de integridade.

---

## 2. Resultados Científicos da Rodada de Amostras Inéditas

| Alvo | TIC | Período (d) | Raio ($R_\oplus$) | Massa ($M_\oplus$) | Densidade ($\text{g/cm}^3$) | Centróide Offset | Status de Vetting | Classificação Interna |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TOI-1001.01** | 88863718 | 4.887 | $2.31 \pm 0.05$ | $6.97 \pm 0.44$ | $3.10 \pm 0.28$ | $0.21''$ ($0.5\sigma$) | **PASSED** (FPP < 1%) | 🌊 **Mundo de Água / Sub-Netuno** |
| **TOI-1007.01** | 65212867 | 6.541 | $13.56 \pm 0.22$ | $239.3 \pm 7.6$ | $0.53 \pm 0.03$ | $0.14''$ ($0.3\sigma$) | **PASSED** (FPP < 1%) | 🪐 **Gigante Gasoso / Júpiter Quente** |
| **TOI-1002.01** | 124709665 | 1.839 | $14.12 \pm 0.35$ | — | — | **$51.3''$ ($112\sigma$)** | ⚠ **REJECTED_FP** | ❌ **Binária Eclipsante de Fundo (BEB)** |

> **Achado Científico Chave:** O módulo de vetting espacial descartou com precisão analítica absoluta o candidato **TOI-1002.01**, localizando o déficit de fluxo a $51.3$ segundos de arco da estrela alvo primária, salvando tempo de observação astronômica no solo e comprovando a eficácia do pipeline.

---

## 3. Roteiro Lógico de Construção da Fase Final (Fase 6)

A execução da Fase 6 seguirá rigorosamente a sequência lógica de engenharia científica, garantindo estabilidade no núcleo do código antes de construir interfaces e redigir o manuscrito:

```mermaid
graph TD
    P1["Passo 1: Empacotamento de Produção & CLI Final v1.0.0<br>(pyproject.toml, CLI robusta, CI/CD GitHub Actions)"]
    P2["Passo 2: Portal Web Interativo & Dashboard Científico<br>(Dashboard estático, catálogo filtrável, curvas e diagramas)"]
    P3["Passo 3: Artigo Científico / Paper de Metodologia<br>(Manuscrito LaTeX JOSS/AAS, formulação matemática, benchmarks)"]
    P4["Passo 4: Preservação Permanente de Dados & Release DOI<br>(Congelamento v1.0.0, Zenodo/Harvard Dataverse, FAIR Data)"]

    P1 -->|Base estável e testada| P2
    P2 -->|Catálogo e figuras integradas| P3
    P3 -->|Manuscrito e código validados| P4
```

### Detalhamento dos 4 Passos Sequenciais:

1. **Passo 1: Empacotamento de Produção & CLI Final (`v1.0.0`)**:
   - Congelamento de versão e tipagem rigorosa de dependências no `pyproject.toml` (extras: `[gpu]`, `[vetting]`, `[rv]`).
   - Refinamento ergonômico da CLI (`astro-exo run`, `vet`, `batch`, `joint-rv`, `dashboard`).
   - Automação de CI/CD via GitHub Actions (testando os 32 testes unitários em Linux, macOS e Windows).

2. **Passo 2: Portal Web Interativo & Dashboard Científico**:
   - Construção do gerador estático do portal (JAMstack / HTML5 + Canvas/WebGL responsivo, pronto para GitHub Pages / Vercel).
   - Tabela dinâmica e filtrável com todos os candidatos analisados (status, raio, massa, densidade, FPP).
   - Visualizadores embutidos: curvas de luz dobradas na fase, centróides TPF, curvas de RV e Diagrama Massa-Raio dinâmico interativo.
   - Botões de exportação direta de relatórios e dados padronizados (JSON, CSV, FITS).

3. **Passo 3: Artigo Científico / Paper de Metodologia (LaTeX - JOSS / AAS)**:
   - Redação do artigo científico formal em LaTeX (`paper.tex` / `paper.md` nos padrões do *Journal of Open Source Software* ou periódicos *AAS/MNRAS*).
   - Formalização matemática completa: trânsito com Kipping (2013), amostragem NUTS em GPU, vetting PRF 2D e solução de Kepler.
   - Apresentação da matriz de validação e descarte com TOIs reais (incluindo o descarte por centróide do BEB TOI-1002.01).
   - Declaração de impacto e dados FAIR para suporte a missões espaciais (TESS, James Webb, PLATO).

4. **Passo 4: Preservação Permanente de Dados & Atribuição de DOI (Zenodo / Release)**:
   - Criação da Release Oficial `v1.0.0` no GitHub.
   - Integração com o repositório Zenodo para emissão do DOI permanente do software e dos catálogos.
   - Inserção dos badges finais no `README.md` e referências de citação.

---

## 4. Estrutura de Arquivos Atualizada

```
Astro-Exo/
├── docs/
│   ├── handout_sessao_fase6.md                 # Este handout com o roteiro lógico da Fase 6
│   ├── handout_sessao_fase5.md                 # Handout da Fase 5 (Joint RV)
│   ├── relatorio_cientifico_fase5.md           # Relatório da modelagem espectroscópica
│   └── batch_processing_guide.md              # Guia operacional do modo lote
├── examples/
│   ├── executar_rodada_amostras_ineditas.py    # Script reprodutível da rodada inédita
│   └── run_phase5_joint_modeling.py            # Modelagem conjunta dos 5 alvos
├── results/
│   └── rodada_amostras_ineditas/               # Pasta dedicada da rodada com 17 PNGs + CSVs + JSON
│       ├── catalogo_amostras_ineditas.csv
│       ├── resumo_rodada.csv
│       ├── resumo_rodada.json
│       ├── painel_geral_rodada.png
│       ├── mass_radius_density_diagram.png
│       ├── relatorio_completo_rodada_inedita.md
│       ├── TIC_88863718_TOI-1001_01/           # Gráficos individuais (transit, corner, PRF, RV, etc.)
│       ├── TIC_65212867_TOI-1007_01/           # Gráficos individuais
│       └── TIC_124709665_TOI-1002_01/          # Gráficos individuais (rejeição de BEB)
├── src/astro_exo/
│   ├── ingestion/                              # rv_loader.py, nasa_archive.py com cache
│   ├── models/                                 # joint_rv.py, gp_noise.py com fallback
│   └── pipeline/                               # runner.py com suporte a arviz/InferenceData
└── tests/
    ├── test_all_modules_novel_samples.py       # Validação completa de amostras inéditas
    └── test_phase5_joint_rv.py                 # Validação da dinâmica Kepleriana
```

---

## 5. Como Iniciar a Próxima Sessão (Fase 6: A Fase Final)

Ao abrir a próxima sessão, envie a seguinte mensagem para iniciar a execução seguindo a ordem lógica:

```markdown
Olá! Estou continuando o desenvolvimento do projeto Astro-Exo.
Todas as Fases 1 a 5 foram concluídas, testadas e validadas com sucesso (32 testes unitários passando).
A rodada completa de amostras inéditas foi executada e os artefatos estão em `results/rodada_amostras_ineditas/`.

Por favor, leia os arquivos:
1. docs/handout_sessao_fase6.md
2. results/rodada_amostras_ineditas/relatorio_completo_rodada_inedita.md
3. README.md

Estamos prontos para executar a FASE FINAL (Fase 6) na ordem lógica estabelecida:
- Passo 1: Empacotamento de Produção & CLI Final v1.0.0 (pyproject.toml, CLI e CI/CD)
- Passo 2: Portal Web Interativo & Dashboard Científico (Interface pública com catálogo e gráficos)
- Passo 3: Artigo Científico / Paper Metodológico (Manuscrito LaTeX JOSS/AAS)
- Passo 4: Preservação Permanente de Dados & Release DOI (Zenodo / GitHub Release v1.0.0)

Vamos iniciar pelo Passo 1!
```
