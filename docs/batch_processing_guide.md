# Guia Completo: Processamento em Lote no Astro-Exo 🪐🚀

Este guia documenta como configurar, alimentar e executar processamentos em lote (*batch processing*) de candidatos a exoplanetas utilizando o **Astro-Exo**, desde pequenos lotes piloto até catálogos de larga escala derivados da missão NASA TESS e arquivos MAST.

---

## 1. Visão Geral da Arquitetura de Lotes

O motor de lotes é gerenciado pela classe `BatchProcessor` (`src/astro_exo/pipeline/batch.py`), projetado para:

1. **Leitura Flexível de Catálogos**: Ingestão direta de arquivos `.csv` ou `.json`.
2. **Tolerância a Falhas (*Fault Tolerance*)**: Se um alvo apresentar dados corrompidos ou indisponibilidade de setor no MAST, o erro é registrado no relatório individual e o lote continua sem interrupções.
3. **Modos de Execução**:
   * **Modo MOCK (Simulado / Offline)**: Gera tensores espaciais e séries temporais com física estelar realista em milissegundos, permitindo validar o fluxo e o algoritmo sem necessitar de conexão com a internet.
   * **Modo LIVE (MAST)**: Conecta-se diretamente aos servidores da NASA via `lightkurve` e `astroquery` para baixar os arquivos FITS reais (`_tp.fits` e `_lc.fits`).
4. **Relatórios Consolidados Automáticos**:
   * `batch_summary.csv`: Tabela estruturada para análise rápida em planilhas ou DataFrames.
   * `batch_summary.json`: Metadados completos do processamento.
   * `TIC_<id>.json`: Relatório individual detalhado por estrela.

---

## 2. O Lote Piloto de Candidatos Pré-Identificados

O repositório já inclui um lote piloto curado em `data/pilot_candidates.csv` e `data/pilot_candidates.json` com candidatos da literatura científica:

| TIC ID | Nome | TOI | Setor | Período (dias) | Época T0 (BJD) | Duração (h) | Profundidade | Tipo / Disposição |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `25155310` | **WASP-126b** | 114.01 | 1 | 3.2888 | 1325.503 | 3.05 | 8.800 ppm | Planeta Confirmado (Júpiter Quente) |
| `100100827` | **WASP-18b** | 172.01 | 2 | 0.9414 | 1354.215 | 2.18 | 9.200 ppm | Planeta Confirmado (Período Ultra-Curto) |
| `52368076` | **TOI-125b** | 125.01 | 1 | 4.6538 | 1327.482 | 2.95 | 1.420 ppm | Planeta Confirmado (Sub-Netuno) |
| `259377017` | **TOI-270b** | 270.01 | 3 | 3.3600 | 1383.524 | 1.68 | 1.980 ppm | Planeta Confirmado (Super-Terra) |
| `99999901` | **Mock-BEB** | 999.01 | 1 | 2.5000 | 1326.100 | 2.50 | 15.000 ppm | **Falso Positivo de Controle (BEB)** |

---

## 3. Como Executar pelo Terminal (CLI)

O pipeline expõe o comando `batch` diretamente na linha de comando:

### Executando o Lote Piloto em Modo Rápido (Mock)
```bash
PYTHONPATH=src python3 -m astro_exo.cli batch --input data/pilot_candidates.csv --outdir results/pilot_batch --mode mock
```

### Executando com Download Real do MAST da NASA (Live)
```bash
astro-exo batch --input data/pilot_candidates.csv --outdir results/real_batch --mode live
```

### Opções do Comando CLI

* `--input`, `-i`: Caminho do arquivo de candidatos (`.csv` ou `.json`). **[Obrigatório]**
* `--outdir`, `-o`: Diretório para salvar os resumos e relatórios (Padrão: `results/batch_run`).
* `--mode`, `-m`: Modo de execução (`mock` para simulação offline de alta precisão ou `live` para download via MAST). Padrão: `mock`.

---

## 4. Como Executar via Código Python

Você pode incorporar o `BatchProcessor` em seus próprios scripts ou notebooks:

```python
from astro_exo.pipeline.batch import BatchProcessor
from astro_exo.pipeline.config import PipelineConfig

# 1. Configurações globais
config = PipelineConfig(
    run_vetting=True,
    run_difference_imaging=True,
    run_gaia_overlay=True,
    output_dir="results/meu_lote"
)

# 2. Inicializa o processador
processor = BatchProcessor(config=config, mode="mock", output_dir="results/meu_lote")

# 3. Processa a tabela de candidatos
results = processor.process_catalog("data/pilot_candidates.csv")

# 4. Inspeciona resultados no console
for r in results:
    print(f"TIC {r.tic_id} ({r.name}): Status = {r.status}, Offset = {r.centroid_offset_arcsec}\"")
```

---

## 5. Estrutura dos Arquivos de Saída

Ao concluir o lote, o diretório de saída contém:

```text
results/pilot_batch/
├── batch_summary.csv        <- Tabela consolidada com todos os alvos
├── batch_summary.json       <- Lista serializada com todas as métricas
├── TIC_25155310.json        <- Detalhamento completo do WASP-126b
├── TIC_100100827.json       <- Detalhamento do WASP-18b
├── TIC_52368076.json        <- Detalhamento do TOI-125b
├── TIC_259377017.json       <- Detalhamento do TOI-270b
└── TIC_99999901.json        <- Registro de rejeição do BEB
```

### Principais Colunas do `batch_summary.csv`
* `status`: `PASSED` (candidato aprovado no vetting) ou `REJECTED_FP` (falso positivo descartado).
* `centroid_offset_arcsec`: Deslocamento angular entre o centróide da queda de fluxo e a coordenada de catálogo do alvo.
* `centroid_offset_sigma`: Significância estatística do deslocamento em sigmas. Valores $> 3.0\sigma$ indicam que o eclipse não ocorreu no alvo.
* `neighbors_ruled_out`: Quantidade de estrelas vizinhas do Gaia DR3 que foram matematicamente descartadas por limite analítico de profundidade ($\Delta m_{\text{max}}$).
* `derived_density_g_cm3`: Densidade bulk do planeta derivada a partir da combinação de trânsito e velocidade radial.
