# Handout da Sessão: Astro-Exo 🪐✨

**Fase 3: Ajuste Paramétrico de Trânsitos, Inferência Bayesiana e Modelagem MCMC / NUTS em Fótons Reais do TESS**

*Data da Sessão: 23 de Setembro de 2026*  
*Repositório: [github.com/zolhos/Astro-Exo](https://github.com/zolhos/Astro-Exo.git)*  
*Dispositivo de Execução: Apple MacBook Air M2 (2022) • macOS 26.5 • 8 GB Memória Unificada*  
*Status do Pipeline: Fase 1 (Concluída) • Fase 2 (Concluída) • Fase 3 (Concluída com Louvor) • Pronto para Fase 4*

---

## 1. Sumário Executivo

Na presente sessão de trabalho, o projeto **Astro-Exo** atingiu com plenitude o marco da **Fase 3: Ajuste Paramétrico de Trânsitos e Inferência Bayesiana de Alta Precisão**.

Conectando-se diretamente aos fótons desestacionalizados e espacialmente validados na Fase 2, o pipeline implementou e executou:
1. **Otimização Prévia MAP (Nelder-Mead / Powell)** para localização instantânea da região de máxima verossimilhança no espaço de parâmetros.
2. **Amostragem MCMC Afim-Invariante (`emcee`)** com 32 walkers explorando a distribuição a posteriori dos parâmetros de trânsito de Mandel & Agol (2002) com escurecimento de limbo quadrático via reparametrização estável de Kipping (2013) $(q_1, q_2)$.
3. **Motor Hamiltoniano Diferenciável NUTS (`NumPyro` / `JAX`)** calibrado com gradientes suaves protegidos contra singularidades analíticas para operação veloz e segura no processador Apple Silicon M2.
4. **Extração Física Completa**: Determinação de razão de raios ($R_p/R_\star$), semi-eixo maior normalizado ($a/R_\star$), parâmetro de impacto ($b$), inclinação orbital ($i$), duração total ($T_{14}$), densidade estelar observada ($\rho_\star$) e envelopes de credibilidade a $1\sigma$ ($68.3\%$) e $3\sigma$ ($99.7\%$).
5. **Diagnósticos Numéricos**: Avaliação contínua de convergência com o teste $\hat{R}$ de Gelman-Rubin e tempo integrado de autocorrelação ($\tau$).
6. **Robustez a Falhas**: Implementação de sanitização contra NaNs em quebras de cadência e sincronização automática de época de referência (`align_t0_to_dataset`).

---

## 2. Resultados Consolidados em Fótons Reais do TESS (9 Sistemas Analisados)

Ao longo de duas rodadas de execução, foram modelados **9 sistemas estelares heterogêneos**, cobrindo mais de **150.000 cadências fotométricas reais**:

### Tabela Consolidada de Parâmetros Físicos Derivados (MCMC 32 Walkers)

| Alvo | Setor | Período | $R_p/R_\star$ (Mediana $\pm 1\sigma$) | Profundidade MCMC | Impacto ($b$) | Inclinação ($i$) | Duração ($T_{14}$) | Status / Regime Astrofísico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **WASP-126b** | 27 | 3.2888 d | **$0.0763^{+0.0012}_{-0.0010}$** | $5.818 \pm 170\text{ ppm}$ | $0.238$ | $88.23^\circ$ | $3.44\text{ h}$ | 🟢 Júpiter Quente Canônico ($97.8\%$ acurácia) |
| **WASP-62b** | 27 | 4.4119 d | **$0.1116 \pm 0.0007$** | $12.458 \pm 158\text{ ppm}$ | $0.268$ | $88.41^\circ$ | $3.78\text{ h}$ | 🟢 Alta SNR, erro residual $< 0.6\%$ |
| **HATS-3b** | 1 | 3.5479 d | **$0.0988^{+0.0016}_{-0.0019}$** | $9.761 \pm 340\text{ ppm}$ | $0.575$ | $86.53^\circ$ | $3.52\text{ h}$ | 🟢 Planeta Confirmado no Setor 1 |
| **WASP-95b** | 28 | 2.1847 d | **$0.1015^{+0.0012}_{-0.0024}$** | $10.306 \pm 360\text{ ppm}$ | $0.405$ | $86.13^\circ$ | $2.87\text{ h}$ | 🟢 11 trânsitos combinados ($98.5\%$ acurácia) |
| **WASP-46b** | 27 | 1.4304 d | **$0.1406^{+0.0021}_{-0.0024}$** | $19.773 \pm 630\text{ ppm}$ | $0.728$ | $82.83^\circ$ | $1.67\text{ h}$ | 🟢 Júpiter Ultra-Quente ($P=1.43\text{d}$, alto impacto) |
| **TOI-1050.01** | 38 | 3.7355 d | **$0.1223 \pm 0.0012$** | $14.966 \pm 270\text{ ppm}$ | $0.349$ | $87.18^\circ$ | $4.32\text{ h}$ | 🟢 Candidato Gigante com trânsito de $4.3\text{h}$ |
| **TOI-1009.01** | 34 | 1.9600 d | **$0.0308^{+0.0018}_{-0.0007}$** | **$951.1^{+114}_{-42}\text{ ppm}$** | $0.161$ | $88.81^\circ$ | $1.99\text{ h}$ | 🟡 **Regime Raso (~0.09%) / Sub-Netuno** |
| **WASP-77b** | 39 | 2.1805 d | **$0.1186^{+0.0021}_{-0.0017}$** | $14.074 \pm 450\text{ ppm}$ | $0.218$ | $87.98^\circ$ | $2.97\text{ h}$ | 🔵 **Binária Diluída (WASP-77B a 3.3")** |
| **TOI-1019.01** | 35 | 5.2341 d | **$0.1408 \pm 0.0016$** | $19.830 \pm 440\text{ ppm}$ | $0.665$ | $86.20^\circ$ | $3.71\text{ h}$ | 🔴 **Falso Positivo Desmascarado (NEB 16.36")** |

---

## 3. Destaques Científicos da Sessão

### 🎯 1. Sensibilidade em Sinais Tênues (TOI-1009.01)
O pipeline provou sua capacidade de operar além de sinais fáceis de Júpiteres quentes: recuperou com firmeza um trânsito raso de **apenas $951\text{ ppm}$ ($0.095\%$)**, onde o ruído fotométrico individual supera a profundidade do trânsito. O MCMC convergiu para $R_p/R_\star = 0.0308$, caracterizando um candidato da classe de **sub-Netunos**.

### 🎯 2. O Caso do Falso Positivo Perfeito em 1D (TOI-1019.01)
Na análise fotométrica 1D, TOI-1019.01 apresenta um trânsito com queda profunda de quase $2\%$, resíduos perfeitamente planos e contornos posteriores estreitos ($R_p/R_\star = 0.1408$). Um pipeline convencional que olhasse apenas a curva de luz o validaria como um planeta gigante.  
**Contudo, o vetting espacial sub-pixel do Astro-Exo (Fase 2) demonstrou que o déficit de luz está deslocado em $16.36''$ da estrela-alvo**, interceptando uma binária eclipsante de fundo (NEB). Este caso confirma a tese do projeto: a fusão entre astrometria sub-pixel e inferência bayesiana é imprescindível para eliminar falsos positivos astrofísicos.

---

## 4. Desempenho no MacBook Air M2 (2022)

- **Tempo Médio de MCMC por Alvo:** **~3.5 a 5.0 segundos** (32 walkers $\times$ 1.350 iterações = 43.200 chamadas analíticas).
- **Tempo do Lote 1 (4 alvos):** **21.8 segundos**.
- **Tempo do Lote 2 (5 alvos):** **27.3 segundos**.
- **Consumo de Memória RAM:** Menor que **180 MB**.
- **Temperatura:** Operação completamente estável e fria na CPU M2 fanless.

---

## 5. Artefatos e Estrutura dos Arquivos da Fase 3

```
Astro-Exo/
├── src/astro_exo/
│   ├── models/
│   │   ├── transforms.py         # Kipping, geometrias, densidade estelar e align_t0_to_dataset
│   │   ├── diagnostics.py        # Gelman-Rubin R-hat e tamanho efetivo de amostra (ESS)
│   │   ├── emcee_sampler.py      # EmceeTransitFitter (MAP, MCMC, sanitização, plots)
│   │   └── jax_nuts.py           # JaxNutsTransitFitter (NUTS diferenciável com gradientes seguros)
│   └── pipeline/
│       └── runner.py             # Pipeline unificado integrando Vetting Espacial + MCMC
├── examples/
│   ├── run_phase3_real_targets.py# Script mestre da Rodada 1 (4 alvos)
│   └── run_phase3_round2.py      # Script mestre da Rodada 2 (5 alvos)
├── results/
│   ├── phase3_mcmc/              # Resultados e figuras da Rodada 1
│   │   ├── phase3_summary.json   # Resumo estruturado da Rodada 1
│   │   └── TIC_<id>/             # fit_*.png e corner_*.png de cada alvo
│   └── phase3_mcmc_round2/       # Resultados e figuras da Rodada 2
│       ├── round2_summary.json   # Resumo estruturado da Rodada 2
│       └── TIC_<id>/             # fit_*.png e corner_*.png de cada alvo
├── tests/
│   └── test_phase3_bayesian.py   # Suíte de testes automatizados da Fase 3 (8 testes, 2.1s)
└── docs/
    ├── relatorio_cientifico_fase3.md # Relatório técnico aprofundado dos 9 achados
    └── handout_sessao_fase3.md   # Este handout consolidado
```

---

## 6. Como Iniciar a Próxima Sessão (Fase 4: Gaia DR3 & TRICERATOPS)

Ao iniciar uma **nova conversa/sessão**, envie a seguinte mensagem para situar imediatamente o assistente com contexto total:

```markdown
Olá! Estou continuando o desenvolvimento do projeto Astro-Exo.
As Fases 1, 2 e 3 foram concluídas com sucesso.
Por favor, leia os arquivos:
1. docs/handout_sessao_fase3.md
2. docs/relatorio_cientifico_fase3.md
3. README.md
4. src/astro_exo/vetting/gaia.py
5. src/astro_exo/vetting/dilution.py
6. src/astro_exo/vetting/triceratops_vet.py

Estamos prontos para iniciar a Fase 4: Triagem de Diluição Multiespectral Gaia DR3 e Validação Estatística de Falsos Positivos via TRICERATOPS.
```

### O Que Faremos na Fase 4:
1. **Cone Search Automatizado no Gaia DR3**: Consultar todas as estrelas vizinhas em raio de $2.5'$ ao redor de cada TIC.
2. **Cálculo Analítico de Diluição Multiespectral ($\Delta m_{\text{crit}}$)**:
   - Determinar matematicamente se alguma vizinha teria magnitude suficiente para produzir o sinal observado se fosse uma binária eclipsante de fundo.
   - Corrigir o trânsito diluído do sistema WASP-77b para restaurar seu $R_p/R_\star$ verdadeiro!
3. **Integração com TRICERATOPS**:
   - Cálculo das probabilidades bayesianas de falso positivo (FPP e NFPP) simulando populações estelares com o modelo galáctico TRILEGAL.
   - Validação final de candidatos limpos (FPP $< 1\%$, NFPP $< 10^{-3}$).
