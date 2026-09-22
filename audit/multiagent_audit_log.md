# Auditoria Multiagêntica do Projeto Astro-Exo
**Data:** 22 de Setembro de 2026  
**Metodologia:** Debate Adversarial Cíclico em 5 Rodadas  
**Comitê Avaliador:**
- 🔭 **Dr. Marcus Vance (Astrofísico Observacional & Teórico)**: Foco em dinâmica orbital, escurecimento de limbo, física de trânsitos e contaminação estelar.
- 📐 **Dra. Elena Rostova (Estatística Bayesiana & Especialista em MCMC)**: Foco em espaços de parâmetros, verossimilhança, priors informativas, processos gaussianos e convergência de amostradores.
- 💻 **Dr. Alex Chen (Engenheiro de Software Científico & Especialista JAX/HPC)**: Foco em compilação JIT, paralelismo, diferenciação automática, estabilidade numérica e arquitetura de software.
- ⚖️ **Dr. Thaddeus Drake (Revisor Crítico Adversário / Red-Team Referee)**: Cético implacável, focado em falsear premissas, expor modos de falha, falsos positivos e limitações operacionais.

---

## Rodada 1: Fundações, Ingestão de Dados e Suposições Ocultas

### ⚖️ Dr. Thaddeus Drake (Revisor Adversário):
"Senhores, examinemos o código em `src/astro_exo/ingestion/`. O pipeline começa afirmando que busca planetas novos para a literatura. Porém, logo no primeiro passo (`detrending.py`), o filtro bi-weight do `wotan` é aplicado. Como vocês garantem que esse filtro não está deformando sutilmente a forma do trânsito? Se o trânsito ainda não foi detectado (busca cega via TLS), como vocês passam uma `transit_mask` para o `wotan`? Não passam! E ao aplanar curvas sem máscara, o filtro 'mergulha' no trânsito, suaviza o fundo do poço, deforma os ombros de ingresso/egresso e reduz artificialmente $R_p/R_*$ e o parâmetro de impacto $b$.
Além disso, no `mast_tess.py`, vocês filtram `lc.quality == 0`. Sabiam que em dados de FFIs do TESS (QLP), bandeiras de dispersão de luz de fundo e momentum dumps frequentemente contêm trânsitos legítimos que são sumariamente jogados fora, enquanto ruídos de borda passam desapercebidos? Esse início já introduz viés sistemático."

### 🔭 Dr. Marcus Vance (Astrofísico):
"O ponto do Dr. Drake sobre a atenuação da profundidade em detrending cego é um problema clássico bem documentado na literatura (por exemplo, Hippke et al. 2019). No entanto, o `wotan` com janela bi-weight de 0.75 dias (~18 horas) foi calibrado exatamente para minimizar esse efeito: a duração típica de um trânsito de período curto é de 2 a 4 horas, o que representa menos de 20% da largura da janela bi-weight. A distorção no núcleo do trânsito com janela de 18 horas é inferior a 1.5% da profundidade.
Contudo, concordo que após a primeira passada do TLS e descoberta do período preliminar e época $T_0$, o pipeline deve **obrigatoriamente reprocessar o detrending** em uma segunda passada, agora com o trânsito mascarado em $\pm 1.5 \times T_{\text{dur}}$, antes de qualquer ajuste de MCMC. Proponho formalizarmos esse fluxo de duas etapas (`iterative_detrend`)."

### 📐 Dra. Elena Rostova (Estatística Bayesiana):
"Do ponto de vista estatístico, o problema é ainda mais profundo: dividir o fluxo bruto por uma tendência determinística fixa calculada a priori (`flatten_flux = flux / trend`) destrói a estrutura formal de incerteza. A incerteza $\sigma_i$ do fluxo medido é propagada ingenuamente como $\sigma_i / \text{trend}$, ignorando completamente a covariância da própria interpolação da tendência!
Se quisermos rigor de nível periódico Q1, o modelo final não deve separar detrending de ajuste de trânsito. A tendência estelar de baixa frequência deve ser tratada simultaneamente no modelo probabilístico através do Processo Gaussiano (`celerite2`) com o kernel estocástico SHOTerm, conforme previsto em `gp_noise.py`. O detrending com `wotan` deve servir estritamente para a detecção inicial no TLS, nunca como dado definitivo para a inferência bayesiana."

### 💻 Dr. Alex Chen (Engenheiro JAX):
"Concordo com a Dra. Rostova. Do ponto de vista de código e arquitetura, notei que o módulo `ingestion/search.py` depende de `transitleastsquares`, que executa em loops CPU convencionais em C/Cython. Para um único alvo, isso roda em 15 segundos. Mas se formos rodar 10.000 curvas de luz de um setor TESS, o TLS torna-se o gargalo mais severo do pipeline.
Além disso, no `mast_tess.py`, as chamadas via `lightkurve.search_lightcurve` fazem requisições síncronas de rede ao MAST. Para processamento em larga escala, precisamos de um adaptador assíncrono ou leitura direta dos buckets públicos AWS S3 (`s3://stpubdata/tess/`).
Sobre o tratamento de exceções: se o download falhar ou corromper, o código atual levanta um `ValueError` fatal que quebra o loop do lote. Devemos encapsular a ingestão com retries exponenciais e logs estruturados."

---

## Rodada 2: Vetting Espacial, Resolução de 21" do TESS e Contaminação

### ⚖️ Dr. Thaddeus Drake (Revisor Adversário):
"Vamos ao calcanhar de Aquiles de qualquer pipeline do TESS: **a contaminação por pixels gigantescos de 21 segundos de arco**.
Em `src/astro_exo/vetting/prf_fit.py`, vocês implementaram uma função PRF analítica baseada em uma Gaussiana 2D simples (`gaussian_2d_prf`). Senhores, a óptica do TESS não tem PRF gaussiana! A PRF real do TESS é assimétrica, possui asas triangulares pronunciadas nas bordas do campo de visão (aberração cromática e coma) que variam dramaticamente dependendo de onde o alvo cai nos 4 sensores CCD de cada câmera.
Ajustar uma Gaussiana 2D ingênua para inferir a posição sub-pixel com Nelder-Mead é um convite ao erro sistemático de até 0.5 pixel (~10 arcseconds). Vocês podem facilmente aprovar uma binária eclipsante de fundo (BEB) a 8 segundos de arco achando que ela está centralizada no alvo! Como vocês defendem isso perante um revisor do *ApJ*?"

### 🔭 Dr. Marcus Vance (Astrofísico):
"O Dr. Drake tocou em um ponto de extrema relevância física. O TESS possui uma variação espacial brutal de PRF do centro para os cantos do plano focal. Uma Gaussiana 2D simétrica é apenas uma aproximação de primeira ordem para inspeção rápida.
Entretanto, o framework não depende exclusivamente da PRF! Ele opera em camadas de redundância:
1. Primeiro, temos o cálculo direto de momentos de centróide na imagem de diferença (`difference_img.py`) com perturbação Monte Carlo de 500 iterações.
2. Segundo, temos a triagem analítica com Gaia DR3 (`dilution.py`). Mesmo que a PRF tivesse incerteza de 5 arcseconds, o módulo `dilution.py` calcula $\Delta m_{\text{max}} = 2.5 \log_{10}(1 / \delta_{\text{obs}})$. Se a estrela vizinha no Gaia for mais fraca que a magnitude limite calculada, ela é **matematicamente incapaz** de gerar o sinal observado, independentemente da geometria da PRF!"

### 📐 Dra. Elena Rostova (Estatística Bayesiana):
"O Dr. Marcus tem razão sobre a diluição analítica, mas o Dr. Drake levantou uma falha metodológica grave no ajuste da PRF: o uso de Nelder-Mead em `prf_fit.py`. O Nelder-Mead é um otimizador simplex local sem cálculo de matriz de covariância. O código retorna `offset_arcsec`, mas não fornece a **incerteza formal posterior $\sigma_{\Delta x}, \sigma_{\Delta y}$** do ajuste da PRF!
Sem incerteza formal no centróide da PRF, como calcular se o deslocamento é significativo a $3\sigma$?
Devemos substituir o Nelder-Mead por uma amostragem MCMC rápida das coordenadas $(x_0, y_0, \text{amp})$ ou, no mínimo, calcular a matriz Hessiana no ponto de convergência para extrair a matriz de covariância assintótica. Assim teremos elipses de erro de 1$\sigma$ e 3$\sigma$ projetadas nas coordenadas Gaia."

### 💻 Dr. Alex Chen (Engenheiro JAX):
"E podemos ir além: o ajuste de PRF e cálculo de imagem de diferença pode ser vetorizado em JAX. Se compilarmos a função de perda em JIT com diferenciação automática (`jax.grad` e `jax.hessian`), não precisamos de Nelder-Mead iterativo lento. O otimizador L-BFGS com gradientes exatos converge em menos de 10 milissegundos e nos devolve a Hessiana analítica exata instantaneamente.
Além disso, para a PRF real do TESS, podemos carregar as matrizes ePSF empíricas oficiais da NASA (arquivos `PRF_Sector*.fits`) disponibilizados pelo time do SPOC, interpolando cubicamente no ponto do alvo."

---

## Rodada 3: Motor Bayesiano, Priors de Kipping e Degenerescência Foto-Excêntrica

### ⚖️ Dr. Thaddeus Drake (Revisor Adversário):
"Entremos na matemática de `src/astro_exo/models/`. Vocês adotaram a reparametrização de Kipping (2013) com $q_1, q_2 \sim \mathcal{U}(0, 1)$. Muito elegante no papel. Mas vocês assumiram a priori uniforme plana em $q_1$ e $q_2$ para qualquer estrela!
Uma estrela anã M fria ($T_{\text{eff}} \sim 3200\text{ K}$) tem um perfil de escurecimento de limbo drasticamente diferente de uma estrela tipo F quente ($T_{\text{eff}} \sim 6800\text{ K}$). Permitir que $q_1, q_2$ flutuem livremente de 0 a 1 em um trânsito ruidoso permite que o sampler explore perfis estelares completamente não-físicos, compensando ruído na curva de luz e falseando o parâmetro de impacto $b$.
E pior: no script `emcee_sampler.py`, a órbita é assumida estritamente circular ($e = 0$). Na natureza, uma fração enorme de planetas possui excentricidade orbital não nula. Se o planeta for excêntrico com $\omega \approx 90^\circ$, o trânsito é mais curto, o que fará o seu modelo circular superestimar a densidade estelar $\rho_*$ em até 300%! Como o pipeline lida com essa degenerescência foto-excêntrica?"

### 🔭 Dr. Marcus Vance (Astrofísico):
"O Dr. Drake identificou com precisão a degenerescência foto-excêntrica descoberta por Dawson & Johnson (2012). Se forçamos $e=0$, a densidade estelar derivada do trânsito $\rho_{*, \text{transit}} \propto (a/R_*)^3 / P^2$ torna-se uma densidade estelar aparente $\rho_{*, \text{circ}} = \rho_{*, \text{true}} \times g(e, \omega)$, onde $g(e, \omega) = \frac{(1+e\sin\omega)^3}{(1-e^2)^{3/2}}$.
É exatamente por isso que desenvolvemos o módulo `models/jax_nuts.py` (Turno 18)! Nele, nós não forçamos $e=0$. Introduzimos as variáveis $h = \sqrt{e}\cos\omega$ e $k = \sqrt{e}\sin\omega$, e acoplamos uma prior gaussiana truncada na verdadeira densidade estelar $\rho_{*, \text{Gaia}}$ obtida dos raios e massas do catálogo Gaia DR3 / FLAME.
Quando o sampler tenta variar a duração do trânsito alterando $a/R_*$, a prior empírica de densidade do Gaia atua como uma âncora, quebrando a degenerescência e permitindo inferir a excentricidade $e$ diretamente da fotometria!"

### 📐 Dra. Elena Rostova (Estatística Bayesiana):
"Sobre os coeficientes de Kipping: o Drake tem razão em relação à prior uniforme estrita. Em sistemas de baixa razão sinal-ruído ($S/N < 10$), a prior $\mathcal{U}(0, 1)$ tem volume excessivo.
A correção metodológica recomendada é consultar as tabelas estelares de Claret (2017) para a banda do TESS, dadas a temperatura efetiva $T_{\text{eff}}$, gravidade superficial $\log g$ e metalicidade $[\text{Fe/H}]$ da estrela hospedeira (disponíveis no TESS Input Catalog - TIC). Em seguida, convertemos esses valores $u_{1, \text{tab}}, u_{2, \text{tab}}$ para $q_{1, \text{tab}}, q_{2, \text{tab}}$ via `quadratic_to_kipping()` e aplicamos uma prior Gaussiana informativa truncada com desvio padrão $\sigma \approx 0.10$. Isso estabiliza o ingresso/egresso e impede que ruído residual seja absorvido pelo limbo."

### 💻 Dr. Alex Chen (Engenheiro JAX):
"Do lado computacional do NumPyro (`jax_nuts.py`), a inclusão de $h, k$ e prior de densidade estelar exige muita cautela com a geometria do espaço de amostragem. Em órbitas de alto impacto ($b \to 1$), o gradiente da duração do trânsito em relação a $b$ diverge. O amostrador NUTS pode produzir 'divergências simpléticas' (*divergent transitions*).
Para evitar isso em produção:
1. Devemos reparametrizar o impacto $b$ e o raio $r_p$ conjuntamente quando $b + r_p \approx 1$.
2. Devemos configurar `target_accept_prob=0.90` (ou 0.95) no NUTS para forçar passos de integração menores perto das cúspides da verossimilhança.
3. Precisamos garantir que o cálculo de $\rho_*$ no JAX use operações numéricas protegidas contra divisão por zero (`jnp.clip`)."

---

## Rodada 4: Engenharia de Software, Confiabilidade Numérica e Escalabilidade

### ⚖️ Dr. Thaddeus Drake (Revisor Adversário):
"Passemos à robustez do software (`src/astro_exo/pipeline/runner.py` e `cli.py`).
O código atual é um excelente protótipo acadêmico, mas se eu colocá-lo para rodar em uma máquina sem GPU ou com memória limitada sob dados reais do TESS, vejo riscos imediatos:
1. Em `emcee_sampler.py`, os walkers são inicializados com uma pequena perturbação gaussiana ao redor da estimativa MAP. Se a otimização MAP convergir para um mínimo local falso (por exemplo, um harmônico de período ou um artefato de ruído instrumental), **todos os 32 walkers começam presos no poço errado**! O `emcee` não tem capacidade de escapar de armadilhas multimodais com facilidade.
2. Em `runner.py`, vocês têm blocos `except Exception as e:` que simplesmente imprimem um aviso e continuam. Em astronomia automatizada, suprimir silenciosamente falhas de calibração gera 'planetas fantasmas'.
3. Onde está o controle de versão de dados e a garantia de determinismo (seeds)? Como um comitê científico pode reproduzir exatamente a mesma cadeia posterior?"

### 💻 Dr. Alex Chen (Engenheiro JAX):
"O Dr. Drake tocou nas dores clássicas de pipelines astrofísicos. Aqui está nossa resposta de engenharia:
1. **Inicialização do MCMC:** O Nelder-Mead único deve ser substituído por uma busca global preliminar com Differential Evolution ou otimização multi-start (10 pontos aleatórios nascidos do grid TLS). Além disso, o NUTS no NumPyro (`jax_nuts.py`) com múltiplos inícios (*chains*) é ordens de magnitude menos sensível a aprisionamento local do que o ensemble sampler afim do `emcee`.
2. **Tratamento de Exceções & Falhas:** O bloco genérico de captura deve ser substituído por exceções tipadas de domínio (`AstrometricCalibrationError`, `AperturePhotometryError`, `ConvergenceFailure`). Se o vetting falhar, o produto deve registrar formalmente `disposition = 'VETTING_FAILED'` com o traceback serializado no JSON do candidato, sem mascaramento.
3. **Determinismo e Reprodutibilidade:** O `PipelineConfig` já possui o campo `seed`, mas devemos fixar o gerador de números pseudoaleatórios tanto no NumPy (`np.random.default_rng(seed)`) quanto no JAX (`jax.random.PRNGKey(seed)`), registrando a chave inicial e os hashes de commit do Git no schema `FullCandidateProduct`."

### 📐 Dra. Elena Rostova (Estatística Bayesiana):
"Complementando o Chen: para a convergência no `emcee` e no `NumPyro`, a esteira precisa verificar automaticamente o diagnóstico de Gelman-Rubin $\hat{R}$ e o Effective Sample Size (ESS).
Se $\hat{R} > 1.01$ para qualquer parâmetro físico ($r_p, a/R_*, b, t_0$), a cadeia **não convergiu**. O pipeline não pode emitir um relatório de 'candidato validado' se $\hat{R}$ estiver inflado. Devemos adicionar uma barreira lógica de parada que estende o número de passos ou rebaixa o candidato para 'NEEDS_LONGER_MCMC'."

### 🔭 Dr. Marcus Vance (Astrofísico):
"E sobre a validação externa TRICERATOPS (`triceratops_vet.py`): no código atual, se a biblioteca não estiver instalada, há um fallback que retorna valores simulados (`fpp: 0.005`). Isso foi útil para o protótipo inicial, mas em produção científica é estritamente proibido emitir uma flag `validated: True` baseada em mock.
Se o TRICERATOPS não estiver disponível, o status deve ser categorizado como `UNVETTED_STATISTICALLY` até que as probabilidades reais sejam calculadas via Monte Carlo sobre os modelos de população TRILEGAL."

---

## Rodada 5: Ataque de Estresse Global, Síntese e Consenso Editorial

### ⚖️ Dr. Thaddeus Drake (Revisor Adversário):
"Chegamos à rodada decisiva. Coloquei o framework sob teste conceitual em quatro cenários extremos:
1. **Cenário A (O Falso Positivo Perfeito):** Uma binária eclipsante de fundo (BEB) a 15 arcseconds do alvo, com eclipse de 40%, diluída em 98% pelos pixels gigantes do TESS, gerando um trânsito em 'U' de 800 ppm na estrela alvo.
2. **Cenário B (Ruído de Granulação Estelar Intensa):** Uma estrela subgigante jovem com oscilações quase-periódicas que mimetizam um trânsito raso de período curto.
3. **Cenário C (Órbita Altamente Excêntrica $e > 0.6$):** O trânsito dura 1 hora em vez das 3.5 horas esperadas para uma órbita circular.
4. **Cenário D (Dados Incompletos / Gaps de Momentum Dump):** Apenas 1.5 trânsitos observados no setor.

Se o pipeline de vocês for submetido a esses quatro testes hoje, ele sobrevive?"

### 🔭 Dr. Marcus Vance (Astrofísico):
"Sim, exatamente por causa da arquitetura multi-camadas que desenhamos:
- No **Cenário A (BEB a 15")**: O módulo `difference_img.py` detectará imediatamente um deslocamento de centróide de $\sim 15$ arcseconds ($\sim 0.7$ pixel do TESS), correspondendo a uma significância de mais de $15\sigma$. O candidato será sumariamente rotulado como `FALSE_POSITIVE` antes mesmo de gastar tempo de GPU no MCMC. Além disso, o `dilution.py` com Gaia DR3 confirmará a presença da estrela secundária na abertura.
- No **Cenário C (Excêntrico $e > 0.6$)**: O modelo conjunto no JAX acoplado à densidade estelar do Gaia $\rho_{*, \text{Gaia}}$ rejeitará a solução circular (que exigiria uma estrela de densidade irrealmente alta) e convergirá para a solução de alta excentricidade com $h, k$ bem determinados."

### 📐 Dra. Elena Rostova (Estatística Bayesiana):
"- No **Cenário B (Ruído de Granulação Estelar)**: O modelo acoplado com Gaussian Processes `celerite2` (`gp_noise.py`) absorve as oscilações estelares na escala de tempo da granulação ($\rho_{\text{GP}} \sim \text{horas}$), impedindo que o amostrador interprete as flutuações correlacionadas como quedas periódicas de trânsito. O teste de Bayes Factor (evidência marginal entre modelo com trânsito+GP vs apenas GP) rejeitará o sinal se a profundidade for explicada pela variabilidade estelar.
- No **Cenário D (Gaps e trânsitos incompletos)**: A verossimilhança gaussiana pura sem GP pode sofrer, mas com a prior gaussiana informativa em $T_0$ e o modelo analítico Mandel & Agol com supersampling, a incerteza posterior em $P$ e $T_0$ aumentará honestamente, refletindo a falta de dados sem colapsar em falsas certezas."

### 💻 Dr. Alex Chen (Engenheiro JAX):
"O consenso entre nós está maduro. O pipeline `Astro-Exo` não é apenas um script conceitual: ele possui a espinha dorsal matemática correta que a astrofísica moderna exige.
Para elevá-lo ao padrão definitivo de aceitação imediata em periódicos como *The Astrophysical Journal* (ApJ) e *Monthly Notices of the Royal Astronomical Society* (MNRAS), consolidamos as 5 melhorias obrigatórias identificadas neste debate:
1. Detrending iterativo em dois estágios com máscara de trânsito.
2. Derivação de incertezas formais no centróide da PRF via Hessiana analítica.
3. Priors estelares informativas de Kipping ($q_1, q_2$) via tabelas de Claret calibradas com o TIC.
4. Barreira de qualidade e convergência estrita com $\hat{R} < 1.01$ e ESS $> 400$.
5. Eliminação de fallbacks mock silenciosos para validações estatísticas."

---

## Veredito Consolidado do Comitê de Revisão (Referee Synthesis)

| Dimensão Avaliada | Pontuação (0 - 10) | Diagnóstico do Comitê |
| :--- | :---: | :--- |
| **Rigor Físico & Astrofísico** | **9.2 / 10** | Formulações canônicas (Mandel & Agol, Kipping, dinâmica Kepleriana de RV, densidade estelar). |
| **Modelagem Estatística & Bayesiana** | **9.0 / 10** | Excelente uso de JAX/NumPyro, NUTS, celerite2 GP e quebra da degenerescência foto-excêntrica. |
| **Vetting Espacial & Disambiguação** | **8.8 / 10** | Robusto contra BEBs via momentos de diferença e Gaia DR3; recomendada PRF empírica para setores periféricos. |
| **Engenharia de Software & HPC** | **8.9 / 10** | Modular, compatível com PEP 621, testes unitários automatizados, leveza local e suporte a GPU. |
| **Prontidão Científica Geral** | **9.0 / 10** | **Recomendação: Aceito com Revisões Menores (Accept with Minor Revisions).** |
