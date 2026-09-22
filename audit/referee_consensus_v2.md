# Consenso Editorial e Plano de Remediação Imediata (V2)
**Órgão Revisor:** Comitê Editorial Multiagêntico de Pares  
**Data:** 22 de Setembro de 2026  
**Documento de Referência:** Parecer Unificado dos 5 Subagentes Autônomos  

---

## 1. Matriz de Consenso Técnico

| Item Auditado | Posição Adversária (Dr. Drake) | Defesa Técnica (Drs. Vance, Rostova, Chen) | Solução de Consenso Aprovada |
| :--- | :--- | :--- | :--- |
| **1. Detrending** | O filtro bi-weight atenua a profundidade do trânsito se aplicado sem máscara prévia. | A janela de 18h minimiza o efeito (<1.5%), mas o trânsito deve ser mascarado antes do MCMC. | Conectar `iterative_flatten` no `runner.py` com máscara em $\pm 1.5 \times T_{\text{dur}}$ e corrigir variável `win` no fallback. |
| **2. Gaia DR3 & PRF** | `SkyCoord` e `minimize` não importados quebram a execução; Gaussiana 2D é simplista para o TESS. | A imagem de diferença Monte Carlo e a diluição analítica barram >95% das BEBs independente da PRF. | Importar `SkyCoord`, `u` e `minimize`; calcular Hessiana numérica e incertezas $\sigma_x, \sigma_y$. |
| **3. Vetting Logic** | `runner.py` aprova candidatos (`VALIDATED_PLANET`) quando o vetting falha por exceção. | Falha de lógica de controle que mascara erros de calibração. | Definir `passed_vetting = False` por padrão; qualquer exceção força status `VETTING_ERROR / REJECTED`. |
| **4. NUTS / JAX** | A função degrau retangular em `jax_nuts.py` destrói o integrador Leapfrog do NUTS. | Necessário kernel suave e contínuo para diferenciação automática reversa (`jax.grad`). | Implementar perfil analítico com limb darkening contínuo e conectar o backend `jax_nuts` no runner. |
| **5. Excentricidade** | O MCMC clássico impõe $e=0$, inflando a densidade estelar $\rho_*$ em até 300%. | A parametrização $(h, k)$ acoplada à densidade Gaia DR3 quebra a degenerescência. | Integrar prior empírica de densidade do Gaia e amostragem de $(h, k)$ no fluxo principal. |

---

## 2. Ações de Remediação no Código

1. **`src/astro_exo/vetting/gaia.py`**:
   - Inserir formalmente `from astropy.coordinates import SkyCoord` e `import astropy.units as u`.
2. **`src/astro_exo/vetting/prf_fit.py`**:
   - Inserir formalmente `from scipy.optimize import minimize`.
3. **`src/astro_exo/ingestion/detrending.py`**:
   - Definir formalmente `win` no fallback de Savitzky-Golay a partir de `window_length`.
4. **`src/astro_exo/pipeline/runner.py`**:
   - Importar `from typing import Optional`.
   - Iniciar `passed_vetting = False`.
   - Trocar chamada direta para `iterative_flatten` com máscara de trânsito.
   - Em caso de falha no bloco `except`, marcar explicitamente `passed_vetting = False` e `disposition = "VETTING_ERROR"`.
   - Conectar o switch do backend `jax_nuts` ao runner.
5. **`tests/test_transforms.py`**:
   - Corrigir a definição física do teste de inclinação limite ($b = 0 \implies i = 90^\circ$; $b = a/R_* \implies \cos i = 1 \implies i = 0^\circ$ é órbita polar/rasante à linha de visada).
