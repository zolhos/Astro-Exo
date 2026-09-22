"""
Quickstart Demo: Running Astro-Exo with a Tiny Sample (Laptop-Friendly)
Designed to run in under 3 seconds using < 50 MB of RAM on a MacBook Air.
"""

import sys
import time
import os
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from astro_exo.models.transforms import (
    kipping_to_quadratic,
    compute_stellar_density,
    impact_param_to_inclination
)
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.dilution import calculate_max_transit_depth, rule_out_neighbors_as_blends


def main():
    print("=" * 60)
    print("   Astro-Exo: Execução de Amostra Mínima (MacBook Air / 8 GB)")
    print("=" * 60)

    # 1. Criação de uma amostra minúscula em memória:
    # 200 pontos de curva de luz (~6 horas ao redor do trânsito de um planeta)
    np.random.seed(42)
    period = 3.52254    # dias
    t0 = 0.0            # dias
    duration_days = 0.12 # ~2.9 horas
    
    time_pts = np.linspace(-0.15, 0.15, 200)
    noise_sigma = 1.5e-4 # ruído fotométrico de 150 ppm
    
    # Injeta um trânsito de 1% de profundidade (Rp/Rs ~ 0.1)
    in_transit = np.abs(time_pts) < (0.5 * duration_days)
    flux_pts = 1.0 - np.where(in_transit, 0.01, 0.0) + np.random.normal(0, noise_sigma, len(time_pts))
    flux_err = np.full_like(time_pts, noise_sigma)

    bytes_used = time_pts.nbytes + flux_pts.nbytes + flux_err.nbytes
    print(f"\n[1] Dados Fotométricos:")
    print(f"    - Quantidade de pontos: {len(time_pts)}")
    print(f"    - Memória ocupada pelo array: {bytes_used} bytes (< 5 KB!)")

    # 2. Vetting Espacial em TPF Minúsculo (7x7 pixels do TESS)
    print("\n[2] Executando Vetting Espacial (Diferença de Imagem + Centróide):")
    t_start = time.time()

    # Cria matriz sintética de 7x7 pixels
    ny, nx = 7, 7
    flux_tpf = np.full((len(time_pts), ny, nx), 1500.0)
    # Mergulho de 10% no pixel central (3, 3)
    flux_tpf[in_transit, 3, 3] -= 150.0
    err_tpf = np.full_like(flux_tpf, 2.0)

    target_catalog_pixel = (3.0, 3.0)
    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        time_pts, flux_tpf, err_tpf, period, t0, duration_days
    )
    centroid_res = measure_centroid_offset(
        i_diff, sigma_diff, target_catalog_pixel,
        tess_pixel_scale_arcsec=21.0,
        n_mc_perturbations=300
    )
    t_vetting = time.time() - t_start

    print(f"    - Tempo de processamento: {t_vetting * 1000:.1f} milissegundos")
    print(f"    - Centróide medido (x, y): ({centroid_res['x_diff_cen']:.2f}, {centroid_res['y_diff_cen']:.2f})")
    print(f"    - Deslocamento angular: {centroid_res['offset_arcsec']:.2f} arcsec ({centroid_res['offset_significance_sigma']:.1f} sigma)")
    print(f"    - Status do Vetting: {'APROVADO (Alvo Real)' if centroid_res['offset_significance_sigma'] < 3.0 else 'REJEITADO (Falso Positivo)'}")

    # 3. Triagem de Vizinhos Gaia (Diluição Analítica)
    print("\n[3] Triagem de Vizinhos de Fundo (Gaia Dilution Check):")
    target_mag = 10.0
    mock_gaia_neighbors = [
        {"source_id": 1001, "phot_g_mean_mag": 15.5, "dist_arcsec": 18.0},
        {"source_id": 1002, "phot_g_mean_mag": 17.2, "dist_arcsec": 32.0}
    ]
    vetted_stars = rule_out_neighbors_as_blends(
        observed_transit_depth_ppm=10000.0, # 1%
        target_mag=target_mag,
        neighbors=mock_gaia_neighbors
    )
    for star in vetted_stars:
        print(f"    - Estrela Gaia {star['source_id']} (Mag {star['phot_g_mean_mag']:.1f}): "
              f"Max Depth Injetável = {star['max_injected_depth_ppm']:.0f} ppm -> "
              f"{'Descartada como causa (Impossível)' if star['ruled_out'] else 'Pode causar eclipse'}")

    # 4. Derivação de Parâmetros Físicos
    print("\n[4] Parâmetros Físicos e Reparametrizações:")
    u1, u2 = kipping_to_quadratic(0.35, 0.28)
    a_rs = 9.5
    b = 0.4
    inc = impact_param_to_inclination(b, a_rs)
    rho_star = compute_stellar_density(period, a_rs)

    print(f"    - Escurecimento de limbo Kipping -> Quadrático: u1={u1:.3f}, u2={u2:.3f}")
    print(f"    - Inclinação orbital: {inc:.2f} graus")
    print(f"    - Densidade estelar estimada: {rho_star:.2f} g/cm^3")

    print("\n" + "=" * 60)
    print("   Execução concluída com sucesso sem estresse de memória!")
    print("=" * 60)


if __name__ == "__main__":
    main()
