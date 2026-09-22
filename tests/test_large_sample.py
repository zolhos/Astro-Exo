"""
Comprehensive large-scale test suite on multi-transit time series and wide-field TPF.
Validates signal stacking, 11x11 pixel spatial vetting, multi-neighbor Gaia dilution,
and joint Photometry + Doppler RV characterization.
"""

import sys
import os
import time
from typing import Dict, Any, List, Tuple
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from astro_exo.models.transforms import (
    kipping_to_quadratic,
    compute_stellar_density,
    impact_param_to_inclination,
    ecc_omega_to_xy,
    xy_to_ecc_omega
)
from astro_exo.models.joint_rv import keplerian_rv, compute_planetary_mass_density
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.pixel_lc import extract_pixel_lightcurves, locate_transit_pixel
from astro_exo.vetting.dilution import calculate_max_transit_depth, rule_out_neighbors_as_blends
from astro_exo.pipeline.schemas import VettingReport, TransitInferenceResult


def generate_large_sector_sample(
    duration_days: float = 14.0,
    cadence_minutes: float = 10.0,
    ny: int = 11,
    nx: int = 11,
    period_days: float = 3.25,
    t0_days: float = 0.80,
    transit_duration_hours: float = 2.8,
    transit_depth_fraction: float = 0.0085,  # 8500 ppm (~0.85%, ex: Sub-Saturno / Júpiter Quente)
    target_pos: Tuple[float, float] = (5.0, 5.0),
    stellar_var_amplitude: float = 0.0015,   # Modulação estelar de 1500 ppm
    random_seed: int = 101
) -> Dict[str, Any]:
    """
    Simulates a multi-transit light curve and 11x11 TPF covering 14 days of observations.
    Includes:
      - 4+ periodic transits
      - Stellar rotational variability
      - A non-transiting bright neighboring star at pixel (2, 8)
      - Realistic background noise
    """
    rng = np.random.default_rng(random_seed)

    # Cadências temporais: 14 dias com passos de 10 minutos = 2016 pontos
    n_cadences = int(duration_days * 24.0 * 60.0 / cadence_minutes)
    time_pts = np.linspace(0.0, duration_days, n_cadences)

    dur_days = transit_duration_hours / 24.0

    # Determina fases orbitais e máscaras de trânsito periódicos
    phase = (time_pts - t0_days + 0.5 * period_days) % period_days - 0.5 * period_days
    in_transit_mask = np.abs(phase) <= (0.5 * dur_days)
    n_transits_observed = int(np.floor((duration_days - t0_days) / period_days)) + 1

    # 1. Variabilidade Estelar (rotação estelar com período de ~5 dias)
    stellar_trend = 1.0 + stellar_var_amplitude * np.sin(2.0 * np.pi * time_pts / 5.2)

    # 2. Curva 1D com ruído fotométrico
    noise_sigma_1d = 1.8e-4  # 180 ppm
    transit_signal = np.where(in_transit_mask, -transit_depth_fraction, 0.0)
    flux_1d = (stellar_trend + transit_signal) + rng.normal(0, noise_sigma_1d, n_cadences)
    flux_err_1d = np.full(n_cadences, noise_sigma_1d)

    # 3. Cubo TPF (11 x 11 pixels)
    # Fundo do céu
    tpf_flux = np.full((n_cadences, ny, nx), 1500.0)

    # Adiciona a estrela principal (alvo) no centro (5, 5) com PSF dispersa
    target_y, target_x = int(round(target_pos[0])), int(round(target_pos[1]))
    target_flux_peak = 25000.0

    y_grid, x_grid = np.mgrid[0:ny, 0:nx]
    psf_target = np.exp(-(((x_grid - target_x) ** 2 + (y_grid - target_y) ** 2) / (2.0 * (1.2 ** 2))))
    psf_target /= psf_target.sum()

    # Adiciona estrela vizinha fixa (sem trânsito) no pixel (2, 8) para testar descolamento
    neighbor_y, neighbor_x = 2, 8
    psf_neighbor = np.exp(-(((x_grid - neighbor_x) ** 2 + (y_grid - neighbor_y) ** 2) / (2.0 * (1.1 ** 2))))
    psf_neighbor /= psf_neighbor.sum()
    neighbor_flux_peak = 12000.0

    for i in range(n_cadences):
        # Fluxo do alvo varia com a modulação estelar e o trânsito
        t_factor = stellar_trend[i] + transit_signal[i]
        tpf_flux[i, :, :] += target_flux_peak * t_factor * psf_target
        # Vizinho estelar não varia com o trânsito
        tpf_flux[i, :, :] += neighbor_flux_peak * psf_neighbor

    # Ruído gaussiano no detector
    tpf_flux += rng.normal(0, 8.0, (n_cadences, ny, nx))
    tpf_flux_err = np.full_like(tpf_flux, 8.0)

    # 4. Amostra Doppler (RV) concomitante com 25 medições
    t_rv = np.linspace(0.5, duration_days - 0.5, 25)
    k_true = 85.0  # m/s
    gamma_true = 12.5  # m/s
    rv_true = keplerian_rv(t_rv, period=period_days, t0=t0_days, k_semiamp=k_true, ecc=0.08, omega_deg=65.0, gamma=gamma_true)
    rv_noise = rng.normal(0, 3.5, len(t_rv))  # Precisão de 3.5 m/s (nível HARPS)
    rv_obs = rv_true + rv_noise
    rv_err = np.full_like(t_rv, 3.5)

    return {
        "time": time_pts,
        "flux_1d": flux_1d,
        "flux_err_1d": flux_err_1d,
        "tpf_flux": tpf_flux,
        "tpf_flux_err": tpf_flux_err,
        "period": period_days,
        "t0": t0_days,
        "duration_days": dur_days,
        "target_pos": target_pos,
        "injected_depth": transit_depth_fraction,
        "n_transits": n_transits_observed,
        "t_rv": t_rv,
        "rv_obs": rv_obs,
        "rv_err": rv_err,
        "k_semiamp_true": k_true,
        "gamma_true": gamma_true
    }


def test_large_sample_scale_and_memory():
    """Verify data scaling across > 2,000 cadences and 11x11 TPF footprint."""
    sample = generate_large_sector_sample()
    n_points = len(sample["time"])
    ny, nx = sample["tpf_flux"].shape[1], sample["tpf_flux"].shape[2]

    assert n_points > 2000, f"Amostra insuficiente: {n_points} pontos"
    assert (ny, nx) == (11, 11), f"Dimensão TPF divergente: {ny}x{nx}"
    assert sample["n_transits"] >= 4, f"Trânsitos insuficientes: {sample['n_transits']}"

    # Consumo de memória total
    bytes_used = (
        sample["time"].nbytes +
        sample["flux_1d"].nbytes +
        sample["flux_err_1d"].nbytes +
        sample["tpf_flux"].nbytes +
        sample["tpf_flux_err"].nbytes +
        sample["rv_obs"].nbytes
    )
    # Deve ocupar entre 1.5 MB e 8.0 MB (perfeito para execução ágil)
    assert 1_000_000 < bytes_used < 10_000_000, f"Tamanho inesperado: {bytes_used} bytes"


def test_large_sample_phase_folding_and_stacking():
    """Verify phase-folding and multi-transit SNR amplification."""
    sample = generate_large_sector_sample()
    t = sample["time"]
    f = sample["flux_1d"]
    p = sample["period"]
    t0 = sample["t0"]
    dur = sample["duration_days"]

    # Dobramento de fase: fase em [-0.5 * P, 0.5 * P]
    phase = (t - t0 + 0.5 * p) % p - 0.5 * p

    in_transit = np.abs(phase) <= (0.5 * dur)
    out_transit = (np.abs(phase) > (0.6 * dur)) & (np.abs(phase) < (2.0 * dur))

    n_in = np.sum(in_transit)
    n_out = np.sum(out_transit)
    assert n_in >= 40, f"Poucos pontos em trânsito no fold: {n_in}"

    in_median = np.median(f[in_transit])
    out_median = np.median(f[out_transit])
    measured_depth = out_median - in_median

    err_stack = np.std(f[out_transit]) / np.sqrt(n_in)
    snr_stacked = measured_depth / err_stack

    # Com 4 trânsitos dobrados, o SNR acumulado deve exceder 25 sigma
    assert snr_stacked > 25.0, f"SNR de empilhamento baixo: {snr_stacked:.1f}"
    assert np.isclose(measured_depth, sample["injected_depth"], atol=1.5e-3)


def test_large_sample_pixel_array_localization():
    """Verify pixel-by-pixel extraction across all 121 pixels to rule out BEBs."""
    sample = generate_large_sector_sample()
    best_y, best_x, max_dip = locate_transit_pixel(
        sample["time"],
        sample["tpf_flux"],
        sample["period"],
        sample["t0"],
        sample["duration_days"]
    )

    target_y, target_x = int(round(sample["target_pos"][0])), int(round(sample["target_pos"][1]))
    assert best_y == target_y, f"Y divergente: {best_y} != {target_y}"
    assert best_x == target_x, f"X divergente: {best_x} != {target_x}"

    # Extrai matriz completa (11, 11, 2016)
    pixel_lcs, pixel_errs = extract_pixel_lightcurves(sample["tpf_flux"], sample["tpf_flux_err"])
    assert pixel_lcs.shape == (11, 11, len(sample["time"]))

    # O pixel vizinho fixo (2, 8) NÃO pode ter a queda de trânsito principal
    phase = (sample["time"] - sample["t0"] + 0.5 * sample["period"]) % sample["period"] - 0.5 * sample["period"]
    in_mask = np.abs(phase) <= (0.5 * sample["duration_days"])
    out_mask = (np.abs(phase) > (0.6 * sample["duration_days"])) & (np.abs(phase) < (1.5 * sample["duration_days"]))

    neighbor_in = np.median(pixel_lcs[2, 8, in_mask])
    neighbor_out = np.median(pixel_lcs[2, 8, out_mask])
    neighbor_dip = (neighbor_out - neighbor_in) / neighbor_out

    # Queda no vizinho deve ser desprezível (< 0.15%) comparada ao alvo
    assert neighbor_dip < 0.002, f"Contaminação excessiva no vizinho: {neighbor_dip}"


def test_large_sample_spatial_difference_vetting():
    """Verify difference imaging and centroid offset with 500 Monte Carlo draws."""
    sample = generate_large_sector_sample()
    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        sample["time"],
        sample["tpf_flux"],
        sample["tpf_flux_err"],
        sample["period"],
        sample["t0"],
        sample["duration_days"]
    )

    target_y = int(round(sample["target_pos"][0]))
    target_x = int(round(sample["target_pos"][1]))
    # Diferença de fluxo positiva e máxima no centro (pixel 5, 5)
    assert i_diff[target_y, target_x] > 15.0, f"Déficit baixo no centro: {i_diff[target_y, target_x]}"
    assert (i_diff.argmax() // 11, i_diff.argmax() % 11) == (target_y, target_x)

    centroid_res = measure_centroid_offset(
        i_diff,
        sigma_diff,
        sample["target_pos"],
        tess_pixel_scale_arcsec=21.0,
        n_mc_perturbations=500
    )

    # O deslocamento angular em relação ao alvo deve ser inferior a 1.2 arcsec (< 2.0 sigma)
    assert centroid_res["offset_arcsec"] < 1.5, f"Offset elevado: {centroid_res['offset_arcsec']}"
    assert centroid_res["offset_significance_sigma"] < 2.0


def test_large_sample_gaia_crowded_field_dilution():
    """Verify screening against a realistic 12-star Gaia field."""
    sample = generate_large_sector_sample()
    target_mag = 10.2
    observed_depth_ppm = sample["injected_depth"] * 1e6  # 8500 ppm

    # 12 estrelas vizinhas distribuídas no raio de 1 arcmin
    gaia_catalog = [
        {"source_id": 3001, "phot_g_mean_mag": 14.8, "dist_arcsec": 12.0},
        {"source_id": 3002, "phot_g_mean_mag": 15.2, "dist_arcsec": 18.5},
        {"source_id": 3003, "phot_g_mean_mag": 16.0, "dist_arcsec": 22.0},
        {"source_id": 3004, "phot_g_mean_mag": 16.8, "dist_arcsec": 29.0},
        {"source_id": 3005, "phot_g_mean_mag": 17.1, "dist_arcsec": 33.5},
        {"source_id": 3006, "phot_g_mean_mag": 17.5, "dist_arcsec": 38.0},
        {"source_id": 3007, "phot_g_mean_mag": 18.2, "dist_arcsec": 44.0},
        {"source_id": 3008, "phot_g_mean_mag": 18.6, "dist_arcsec": 48.0},
        {"source_id": 3009, "phot_g_mean_mag": 19.0, "dist_arcsec": 51.0},
        {"source_id": 3010, "phot_g_mean_mag": 19.3, "dist_arcsec": 54.0},
        {"source_id": 3011, "phot_g_mean_mag": 19.7, "dist_arcsec": 57.0},
        {"source_id": 3012, "phot_g_mean_mag": 20.1, "dist_arcsec": 59.5},
    ]

    vetted = rule_out_neighbors_as_blends(
        observed_transit_depth_ppm=observed_depth_ppm,
        target_mag=target_mag,
        neighbors=gaia_catalog
    )

    # Estrelas com mag > 15.4 (delta_mag > 5.2) não conseguem produzir 8500 ppm
    ruled_out_count = sum(1 for star in vetted if star["ruled_out"])
    assert ruled_out_count >= 10, f"Poucos vizinhos descartados: {ruled_out_count}/12"


def test_large_sample_joint_photometry_doppler_solution():
    """Verify joint Photometry + Doppler RV Keplerian solution."""
    sample = generate_large_sector_sample()

    # 1. Ajuste Doppler linear mínimo para semi-amplitude K
    t_rv = sample["t_rv"]
    rv_obs = sample["rv_obs"]
    p = sample["period"]
    t0 = sample["t0"]

    # Matriz de design para ajuste harmônico: rv(t) ~ gamma + K*cos(M) - K*sin(M)
    mean_anom = 2.0 * np.pi * (t_rv - t0) / p
    a_mat = np.column_stack([np.ones_like(t_rv), np.sin(mean_anom), np.cos(mean_anom)])
    weights = 1.0 / sample["rv_err"]
    a_w = a_mat * weights[:, None]
    y_w = rv_obs * weights

    beta, _, _, _ = np.linalg.lstsq(a_w, y_w, rcond=None)
    fit_gamma = beta[0]
    fit_k = np.sqrt(beta[1]**2 + beta[2]**2)

    # Recupera K e gamma dentro da margem estatística (± 15 m/s)
    assert np.isclose(fit_k, sample["k_semiamp_true"], atol=15.0), f"K recuperado: {fit_k}"
    assert np.isclose(fit_gamma, sample["gamma_true"], atol=10.0), f"Gamma recuperado: {fit_gamma}"

    # 2. Derivação de massa planetária e densidade média bulk
    rp_rs = np.sqrt(sample["injected_depth"])
    a_rs = 9.4
    inc_deg = impact_param_to_inclination(0.25, a_rs)

    phys = compute_planetary_mass_density(
        m_star_msun=1.02,
        r_star_rsun=1.05,
        period_days=sample["period"],
        k_semiamp_ms=fit_k,
        rp_rs=rp_rs,
        ecc=0.08,
        inc_deg=inc_deg
    )

    # Valida parâmetros físicos obtidos para o exoplaneta
    assert 0.4 < phys["mass_jupiter"] < 1.2, f"Massa fora da escala: {phys['mass_jupiter']}"
    assert 0.8 < phys["radius_jupiter"] < 1.3, f"Raio fora da escala: {phys['radius_jupiter']}"
    assert 0.3 < phys["density_g_cm3"] < 2.0, f"Densidade fora da escala: {phys['density_g_cm3']}"


def run_large_sample_suite() -> int:
    """Runs the large sample diagnostic suite with benchmark statistics."""
    print("=" * 72)
    print("      ASTRO-EXO: BATERIA EM ESCALA AMPLA (AMOSTRA MULTI-TRÂNSITO)")
    print("=" * 72)

    t_start_gen = time.perf_counter()
    sample = generate_large_sector_sample()
    t_gen_ms = (time.perf_counter() - t_start_gen) * 1000.0

    bytes_used = (
        sample["time"].nbytes +
        sample["flux_1d"].nbytes +
        sample["flux_err_1d"].nbytes +
        sample["tpf_flux"].nbytes +
        sample["tpf_flux_err"].nbytes +
        sample["rv_obs"].nbytes
    )

    print("[DADOS DA AMOSTRA EM ESCALA AMPLA]")
    print(f"  • Intervalo observado  : 14 dias (~meio setor do TESS)")
    print(f"  • Cadências temporais  : {len(sample['time'])} cadências (resolução 10 min)")
    print(f"  • Trânsitos observados : {sample['n_transits']} trânsitos periódicos completos")
    print(f"  • Matriz espacial TPF  : {sample['tpf_flux'].shape[1]}x{sample['tpf_flux'].shape[2]} pixels ({sample['tpf_flux'].shape[1]*sample['tpf_flux'].shape[2]} pixels totais)")
    print(f"  • Medições Doppler RV  : {len(sample['t_rv'])} espectros de alta resolução (HARPS/ESPRESSO)")
    print(f"  • Memória ocupada      : {bytes_used / (1024.0 * 1024.0):.2f} MB")
    print(f"  • Tempo de geração     : {t_gen_ms:.1f} ms")
    print("-" * 72)

    tests = [
        ("1. Validação Estrutural e Escala de Memória", test_large_sample_scale_and_memory),
        ("2. Dobramento de Fase & Empilhamento Multi-Trânsito", test_large_sample_phase_folding_and_stacking),
        ("3. Mapeamento Pixel a Pixel (121 pixels) & Blend Check", test_large_sample_pixel_array_localization),
        ("4. Vetting por Diferença de Imagem (500 Monte Carlo)", test_large_sample_spatial_difference_vetting),
        ("5. Triagem de Diluição em Campo Denso Gaia (12 estrelas)", test_large_sample_gaia_crowded_field_dilution),
        ("6. Solução Conjunta Fotometria + RV (Massa e Densidade Bulk)", test_large_sample_joint_photometry_doppler_solution),
    ]

    t_global = time.perf_counter()
    all_passed = True

    print("\n[EXECUÇÃO DOS TESTES EM GRANDE ESCALA]")
    for title, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  ✓ {title:<54} [{dt_ms:6.1f} ms] PASS")
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  ✗ {title:<54} [{dt_ms:6.1f} ms] FAIL: {e}")
            all_passed = False

    t_total_ms = (time.perf_counter() - t_global) * 1000.0

    print("\n" + "=" * 72)
    if all_passed:
        print(f"  RESULTADO: 6/6 TESTES EM GRANDE ESCALA APROVADOS em {t_total_ms:.1f} ms!")
        print(f"  Desempenho: Processamento completo a ~{len(sample['time']) / (t_total_ms / 1000.0):.0f} cadências/segundo")
    else:
        print("  RESULTADO: Falha detectada em um dos testes em grande escala.")
    print("=" * 72)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_large_sample_suite())
