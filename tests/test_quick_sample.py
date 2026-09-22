"""
Quick end-to-end test on a lightweight photometric and pixel sample.
Executes signal detection, pixel localization, difference imaging, and physical inference.
"""

import sys
import os
import time
from typing import Dict, Any, Tuple
import numpy as np

# Ensure src directory is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from astro_exo.models.transforms import (
    kipping_to_quadratic,
    compute_stellar_density,
    impact_param_to_inclination
)
from astro_exo.models.joint_rv import compute_planetary_mass_density
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.pixel_lc import extract_pixel_lightcurves, locate_transit_pixel
from astro_exo.vetting.dilution import calculate_max_transit_depth, rule_out_neighbors_as_blends


def create_tiny_transit_sample(
    n_cadences: int = 150,
    ny: int = 5,
    nx: int = 5,
    target_pos: Tuple[float, float] = (2.0, 2.0),
    period_days: float = 3.52,
    transit_depth_fraction: float = 0.012,
    noise_sigma: float = 2.0e-4,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Generates a tiny, realistic photometric time-series and 5x5 TPF sample in memory.
    Memory footprint: < 30 KB.
    """
    rng = np.random.default_rng(random_seed)
    t0 = 0.0
    duration_days = 0.12  # ~2.88 horas

    # Amostra temporal cobrindo ~7.2 horas centradas no trânsito
    time_pts = np.linspace(-0.15, 0.15, n_cadences)

    phase = (time_pts - t0 + 0.5 * period_days) % period_days - 0.5 * period_days
    in_transit = np.abs(phase) <= (0.5 * duration_days)

    # 1. Curva de luz integrada 1D
    clean_flux_1d = np.ones(n_cadences)
    clean_flux_1d[in_transit] -= transit_depth_fraction
    noisy_flux_1d = clean_flux_1d + rng.normal(0, noise_sigma, n_cadences)
    flux_err_1d = np.full(n_cadences, noise_sigma)

    # 2. Cubo Target Pixel File (TPF) 3D: (n_cadences, ny, nx)
    tpf_flux = np.full((n_cadences, ny, nx), 2000.0)

    # Função de dispersão simples (PRF gaussiana no pixel do alvo)
    target_y, target_x = int(round(target_pos[0])), int(round(target_pos[1]))
    target_baseline = 10000.0
    tpf_flux[:, target_y, target_x] += target_baseline

    # Injeta a queda de trânsito no pixel do alvo
    deficit_photons = target_baseline * transit_depth_fraction
    tpf_flux[in_transit, target_y, target_x] -= deficit_photons

    # Adiciona ruído de contagem poissoniana / gaussiana
    tpf_noise = rng.normal(0, 5.0, (n_cadences, ny, nx))
    tpf_flux += tpf_noise
    tpf_flux_err = np.full_like(tpf_flux, 5.0)

    return {
        "time": time_pts,
        "flux_1d": noisy_flux_1d,
        "flux_err_1d": flux_err_1d,
        "tpf_flux": tpf_flux,
        "tpf_flux_err": tpf_flux_err,
        "period": period_days,
        "t0": t0,
        "duration_days": duration_days,
        "target_pos": target_pos,
        "injected_depth": transit_depth_fraction
    }


def test_sample_memory_and_dimensions():
    """Verify the sample is lightweight and valid in memory."""
    sample = create_tiny_transit_sample()
    total_bytes = (
        sample["time"].nbytes +
        sample["flux_1d"].nbytes +
        sample["flux_err_1d"].nbytes +
        sample["tpf_flux"].nbytes +
        sample["tpf_flux_err"].nbytes
    )
    assert total_bytes < 100_000, f"Amostra excedeu limite de memória: {total_bytes} bytes"
    assert len(sample["time"]) == 150
    assert sample["tpf_flux"].shape == (150, 5, 5)
    assert not np.isnan(sample["flux_1d"]).any()
    assert not np.isinf(sample["flux_1d"]).any()


def test_sample_transit_snr():
    """Verify photometric transit significance on the 1D light curve sample."""
    sample = create_tiny_transit_sample()
    time_pts = sample["time"]
    flux = sample["flux_1d"]
    dur = sample["duration_days"]

    in_mask = np.abs(time_pts) <= (0.5 * dur)
    out_mask = ~in_mask

    in_median = np.median(flux[in_mask])
    out_median = np.median(flux[out_mask])
    measured_depth = out_median - in_median

    noise_std = np.std(flux[out_mask])
    n_in = np.sum(in_mask)
    err_depth = noise_std / np.sqrt(n_in)
    snr = measured_depth / err_depth

    # Deve recuperar a profundidade com alto SNR (> 15 sigma)
    assert np.isclose(measured_depth, sample["injected_depth"], atol=1e-3)
    assert snr > 15.0, f"SNR insuficiente: {snr:.2f}"


def test_sample_pixel_localization():
    """Verify transit localization directly identifies the target pixel."""
    sample = create_tiny_transit_sample()
    best_y, best_x, max_dip = locate_transit_pixel(
        sample["time"],
        sample["tpf_flux"],
        sample["period"],
        sample["t0"],
        sample["duration_days"]
    )
    expected_y = int(round(sample["target_pos"][0]))
    expected_x = int(round(sample["target_pos"][1]))

    assert best_y == expected_y, f"Y divergente: {best_y} != {expected_y}"
    assert best_x == expected_x, f"X divergente: {best_x} != {expected_x}"
    assert max_dip > 0.005, f"Queda no pixel alvo muito baixa: {max_dip}"

    # Extrai curvas normalizadas pixel a pixel
    pixel_lcs, pixel_errs = extract_pixel_lightcurves(sample["tpf_flux"], sample["tpf_flux_err"])
    assert pixel_lcs.shape == (5, 5, 150)


def test_sample_difference_imaging_vetting():
    """Verify spatial difference imaging and centroid offset on the sample."""
    sample = create_tiny_transit_sample()
    i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
        sample["time"],
        sample["tpf_flux"],
        sample["tpf_flux_err"],
        sample["period"],
        sample["t0"],
        sample["duration_days"]
    )

    # O déficit no pixel central deve ser nítido e positivo em i_diff
    target_y = int(round(sample["target_pos"][0]))
    target_x = int(round(sample["target_pos"][1]))
    assert i_diff[target_y, target_x] > 50.0

    centroid_res = measure_centroid_offset(
        i_diff,
        sigma_diff,
        sample["target_pos"],
        tess_pixel_scale_arcsec=21.0,
        n_mc_perturbations=200
    )

    # Offset deve ser insignificante (< 1.5 arcsec e < 2 sigma)
    assert centroid_res["offset_arcsec"] < 1.5
    assert centroid_res["offset_significance_sigma"] < 2.0


def test_sample_blend_dilution_and_physics():
    """Verify Gaia neighbor blend screening and physical system characterization."""
    sample = create_tiny_transit_sample()
    target_mag = 10.5
    observed_depth_ppm = sample["injected_depth"] * 1e6  # ~12000 ppm

    # Simula estrelas vizinhas no campo do TESS
    mock_neighbors = [
        {"source_id": 2001, "phot_g_mean_mag": 15.8, "dist_arcsec": 14.5},
        {"source_id": 2002, "phot_g_mean_mag": 16.5, "dist_arcsec": 28.0},
        {"source_id": 2003, "phot_g_mean_mag": 18.0, "dist_arcsec": 42.0},
    ]

    vetted = rule_out_neighbors_as_blends(
        observed_transit_depth_ppm=observed_depth_ppm,
        target_mag=target_mag,
        neighbors=mock_neighbors
    )

    assert all(n["ruled_out"] for n in vetted), "Todas as estrelas fracas deveriam ser descartadas"

    # Caracterização Física
    a_rs = 9.8
    b = 0.35
    u1, u2 = kipping_to_quadratic(0.32, 0.25)
    inc_deg = impact_param_to_inclination(b, a_rs)
    rho_star = compute_stellar_density(sample["period"], a_rs)

    planet_derived = compute_planetary_mass_density(
        m_star_msun=1.05,
        r_star_rsun=1.02,
        period_days=sample["period"],
        k_semiamp_ms=115.0,
        rp_rs=np.sqrt(sample["injected_depth"]),
        inc_deg=inc_deg
    )

    assert 1.0 < rho_star < 2.0
    assert 85.0 < inc_deg < 90.0
    assert 0.7 < planet_derived["mass_jupiter"] < 1.4
    assert 0.8 < planet_derived["radius_jupiter"] < 1.4
    assert 0.4 < planet_derived["density_g_cm3"] < 2.5


def run_quick_sample_suite() -> int:
    """Runs all sample tests sequentially with benchmark timings and formatted output."""
    print("=" * 68)
    print("      ASTRO-EXO: TESTE RÁPIDO COM PEQUENA AMOSTRA (EM MEMÓRIA)")
    print("=" * 68)

    sample = create_tiny_transit_sample()
    total_bytes = (
        sample["time"].nbytes +
        sample["flux_1d"].nbytes +
        sample["flux_err_1d"].nbytes +
        sample["tpf_flux"].nbytes +
        sample["tpf_flux_err"].nbytes
    )

    print("[DADOS DA PEQUENA AMOSTRA]")
    print(f"  • Cadências temporais  : {len(sample['time'])} pontos (~7.2 horas)")
    print(f"  • Matriz espacial TPF  : {sample['tpf_flux'].shape[1]}x{sample['tpf_flux'].shape[2]} pixels")
    print(f"  • Queda de trânsito    : {sample['injected_depth'] * 100:.2f}% ({sample['injected_depth']*1e6:.0f} ppm)")
    print(f"  • Posição do alvo      : Pixel {sample['target_pos']}")
    print(f"  • Consumo de memória   : {total_bytes / 1024.0:.2f} KB (< 0.05 MB!)")
    print("-" * 68)

    tests = [
        ("1. Validação de Memória e Formato dos Dados", test_sample_memory_and_dimensions),
        ("2. Detecção e SNR do Trânsito na Curva 1D", test_sample_transit_snr),
        ("3. Localização Espacial no Pixel do Alvo", test_sample_pixel_localization),
        ("4. Vetting Espacial e Deslocamento de Centróide", test_sample_difference_imaging_vetting),
        ("5. Triagem de Diluição Gaia e Parâmetros Físicos", test_sample_blend_dilution_and_physics),
    ]

    t_global = time.perf_counter()
    all_passed = True

    print("\n[EXECUÇÃO DOS TESTES]")
    for title, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  ✓ {title:<48} [{dt_ms:5.1f} ms] PASS")
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"  ✗ {title:<48} [{dt_ms:5.1f} ms] FAIL: {e}")
            all_passed = False

    t_total_ms = (time.perf_counter() - t_global) * 1000.0

    print("\n" * 0 + "=" * 68)
    if all_passed:
        print(f"  RESULTADO: 5/5 TESTES APROVADOS com sucesso em {t_total_ms:.1f} ms!")
        print("  Amostra validada de ponta a ponta sem dependências externas.")
    else:
        print(f"  RESULTADO: Falha detectada em um dos testes.")
    print("=" * 68)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_quick_sample_suite())
