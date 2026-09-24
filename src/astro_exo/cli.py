"""
Command-Line Interface (CLI) for the Astro-Exo pipeline.
High-Precision Bayesian Inference, Spatial Vetting, and Radial Velocity Modeling.
"""

import os
import sys
import json
import argparse
import webbrowser
import numpy as np

from astro_exo import __version__
from astro_exo.pipeline.config import TargetConfig, PipelineConfig
from astro_exo.pipeline.runner import ExoplanetPipelineRunner


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="astro-exo",
        description="Astro-Exo v1.0.0: Exoplanet Discovery, Spatial Vetting & Bayesian Inference Pipeline."
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit"
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Pipeline subcommands")

    # =========================================================================
    # Subcommand: run
    # =========================================================================
    run_parser = subparsers.add_parser("run", help="Run full pipeline on a target TIC ID")
    run_parser.add_argument("--tic", type=int, required=True, help="TESS Input Catalog (TIC) ID")
    run_parser.add_argument("--sector", type=int, default=None, help="TESS observing sector")
    run_parser.add_argument("--period", type=float, default=None, help="Orbital period in days")
    run_parser.add_argument("--t0", type=float, default=None, help="Time of mid-transit in BJD/BTJD")
    run_parser.add_argument("--duration", type=float, default=None, help="Transit duration in hours")
    run_parser.add_argument("--backend", choices=["emcee", "jax_nuts"], default="emcee", help="Sampling backend (emcee or jax_nuts)")
    run_parser.add_argument("--enable-gp", action="store_true", help="Enable Gaussian Process correlated noise modeling")
    run_parser.add_argument("--no-vetting", dest="run_vetting", action="store_false", help="Bypass spatial vetting")
    run_parser.set_defaults(run_vetting=True)
    run_parser.add_argument("--outdir", type=str, default="results", help="Output directory for products and diagnostics")

    # =========================================================================
    # Subcommand: vet
    # =========================================================================
    vet_parser = subparsers.add_parser("vet", help="Run spatial difference imaging and Gaia blend vetting")
    vet_parser.add_argument("--tic", type=int, required=True, help="TIC ID of target")
    vet_parser.add_argument("--sector", type=int, default=None, help="TESS sector")
    vet_parser.add_argument("--period", type=float, required=True, help="Orbital period in days")
    vet_parser.add_argument("--t0", type=float, required=True, help="Epoch of mid-transit in BJD/BTJD")
    vet_parser.add_argument("--duration", type=float, default=3.0, help="Transit duration in hours (default: 3.0)")
    vet_parser.add_argument("--outdir", type=str, default="results/vetting", help="Output directory")

    # =========================================================================
    # Subcommand: joint-rv
    # =========================================================================
    rv_parser = subparsers.add_parser("joint-rv", help="Run multi-instrument Doppler Radial Velocity Keplerian modeling")
    rv_parser.add_argument("--target", "--name", dest="target", type=str, default="Target", help="Exoplanet/target name")
    rv_parser.add_argument("--rv-file", type=str, default=None, help="Path to RV CSV data file (time, rv, rv_err, instrument)")
    rv_parser.add_argument("--period", type=float, default=None, help="Orbital period in days")
    rv_parser.add_argument("--t0", type=float, default=None, help="Epoch of mid-transit in BJD")
    rv_parser.add_argument("--m-star", type=float, default=1.0, help="Stellar mass in solar masses (default: 1.0)")
    rv_parser.add_argument("--r-star", type=float, default=1.0, help="Stellar radius in solar radii (default: 1.0)")
    rv_parser.add_argument("--rp-rs", type=float, default=None, help="Planet-to-star radius ratio from transit fit")
    rv_parser.add_argument("--nwalkers", type=int, default=28, help="Number of MCMC walkers (default: 28)")
    rv_parser.add_argument("--nburn", type=int, default=150, help="Burn-in steps (default: 150)")
    rv_parser.add_argument("--nsteps", type=int, default=350, help="Production steps (default: 350)")
    rv_parser.add_argument("--outdir", type=str, default="results/joint_rv", help="Output directory")
    rv_parser.add_argument("--demo", action="store_true", help="Run benchmark demo on WASP-77b with multi-spectrograph data")

    # =========================================================================
    # Subcommand: batch
    # =========================================================================
    batch_parser = subparsers.add_parser("batch", help="Run batch processing across a candidate catalog")
    batch_parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV or JSON catalog")
    batch_parser.add_argument("--outdir", "--output-dir", "-o", dest="outdir", type=str, default="results/batch_run", help="Output directory")
    batch_parser.add_argument("--mode", "-m", choices=["mock", "live"], default="mock", help="Mode: 'mock' (offline/synthetic) or 'live' (MAST)")

    # =========================================================================
    # Subcommand: dashboard
    # =========================================================================
    dash_parser = subparsers.add_parser("dashboard", help="Generate and open the interactive scientific web dashboard")
    dash_parser.add_argument("--results-dir", type=str, default="results/rodada_amostras_ineditas", help="Path to results directory")
    dash_parser.add_argument("--output", "-o", type=str, default=None, help="Path for generated HTML dashboard")
    dash_parser.add_argument("--open", dest="open_browser", action="store_true", help="Automatically open dashboard in browser")

    # =========================================================================
    # Subcommand: fetch-tois
    # =========================================================================
    fetch_parser = subparsers.add_parser("fetch-tois", help="Fetch candidates from NASA Exoplanet Archive with quota protection")
    fetch_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of TOIs to fetch")
    fetch_parser.add_argument("--output", "-o", type=str, default="data/nasa_tois_batch.csv", help="Output CSV path")
    fetch_parser.add_argument("--refresh", action="store_true", help="Force refresh bypassing local 24h cache")

    # =========================================================================
    # Subcommand: smoke
    # =========================================================================
    subparsers.add_parser("smoke", help="Run complete module diagnostics and functional smoke tests")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Handle: run
    # -------------------------------------------------------------------------
    if args.subcommand == "run":
        target = TargetConfig(
            tic_id=args.tic,
            sector=args.sector,
            period_days=args.period,
            t0_bjd=args.t0,
            duration_hours=args.duration
        )
        config = PipelineConfig(
            sampler_backend=args.backend,
            run_vetting=args.run_vetting,
            enable_gp=args.enable_gp,
            output_dir=args.outdir
        )
        runner = ExoplanetPipelineRunner(target, config)
        product = runner.run()
        print(f"\n[SUCESSO] Análise concluída para TIC {args.tic}!")
        print(f"Status Final: {product.disposition}")
        if product.inference and product.inference.rp_rs:
            print(f"Rp/Rs: {product.inference.rp_rs:.4f} ± {product.inference.rp_rs_err:.4f}")
        return 0

    # -------------------------------------------------------------------------
    # Handle: vet
    # -------------------------------------------------------------------------
    elif args.subcommand == "vet":
        print(f"[VETTING] Iniciando triagem espacial para TIC {args.tic}...")
        from astro_exo.ingestion.mast_tess import fetch_tess_tpf
        from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
        from astro_exo.vetting.gaia import query_gaia_neighbors
        from astro_exo.vetting.dilution import calculate_critical_delta_mag

        tpf = fetch_tess_tpf(args.tic, sector=args.sector)
        i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
            tpf.time.value, tpf.flux.value, tpf.flux_err.value,
            args.period, args.t0, args.duration / 24.0
        )
        pix_x, pix_y = tpf.wcs.all_world2pix(tpf.ra, tpf.dec, 0)
        res = measure_centroid_offset(i_diff, sigma_diff, (pix_x, pix_y))

        print("\n========================================================")
        print("          ASTRO-EXO: RESULTADO DE VETTING ESPACIAL      ")
        print("========================================================")
        print(f"Offset Astrométrico       : {res['offset_arcsec']:.2f}\"")
        print(f"Significância Estatística : {res['offset_significance_sigma']:.1f}σ")
        status = "PASSED" if res['offset_significance_sigma'] < 3.0 else "REJECTED_FP"
        print(f"Decisão de Centróide      : {status}")
        print("========================================================")
        return 0

    # -------------------------------------------------------------------------
    # Handle: joint-rv
    # -------------------------------------------------------------------------
    elif args.subcommand == "joint-rv":
        from astro_exo.ingestion.rv_loader import load_rv_csv
        from astro_exo.models.joint_rv import JointTransitRVSampler

        target_name = args.target
        rv_file = args.rv_file
        period = args.period
        t0 = args.t0
        m_star = args.m_star
        r_star = args.r_star
        rp_rs = args.rp_rs

        # Demo Mode Fallback
        if args.demo or (not rv_file and not period):
            print("[JOINT-RV] Executando modo DEMO com o sistema de referência WASP-77b...")
            target_name = "WASP-77b"
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            rv_file = os.path.join(repo_root, "data", "rv_data", "wasp77_rv.csv")
            period = 1.36003
            t0 = 2456200.5
            m_star = 1.00
            r_star = 0.95
            rp_rs = 0.1324  # Valor de-diluído de Fase 4

        if not rv_file or not os.path.isfile(rv_file):
            print(f"[ERRO] Arquivo RV não encontrado: {rv_file}", file=sys.stderr)
            return 1
        if period is None or t0 is None:
            print("[ERRO] Os parâmetros --period e --t0 são obrigatórios.", file=sys.stderr)
            return 1

        print(f"[JOINT-RV] Carregando observações espectroscópicas de: {rv_file}")
        rv_dataset = load_rv_csv(rv_file)
        print(f"           {rv_dataset.n_points} pontos observados | Espectrógrafos: {rv_dataset.instrument_names}")

        sampler = JointTransitRVSampler(
            rv_dataset=rv_dataset,
            period_days=period,
            t0_bjd=t0,
            m_star_msun=m_star,
            r_star_rsun=r_star,
            rp_rs_prior=rp_rs
        )

        print(f"[JOINT-RV] Executando amostragem Bayesiana MCMC ({args.nwalkers} walkers, {args.nsteps} passos)...")
        fit_res = sampler.run_rv_mcmc(nwalkers=args.nwalkers, nburn=args.nburn, nsteps=args.nsteps, random_seed=42)
        phys = fit_res["physical"]
        k_fit = fit_res["parameters"]["k_semiamp"]

        print("\n========================================================")
        print(f"   ASTRO-EXO: PARÂMETROS FÍSICOS & EQUAÇÃO DE ESTADO   ")
        print("========================================================")
        print(f"Alvo                     : {target_name}")
        print(f"Semi-amplitude K         : {k_fit['median']:.2f} ± {k_fit['error_plus']:.2f} m/s")
        print(f"Massa Planetária (Mp)    : {phys['mass_jupiter']:.3f} ± {phys['mass_jupiter_err']:.3f} M_Jup ({phys['mass_earth']:.1f} M_⊕)")
        print(f"Raio Planetário (Rp)     : {phys['radius_jupiter']:.3f} R_Jup ({phys['radius_earth']:.2f} R_⊕)")
        print(f"Densidade Média (ρp)     : {phys['density_g_cm3']:.2f} ± {phys.get('density_g_cm3_err', 0.0):.2f} g/cm³")
        print(f"Gravidade Superfície logg: {phys.get('log_g_cgs', phys.get('logg_cgs', 0.0)):.2f} [cgs]")
        print(f"Velocidade de Escape vesc: {phys.get('v_esc_kms', phys.get('escape_velocity_km_s', 0.0)):.1f} km/s")
        print(f"Classificação Estrutura  : {phys['interior_classification']}")
        print(f"Chi² Reduzido            : {fit_res['reduced_chi2']:.2f}")
        print("========================================================")

        os.makedirs(args.outdir, exist_ok=True)
        summary_out = os.path.join(args.outdir, f"{target_name.lower().replace('-', '_')}_rv_fit.json")
        with open(summary_out, "w", encoding="utf-8") as f:
            json_safe = {
                "target": target_name,
                "parameters": fit_res["parameters"],
                "physical": phys,
                "reduced_chi2": fit_res["reduced_chi2"],
                "rms_residuals_ms": fit_res["rms_residuals_ms"]
            }
            json.dump(json_safe, f, indent=2)
        print(f"[ARTEFATO] Resultados salvos em: {summary_out}")
        return 0

    # -------------------------------------------------------------------------
    # Handle: batch
    # -------------------------------------------------------------------------
    elif args.subcommand == "batch":
        from astro_exo.pipeline.batch import BatchProcessor
        processor = BatchProcessor(mode=args.mode, output_dir=args.outdir)
        processor.process_catalog(args.input)
        return 0

    # -------------------------------------------------------------------------
    # Handle: dashboard
    # -------------------------------------------------------------------------
    elif args.subcommand == "dashboard":
        from astro_exo.pipeline.dashboard import build_scientific_dashboard
        out_path = build_scientific_dashboard(results_dir=args.results_dir, output_html_path=args.output)
        print(f"\n[SUCESSO] Portal & Dashboard Científico Astro-Exo gerado em:")
        print(f"          {out_path}")
        if args.open_browser:
            print("[NAVEGADOR] Abrindo dashboard no navegador padrão...")
            webbrowser.open(f"file://{out_path}")
        return 0

    # -------------------------------------------------------------------------
    # Handle: fetch-tois
    # -------------------------------------------------------------------------
    elif args.subcommand == "fetch-tois":
        from astro_exo.ingestion.nasa_archive import fetch_nasa_tois, export_tois_to_csv
        tois = fetch_nasa_tois(limit=args.limit, force_refresh=args.refresh)
        out_path = export_tois_to_csv(tois, args.output)
        print(f"[SUCESSO] {len(tois)} candidatos da NASA salvos em: {out_path}")
        return 0

    # -------------------------------------------------------------------------
    # Handle: smoke
    # -------------------------------------------------------------------------
    elif args.subcommand == "smoke":
        from astro_exo.smoke import run_full_diagnostics_and_smoke
        return run_full_diagnostics_and_smoke()

    return 0


if __name__ == "__main__":
    sys.exit(main())
