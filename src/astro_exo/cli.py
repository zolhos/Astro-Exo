"""
Command-Line Interface (CLI) for the Astro-Exo pipeline.
"""

import argparse
import sys
from astro_exo.pipeline.config import TargetConfig, PipelineConfig
from astro_exo.pipeline.runner import ExoplanetPipelineRunner


def main():
    parser = argparse.ArgumentParser(
        prog="astro-exo",
        description="Astro-Exo: Exoplanet Discovery, Spatial Vetting & Bayesian Inference Pipeline."
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Pipeline subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run full pipeline on a target TIC ID")
    run_parser.add_argument("--tic", type=int, required=True, help="TESS Input Catalog (TIC) ID")
    run_parser.add_argument("--sector", type=int, default=None, help="TESS observing sector")
    run_parser.add_argument("--period", type=float, default=None, help="Orbital period in days")
    run_parser.add_argument("--t0", type=float, default=None, help="Time of mid-transit in BJD/BTJD")
    run_parser.add_argument("--duration", type=float, default=None, help="Transit duration in hours")
    run_parser.add_argument("--backend", choices=["emcee", "jax_nuts"], default="emcee", help="Sampling backend")
    run_parser.add_argument("--outdir", type=str, default="results", help="Output directory")

    # Command: vet-only
    vet_parser = subparsers.add_parser("vet", help="Run spatial difference imaging and Gaia vetting only")
    vet_parser.add_argument("--tic", type=int, required=True, help="TIC ID")
    vet_parser.add_argument("--sector", type=int, default=None, help="TESS sector")
    vet_parser.add_argument("--period", type=float, required=True, help="Orbital period in days")
    vet_parser.add_argument("--t0", type=float, required=True, help="Epoch of mid-transit")
    vet_parser.add_argument("--duration", type=float, default=3.0, help="Transit duration in hours")

    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

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
            output_dir=args.outdir
        )
        runner = ExoplanetPipelineRunner(target, config)
        runner.run()

    elif args.subcommand == "vet":
        target = TargetConfig(
            tic_id=args.tic,
            sector=args.sector,
            period_days=args.period,
            t0_bjd=args.t0,
            duration_hours=args.duration
        )
        config = PipelineConfig(
            run_vetting=True,
            sampler_backend="none"
        )
        print(f"[VETTING] Running spatial vetting on TIC {args.tic}...")
        # Spatial vetting invocation
        from astro_exo.ingestion.mast_tess import fetch_tess_tpf
        from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset

        tpf = fetch_tess_tpf(args.tic, sector=args.sector)
        i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
            tpf.time.value, tpf.flux.value, tpf.flux_err.value,
            args.period, args.t0, args.duration / 24.0
        )
        pix_x, pix_y = tpf.wcs.all_world2pix(tpf.ra, tpf.dec, 0)
        res = measure_centroid_offset(i_diff, sigma_diff, (pix_x, pix_y))
        print("Centroid Offset Results:")
        for k, v in res.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
