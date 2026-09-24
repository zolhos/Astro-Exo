"""
End-to-end pipeline runner orchestrating Ingestion -> Vetting -> Sampling -> Report.
"""

import os
from typing import Optional
import numpy as np
from astro_exo.pipeline.config import TargetConfig, PipelineConfig
from astro_exo.pipeline.schemas import VettingReport, TransitInferenceResult, FullCandidateProduct
from astro_exo.ingestion.mast_tess import fetch_tess_lightcurve, fetch_tess_tpf
from astro_exo.ingestion.detrending import flatten_lightcurve, iterative_flatten
from astro_exo.ingestion.search import verify_against_exoplanet_archive
from astro_exo.vetting.difference_img import calculate_difference_image, measure_centroid_offset
from astro_exo.vetting.gaia import query_gaia_neighbors, overlay_gaia_on_wcs
from astro_exo.vetting.dilution import rule_out_neighbors_as_blends
from astro_exo.vetting.triceratops_vet import run_triceratops_validation
from astro_exo.models.emcee_sampler import EmceeTransitFitter
from astro_exo.models.transforms import compute_stellar_density, impact_param_to_inclination


class ExoplanetPipelineRunner:
    """Executes the full exoplanet discovery and characterization workflow."""

    def __init__(self, target_cfg: TargetConfig, pipeline_cfg: Optional[PipelineConfig] = None):
        self.target = target_cfg
        self.config = pipeline_cfg or PipelineConfig()

    def run(self) -> FullCandidateProduct:
        print(f"[PIPELINE] Starting analysis for TIC {self.target.tic_id} (Sector {self.target.sector})...")

        # 1. Ingestion
        lc = fetch_tess_lightcurve(self.target.tic_id, sector=self.target.sector, author=self.target.author)
        t_raw = lc.time.value
        f_raw = lc.flux.value
        e_raw = lc.flux_err.value

        # 2. Ephemeris defaults
        period = self.target.period_days or 3.5
        t0 = self.target.t0_bjd or float(np.nanmin(t_raw) + 0.5)
        dur_hours = self.target.duration_hours or 3.0
        dur_days = dur_hours / 24.0

        # Iterative detrending with transit mask
        flux_flat, trend, _ = iterative_flatten(t_raw, f_raw, period, t0, dur_days)

        # Cross-match with NASA Exoplanet Archive
        is_new, match_info = verify_against_exoplanet_archive(self.target.tic_id, period, t0)
        print(f"[ARCHIVE] {match_info}")

        # 3. Spatial Vetting on TPF
        passed_vetting = False
        vetting_report = VettingReport(
            centroid_offset_arcsec=0.0,
            centroid_significance_sigma=0.0,
            target_pixel_x=0.0,
            target_pixel_y=0.0,
            diff_centroid_x=0.0,
            diff_centroid_y=0.0,
            gaia_neighbors_count=0,
            neighbors_ruling_out_count=0,
            passed_spatial_vetting=False
        )

        if self.config.run_vetting:
            try:
                tpf = fetch_tess_tpf(self.target.tic_id, sector=self.target.sector, author=self.target.author)
                t_tpf = tpf.time.value
                f_tpf = tpf.flux.value
                e_tpf = tpf.flux_err.value

                # Difference imaging
                i_out, i_in, i_diff, sigma_diff = calculate_difference_image(
                    t_tpf, f_tpf, e_tpf, period, t0, dur_days
                )

                # Target WCS pixel coordinate
                ra, dec = tpf.ra, tpf.dec
                pix_x_tgt, pix_y_tgt = tpf.wcs.all_world2pix(ra, dec, 0)

                # Centroid offset
                cen_res = measure_centroid_offset(i_diff, sigma_diff, (pix_x_tgt, pix_y_tgt))
                passed_vetting = cen_res["offset_significance_sigma"] < 3.0

                # Gaia DR3 query
                neighbors = query_gaia_neighbors(ra, dec, radius_arcmin=1.0)
                vetted_neighbors = rule_out_neighbors_as_blends(
                    observed_transit_depth_ppm=1000.0,  # approximate depth
                    target_mag=10.0,
                    neighbors=neighbors
                )
                ruled_out_count = sum(1 for n in vetted_neighbors if n.get("ruled_out", False))

                vetting_report = VettingReport(
                    centroid_offset_arcsec=cen_res["offset_arcsec"],
                    centroid_significance_sigma=cen_res["offset_significance_sigma"],
                    target_pixel_x=float(pix_x_tgt),
                    target_pixel_y=float(pix_y_tgt),
                    diff_centroid_x=cen_res["x_diff_cen"],
                    diff_centroid_y=cen_res["y_diff_cen"],
                    gaia_neighbors_count=len(neighbors),
                    neighbors_ruling_out_count=ruled_out_count,
                    passed_spatial_vetting=passed_vetting
                )
            except Exception as e:
                print(f"[WARN] Spatial vetting encountered issue: {e}. Marking passed_vetting=False.")
                passed_vetting = False
                vetting_report.passed_spatial_vetting = False

        # 4. Bayesian Transit Modeling
        print(f"[MCMC] Running transit inference via backend={self.config.sampler_backend}...")
        # Isolate transit window
        transit_mask = np.abs((t_raw - t0 + 0.5 * period) % period - 0.5 * period) < (2.0 * dur_days)
        t_fit = t_raw[transit_mask]
        f_fit = flux_flat[transit_mask]
        e_fit = e_raw[transit_mask]

        summary = None
        if self.config.sampler_backend == "jax_nuts":
            try:
                from astro_exo.models.jax_nuts import JaxNutsTransitFitter
                print("[JAX/NUTS] Initializing Hamiltonian Monte Carlo sampler...")
                jax_fitter = JaxNutsTransitFitter(
                    time=t_fit,
                    flux=f_fit,
                    flux_err=e_fit,
                    period=period,
                    t0_prior_mean=t0
                )
                idata = jax_fitter.run_nuts(
                    num_warmup=self.config.nuts_warmup,
                    num_samples=self.config.nuts_samples
                )
                post = idata.posterior
                rp_med = float(np.median(post["rp"].values))
                rp_err = float(np.std(post["rp"].values))
                a_rs_med = float(np.median(post["a_rs"].values))
                a_rs_err = float(np.std(post["a_rs"].values))
                b_med = float(np.median(post["b"].values))
                b_err = float(np.std(post["b"].values))
                t0_med = float(np.median(post["t0"].values))
                t0_err = float(np.std(post["t0"].values))
                q1_med = float(np.median(post["q1"].values))
                q2_med = float(np.median(post["q2"].values))
            except Exception as jax_err:
                print(f"[WARN] JAX/NUTS fallback to emcee: {jax_err}")
                summary = "fallback"

        rhat_max = None
        if self.config.sampler_backend != "jax_nuts" or summary == "fallback":
            fitter = EmceeTransitFitter(
                time=t_fit,
                flux=f_fit,
                flux_err=e_fit,
                period=period,
                t0_expected=t0
            )
            fitter.run_mcmc(
                nwalkers=self.config.mcmc_walkers,
                nburn=self.config.mcmc_burnin,
                nprod=self.config.mcmc_production
            )
            summary_mcmc = fitter.get_summary()

            rp_med = summary_mcmc["rp"]["median"]
            rp_err = 0.5 * (summary_mcmc["rp"]["err_plus_1s"] + summary_mcmc["rp"]["err_minus_1s"])
            a_rs_med = summary_mcmc["a_rs"]["median"]
            a_rs_err = 0.5 * (summary_mcmc["a_rs"]["err_plus_1s"] + summary_mcmc["a_rs"]["err_minus_1s"])
            b_med = summary_mcmc["b"]["median"]
            b_err = 0.5 * (summary_mcmc["b"]["err_plus_1s"] + summary_mcmc["b"]["err_minus_1s"])
            t0_med = summary_mcmc["t0"]["median"]
            t0_err = 0.5 * (summary_mcmc["t0"]["err_plus_1s"] + summary_mcmc["t0"]["err_minus_1s"])
            q1_med = summary_mcmc["q1"]["median"]
            q2_med = summary_mcmc["q2"]["median"]

            diag = fitter.get_diagnostics()
            rhat_vals = [v for v in diag["r_hat"].values() if np.isfinite(v)]
            rhat_max = float(max(rhat_vals)) if rhat_vals else None

            # Generate corner & fit plot in target folder
            target_plot_dir = os.path.join(self.config.output_dir, f"TIC_{self.target.tic_id}")
            os.makedirs(target_plot_dir, exist_ok=True)
            try:
                fitter.plot_corner(
                    os.path.join(target_plot_dir, "mcmc_corner.png"),
                    title=f"TIC {self.target.tic_id} - MCMC Posteriors"
                )
                fitter.plot_fit(
                    os.path.join(target_plot_dir, "transit_fit.png"),
                    title=f"TIC {self.target.tic_id} (P = {period:.4f} d) - Transit Fit"
                )
            except Exception as plot_err:
                print(f"[WARN] Could not generate MCMC plots: {plot_err}")

        inc_med = impact_param_to_inclination(b_med, a_rs_med)
        rho_star = compute_stellar_density(period, a_rs_med)

        inference_result = TransitInferenceResult(
            t0_bjd=t0_med,
            t0_err=t0_err,
            rp_rs=rp_med,
            rp_rs_err=rp_err,
            a_rs=a_rs_med,
            a_rs_err=a_rs_err,
            impact_parameter_b=b_med,
            impact_parameter_err=b_err,
            inclination_deg=inc_med,
            inclination_err=1.0,
            limb_dark_q1=q1_med,
            limb_dark_q2=q2_med,
            stellar_density_g_cm3=rho_star,
            gelman_rubin_rhat_max=rhat_max
        )

        disposition = "CANDIDATE"
        if not passed_vetting:
            disposition = "FALSE_POSITIVE"
        elif is_new and passed_vetting:
            disposition = "VALIDATED_PLANET"

        product = FullCandidateProduct(
            tic_id=self.target.tic_id,
            sector=self.target.sector,
            period_days=period,
            is_uncataloged=is_new,
            vetting=vetting_report,
            inference=inference_result,
            disposition=disposition
        )

        # Ensure output directory exists and write JSON
        os.makedirs(self.config.output_dir, exist_ok=True)
        out_path = os.path.join(self.config.output_dir, f"TIC_{self.target.tic_id}_product.json")
        with open(out_path, "w") as f:
            f.write(product.to_json())

        print(f"[PIPELINE] Complete. Disposition: {disposition}. Output saved to {out_path}")
        return product
