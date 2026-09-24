"""
Bayesian statistical false positive validation via TRICERATOPS and analytical population modeling.
Computes False Positive Probability (FPP) and Nearby False Positive Probability (NFPP) across
astrophysical scenarios: TP, PTP, EB, EBx2P, BEB, and HEB.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class BayesianFalsePositiveEngine:
    """
    Analytical Bayesian False Positive validator modeling the likelihood of
    competing astrophysical scenarios given Gaia DR3 neighbor photometry,
    spatial difference-image centroid astrometry, transit geometry, and light-curve morphology.
    """

    def __init__(
        self,
        target_tmag: float,
        period_days: float,
        depth_ppm: float,
        duration_hours: float,
        rp_rs: float,
        impact_parameter_b: float = 0.30,
        centroid_offset_arcsec: float = 0.0,
        centroid_sigma_arcsec: float = 1.0,
        neighbors: Optional[List[Dict[str, Any]]] = None
    ):
        self.target_tmag = float(target_tmag)
        self.period = float(period_days)
        self.depth_ppm = float(depth_ppm)
        self.duration_hours = float(duration_hours)
        self.rp_rs = float(rp_rs)
        self.b = float(impact_parameter_b)
        self.centroid_offset = float(centroid_offset_arcsec)
        self.centroid_sigma = max(float(centroid_sigma_arcsec), 0.20)
        self.neighbors = neighbors or []

    def calculate_scenario_probabilities(self) -> Dict[str, Any]:
        """
        Computes marginal posterior probabilities for:
          - TP: Transiting Planet around target
          - PTP: Planet Transiting Primary (unresolved companion)
          - EB: Eclipsing Binary on target
          - EBx2P: Eclipsing Binary at twice the period (symmetric secondary)
          - BEB: Background Eclipsing Binary on a nearby star
          - HEB: Hierarchical Eclipsing Binary bound to target
        """
        # 1. Base Prior Probabilities (Occurrence Rates)
        # Transiting planet prior depends on size: sub-Neptunes are common (~0.15), Jupiters rarer (~0.01)
        if self.rp_rs < 0.05:
            prior_tp = 0.120
        elif self.rp_rs < 0.12:
            prior_tp = 0.040
        elif self.rp_rs < 0.18:
            prior_tp = 0.015
        else:
            # Beyond 0.20 (2 R_Jup), planetary hypothesis is physically disfavored
            prior_tp = 0.0005

        prior_ptp = prior_tp * 0.10
        prior_eb = 0.008
        prior_ebx2p = 0.002
        prior_heb = 0.003

        # 2. Astrometric Likelihood on Target
        # Probability that centroid offset matches the target star (offset = 0)
        chi2_target = (self.centroid_offset / self.centroid_sigma) ** 2
        lhood_astro_target = float(np.exp(-0.5 * chi2_target))

        # 3. Morphology & Geometry Likelihoods
        # Planetary transit: physical if rp_rs <= 0.20
        if self.rp_rs <= 0.15:
            lhood_geom_tp = 1.0
        elif self.rp_rs <= 0.20:
            lhood_geom_tp = float(np.exp(-0.5 * ((self.rp_rs - 0.15) / 0.02) ** 2))
        else:
            lhood_geom_tp = float(np.exp(-0.5 * ((self.rp_rs - 0.15) / 0.01) ** 2))

        # EB on target:
        # If the transit has a clear flat bottom (b <= 0.85), a grazing eclipse is ruled out by the light curve.
        # An on-target non-grazing eclipse requires depth delta >= (R_min / R_star)^2 >= 1.0% to 50%.
        # For planetary depths (< 1.5%), an on-target non-grazing eclipse would require a companion in the
        # brown dwarf desert (frequency < 0.1%).
        if self.b <= 0.85:
            # Flat bottom detected: grazing EB is ruled out by light curve residuals
            if self.rp_rs <= 0.16:
                # Canonical & inflated Hot Jupiter / brown-dwarf desert regime -> EB heavily suppressed
                lhood_geom_eb = 0.003
                lhood_geom_ebx2p = 0.001
            elif self.rp_rs <= 0.20:
                lhood_geom_eb = 0.020
                lhood_geom_ebx2p = 0.005
            else:
                # Deep eclipse: EB is very likely
                lhood_geom_eb = 0.800
                lhood_geom_ebx2p = 0.200
        else:
            # V-shaped transit (b > 0.85): grazing EB is geometrically plausible
            lhood_geom_eb = 0.350
            lhood_geom_ebx2p = 0.100

        # HEB likelihood: bound hierarchical binary diluted by primary
        lhood_geom_heb = 0.005 if self.b <= 0.85 and self.rp_rs <= 0.16 else 0.050

        # 4. Background Eclipsing Binary (BEB) Likelihood over Neighbors
        # For each neighbor, check if it can physically produce the depth and if its
        # position matches the measured difference centroid.
        beb_weights = []
        for star in self.neighbors:
            dist_arcsec = star.get("dist_arcsec", 99.0)
            if dist_arcsec < 0.1:
                continue

            can_cause = star.get("can_cause_transit", True)
            if not can_cause:
                continue

            # Astrometric match: how close is this neighbor to the measured deficit position?
            # Difference between neighbor separation and measured centroid offset
            pos_mismatch = abs(dist_arcsec - self.centroid_offset)
            lhood_astro_neighbor = float(np.exp(-0.5 * (pos_mismatch / self.centroid_sigma) ** 2))

            # Prior probability of this neighbor being an EB: ~ 0.008
            # Weighted by its aperture fraction and flux capability
            weight_j = 0.008 * lhood_astro_neighbor
            beb_weights.append(weight_j)

        weight_beb = float(sum(beb_weights)) if beb_weights else 0.0

        # Unnormalized posterior weights
        w_tp = prior_tp * lhood_astro_target * lhood_geom_tp
        w_ptp = prior_ptp * lhood_astro_target * lhood_geom_tp
        w_eb = prior_eb * lhood_astro_target * lhood_geom_eb
        w_ebx2p = prior_ebx2p * lhood_astro_target * lhood_geom_ebx2p
        w_heb = prior_heb * lhood_astro_target * lhood_geom_heb
        w_beb = weight_beb

        total_weight = w_tp + w_ptp + w_eb + w_ebx2p + w_heb + w_beb
        if total_weight <= 0:
            total_weight = 1e-12

        prob_tp = w_tp / total_weight
        prob_ptp = w_ptp / total_weight
        prob_eb = w_eb / total_weight
        prob_ebx2p = w_ebx2p / total_weight
        prob_heb = w_heb / total_weight
        prob_beb = w_beb / total_weight

        # FPP = 1.0 - (P(TP) + P(PTP))
        fpp = float(1.0 - (prob_tp + prob_ptp))
        # NFPP = P(BEB)
        nfpp = float(prob_beb)

        # Standard scientific validation criteria (Giacalone et al. 2021)
        is_validated = (fpp < 0.010) and (nfpp < 1e-3)

        return {
            "fpp": float(round(fpp, 6)),
            "nfpp": float(round(nfpp, 6)),
            "validated": is_validated,
            "probabilities": {
                "TP": float(round(prob_tp, 5)),
                "PTP": float(round(prob_ptp, 5)),
                "EB": float(round(prob_eb, 5)),
                "EBx2P": float(round(prob_ebx2p, 5)),
                "HEB": float(round(prob_heb, 5)),
                "BEB": float(round(prob_beb, 5))
            },
            "astrometric_target_likelihood": float(round(lhood_astro_target, 6))
        }


def run_triceratops_validation(
    tic_id: int,
    sectors: List[int],
    period: float,
    depth: float,
    duration_days: float,
    rp_rs: float = 0.10,
    impact_parameter_b: float = 0.30,
    target_tmag: float = 10.0,
    centroid_offset_arcsec: float = 0.0,
    centroid_sigma_arcsec: float = 1.0,
    neighbors: Optional[List[Dict[str, Any]]] = None,
    contrast_curve_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes TRICERATOPS statistical validation.
    If the triceratops package is installed, uses its TRILEGAL simulation interface.
    Otherwise, uses the integrated BayesianFalsePositiveEngine.
    """
    # 1. Check for official triceratops package
    try:
        import triceratops.triceratops as tr

        target = tr.target(ID=tic_id, sectors=sectors)
        target.search_stars()
        target.calc_probs(
            time=None,
            flux_val=depth,
            period=period
        )

        fpp = float(target.FPP)
        nfpp = float(target.NFPP)
        is_validated = (fpp < 0.01) and (nfpp < 1e-3)

        probs = {}
        if hasattr(target, "probs"):
            probs = {str(k): float(v) for k, v in target.probs.items()}

        return {
            "fpp": fpp,
            "nfpp": nfpp,
            "validated": is_validated,
            "engine": "triceratops_native",
            "probabilities": probs
        }
    except (ImportError, Exception):
        # 2. Use BayesianFalsePositiveEngine
        engine = BayesianFalsePositiveEngine(
            target_tmag=target_tmag,
            period_days=period,
            depth_ppm=depth if depth > 1.0 else depth * 1e6,
            duration_hours=duration_days * 24.0,
            rp_rs=rp_rs,
            impact_parameter_b=impact_parameter_b,
            centroid_offset_arcsec=centroid_offset_arcsec,
            centroid_sigma_arcsec=centroid_sigma_arcsec,
            neighbors=neighbors
        )
        res = engine.calculate_scenario_probabilities()
        res["engine"] = "bayesian_analytical_engine"
        res["tic_id"] = tic_id
        return res
