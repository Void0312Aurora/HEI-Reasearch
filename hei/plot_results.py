"""Visualization helpers for CCD hyperbolic simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimulationConfig, run_simulation
from .metrics import (
    compute_hierarchical_metrics,
    compute_label_timeseries,
    silhouette_timeseries,
    cluster_summary_kmeans,
)
from .potential import HierarchicalSoftminPotential


def plot_log(
    energy: list[float],
    xi_norm: list[float],
    potential: list[float],
    kinetic: list[float],
    z_series: list[float],
    grad_norm: list[float],
    positions_disk: list[np.ndarray],
    centroid_disk: list[complex] | None = None,
    p_proxy: list[float] | None = None,
    q_proj: list[tuple[float, float]] | None = None,
    p_vec: list[np.ndarray] | None = None,
    out_path: Path = Path("outputs/simulation.png"),
    track_points: int = 5,
    hier_series: dict | None = None,
    sil_series: np.ndarray | None = None,
    dynamics: dict | None = None,
    dt_series: list[float] | None = None,
    gamma_series: list[float] | None = None,
    residual_contact: list[float] | None = None,
    residual_momentum: list[float] | None = None,
    cluster_summary: dict | None = None,
    gap_median: list[float] | None = None,
    gap_frac_small: list[float] | None = None,
    V_ent: list[float] | None = None,
    bridge_ratio: list[float] | None = None,
    ratio_break: list[float] | None = None,
    beta_series: list[float] | None = None,
    rigid_speed2: list[float] | None = None,
    relax_speed2: list[float] | None = None,
    total_speed2: list[float] | None = None,
) -> None:
    """Plot energy/xi norms/grad and trajectories on the disk."""
    steps = len(energy)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].plot(energy, label="Energy")
    axes[0, 0].set_title("Total energy (per point)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xi_norm, color="tab:orange", label="||xi||")
    axes[0, 1].set_title("xi norm")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Norm")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(grad_norm, color="tab:green", label="||grad||")
    axes[0, 2].set_title("Gradient norm")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Norm")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(potential, label="Potential", color="tab:blue")
    axes[1, 0].plot(kinetic, label="Kinetic", color="tab:orange")
    axes[1, 0].set_title("Energy components (per point)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Energy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(z_series, label="Cumulative z", color="tab:red")
    axes[1, 1].set_title("Contact action Z")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Z")
    axes[1, 1].grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.set_title(f"Trajectories on disk (first {track_points})")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    if positions_disk:
        pos_arr = np.stack(positions_disk)  # (steps, N)
        n_pts = min(track_points, pos_arr.shape[1])
        cmap = plt.get_cmap("tab10", n_pts)
        for i in range(n_pts):
            traj = pos_arr[:, i]
            ax.plot(traj.real, traj.imag, color=cmap(i))
            ax.scatter(traj.real[-1], traj.imag[-1], color=cmap(i), s=10, alpha=0.8)
    ax.axis("off")

    if q_proj is not None and p_vec is not None:
        q_arr = np.array(q_proj)
        p_arr = np.stack(p_vec)
        fig_phase, axes_phase = plt.subplots(1, 2, figsize=(8, 4))
        axes_phase[0].set_title("Phase proj: q_Re vs p_u")
        axes_phase[0].plot(q_arr[:, 0], p_arr[:, 0], color="tab:purple")
        axes_phase[0].set_xlabel("q_Re (centroid)")
        axes_phase[0].set_ylabel("p_u")
        axes_phase[0].grid(True, alpha=0.3)

        axes_phase[1].set_title("Phase proj: q_Im vs p_v")
        axes_phase[1].plot(q_arr[:, 1], p_arr[:, 1], color="tab:olive")
        axes_phase[1].set_xlabel("q_Im (centroid)")
        axes_phase[1].set_ylabel("p_v")
        axes_phase[1].grid(True, alpha=0.3)

        phase_path = out_path.with_name(out_path.stem + "_phase.png")
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        fig_phase.tight_layout()
        fig_phase.savefig(phase_path, dpi=200, bbox_inches="tight")
        plt.close(fig_phase)
        print(f"Saved phase plot to {phase_path}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path} (steps={steps})")

    if dt_series or gamma_series or residual_contact or residual_momentum:
        fig_diag, axd = plt.subplots(1, 3, figsize=(14, 3))
        if dt_series and gamma_series:
            hk = np.array(dt_series) * np.array(gamma_series)
            axd[0].plot(dt_series, label="dt", color="tab:blue")
            axd[0].plot(gamma_series, label="gamma", color="tab:red")
            axd[0].plot(hk, label="h*gamma", color="tab:green")
            axd[0].set_title("Adaptive dt, gamma, h*gamma")
            axd[0].set_xlabel("Step")
            axd[0].grid(True, alpha=0.3)
            axd[0].legend()
        else:
            axd[0].axis("off")

        if residual_contact:
            axd[1].semilogy(np.abs(residual_contact), color="tab:purple")
            axd[1].set_title("|r_contact|")
            axd[1].set_xlabel("Step")
            axd[1].grid(True, alpha=0.3)
        else:
            axd[1].axis("off")

        if residual_momentum:
            axd[2].semilogy(residual_momentum, color="tab:orange")
            axd[2].set_title("||F|| (momentum residual)")
            axd[2].set_xlabel("Step")
            axd[2].grid(True, alpha=0.3)
        else:
            axd[2].axis("off")

        fig_diag.tight_layout()
        diag_path = out_path.with_name(out_path.stem + "_diag.png")
        fig_diag.savefig(diag_path, dpi=200, bbox_inches="tight")
        plt.close(fig_diag)
        print(f"Saved diagnostics plot to {diag_path}")

    if gap_median or bridge_ratio or V_ent or beta_series is not None:
        series_list = []
        titles = []
        colors = []
        labels_gap = []
        if gap_median:
            series_list.append((gap_median, gap_frac_small))
            titles.append("Gap stats")
            colors.append(("tab:blue", "tab:orange"))
            labels_gap.append(("gap_median", "frac_gap<eps"))
        if bridge_ratio:
            series_list.append((bridge_ratio, None))
            titles.append("Bridge ratio")
            colors.append(("tab:orange", None))
            labels_gap.append(("bridge k=3", None))
        if V_ent:
            series_list.append((V_ent, None))
            titles.append("Entropy energy")
            colors.append(("tab:purple", None))
            labels_gap.append(("V_ent", None))
        if beta_series is not None:
            series_list.append((beta_series, ratio_break if ratio_break else None))
            titles.append("Beta & break ratio")
            colors.append(("tab:gray", "tab:red"))
            labels_gap.append(("beta", "ratio_break"))
        ncols = len(series_list)
        fig_gap, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3))
        if ncols == 1:
            axes = [axes]
        for ax, series, title, color_pair, lbl_pair in zip(axes, series_list, titles, colors, labels_gap):
            s1, s2 = series
            c1, c2 = color_pair
            l1, l2 = lbl_pair
            ax.plot(s1, label=l1, color=c1)
            if s2 is not None:
                ax.plot(s2, label=l2, color=c2)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig_gap.tight_layout()
        gap_path = out_path.with_name(out_path.stem + "_gap.png")
        fig_gap.savefig(gap_path, dpi=200, bbox_inches="tight")
        plt.close(fig_gap)
        print(f"Saved gap/bridge plot to {gap_path}")

    if cluster_summary and ("cluster_counts_full" in cluster_summary or "cluster_counts" in cluster_summary):
        counts = cluster_summary.get("cluster_counts_full") or cluster_summary.get("cluster_counts")
        fig_clu, axc = plt.subplots(1, 1, figsize=(5, 3))
        axc.bar(range(len(counts)), counts, color="tab:cyan")
        axc.set_xlabel("Cluster id (sorted by size)")
        axc.set_ylabel("Count")
        title = f"k={cluster_summary.get('best_k')}, sil={cluster_summary.get('best_silhouette', 0):.2f}"
        axc.set_title(f"Hyperbolic k-means summary ({title})")
        fig_clu.tight_layout()
        clu_path = out_path.with_name(out_path.stem + "_clusters.png")
        fig_clu.savefig(clu_path, dpi=200, bbox_inches="tight")
        plt.close(fig_clu)
        print(f"Saved cluster summary plot to {clu_path}")

    if sil_series is not None and dynamics is not None:
        fig_dyn, axd = plt.subplots(1, 3, figsize=(14, 3))
        for ax, series, title, ylabel in [
            (axd[0], dynamics["xi_norm"], "Silhouette vs |xi|", "|xi|"),
            (axd[1], dynamics["grad_norm"], "Silhouette vs grad_norm", "||grad||"),
            (axd[2], dynamics["energy"], "Silhouette vs energy", "Energy/pt"),
        ]:
            ax.plot(series, label="dyn", color="tab:orange")
            ax_t = ax.twinx()
            ax_t.plot(sil_series, label="silhouette", color="tab:blue", alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel, color="tab:orange")
            ax_t.set_ylabel("Silhouette", color="tab:blue")
            ax.grid(True, alpha=0.3)
        fig_dyn.tight_layout()
        dyn_path = out_path.with_name(out_path.stem + "_sil_overlay.png")
        fig_dyn.savefig(dyn_path, dpi=200, bbox_inches="tight")
        plt.close(fig_dyn)
        print(f"Saved silhouette overlay plot to {dyn_path}")

    if rigid_speed2 is not None and relax_speed2 is not None:
        fig_speed, ax_speed = plt.subplots(1, 1, figsize=(6, 3))
        ax_speed.plot(rigid_speed2, label="rigid |v|_g^2", color="tab:blue")
        ax_speed.plot(relax_speed2, label="relax |v|_g^2", color="tab:orange")
        if total_speed2 is not None:
            ax_speed.plot(total_speed2, label="total |v|_g^2", color="tab:green")
        ax_speed.set_title("Hyperbolic mean squared speeds")
        ax_speed.set_xlabel("Step")
        ax_speed.set_ylabel("Mean |v|_g^2")
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend()
        fig_speed.tight_layout()
        speed_path = out_path.with_name(out_path.stem + "_speed.png")
        fig_speed.savefig(speed_path, dpi=200, bbox_inches="tight")
        plt.close(fig_speed)
        print(f"Saved speed plot to {speed_path}")

    if hier_series is not None:
        fig_hier, axh = plt.subplots(1, 4, figsize=(16, 3))
        axh[0].plot(hier_series["mean_depth"])
        axh[0].set_title("E[d(t)]")
        axh[0].set_xlabel("Step")
        axh[0].set_ylabel("Depth")
        axh[0].grid(True, alpha=0.3)

        axh[1].plot(hier_series["deep_frac"])
        axh[1].set_title("Deep leaf fraction")
        axh[1].set_xlabel("Step")
        axh[1].set_ylabel("Frac")
        axh[1].grid(True, alpha=0.3)

        axh[2].plot(hier_series["entropy"])
        axh[2].set_title("Label entropy")
        axh[2].set_xlabel("Step")
        axh[2].set_ylabel("Entropy")
        axh[2].grid(True, alpha=0.3)

        if sil_series is not None:
            axh[3].plot(sil_series)
            axh[3].set_title("Silhouette(t)")
            axh[3].set_xlabel("Step")
            axh[3].set_ylabel("Silhouette")
            axh[3].grid(True, alpha=0.3)
        else:
            axh[3].axis("off")

        fig_hier.tight_layout()
        hier_path = out_path.with_name(out_path.stem + "_hier.png")
        fig_hier.savefig(hier_path, dpi=200, bbox_inches="tight")
        plt.close(fig_hier)
        print(f"Saved hierarchical series plot to {hier_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCD simulation and plot results.")
    parser.add_argument("--steps", type=int, default=400, help="Number of integration steps")
    parser.add_argument("--eps-dt", type=float, default=1e-2, dest="eps_dt", help="Adaptive step tolerance")
    parser.add_argument("--max-dt", type=float, default=5e-2, dest="max_dt", help="Max adaptive step")
    parser.add_argument("--min-dt", type=float, default=1e-4, dest="min_dt", help="Min adaptive step")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument(
        "--track-points", type=int, default=5, help="Number of point trajectories to draw"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/simulation.png"),
        help="Output image path",
    )
    parser.add_argument("--hier", action="store_true", help="Use hierarchical potential")
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional path to write clustering metrics JSON (hierarchical potential only)",
    )
    args = parser.parse_args()

    cfg = SimulationConfig(
        steps=args.steps,
        eps_dt=args.eps_dt,
        max_dt=args.max_dt,
        min_dt=args.min_dt,
    )
    potential = None
    rng_main = np.random.default_rng(args.seed)
    if args.hier:
        from .potential import build_hierarchical_potential

        potential = build_hierarchical_potential(
            n_points=cfg.n_points,
            max_rho=cfg.max_rho,
            rng=np.random.default_rng(args.seed),
        )

    log = run_simulation(potential=potential, config=cfg, rng=rng_main)

    cluster_summary = cluster_summary_kmeans(log.positions_disk[-1])
    print(
        f"Clusters: k={cluster_summary.get('best_k')}, "
        f"sil={cluster_summary.get('best_silhouette')}, "
        f"counts_full={cluster_summary.get('cluster_counts_full')}"
    )

    hier_series = None
    hier_metrics = None
    if isinstance(potential, HierarchicalSoftminPotential):
        hier_series = compute_label_timeseries(
            positions_disk=log.positions_disk,
            anchors_disk=potential.centers_disk,
            depths=potential.depths,
            max_depth=potential.max_depth,
        )
        sil_series = silhouette_timeseries(
            positions_disk=log.positions_disk,
            anchors_disk=potential.centers_disk,
        )
        hier_metrics = compute_hierarchical_metrics(
            z_disk=log.positions_disk[-1],
            anchors_disk=potential.centers_disk,
            depths=potential.depths,
            parents=potential.parents,
        )
        print(f"tree_dist_corr={hier_metrics.get('tree_dist_corr')}")
    else:
        sil_series = None

    plot_log(
        energy=log.energy,
        xi_norm=log.xi_norm,
        potential=log.potential,
        kinetic=log.kinetic,
        z_series=log.z_series,
        grad_norm=log.grad_norm,
        positions_disk=log.positions_disk,
        centroid_disk=log.centroid_disk,
        p_proxy=log.p_proxy,
        q_proj=log.q_proj,
        p_vec=log.p_vec,
        out_path=args.out,
        track_points=args.track_points,
        hier_series=hier_series,
        sil_series=sil_series,
        dynamics={"xi_norm": log.xi_norm, "grad_norm": log.grad_norm, "energy": log.energy},
        dt_series=log.dt_series if hasattr(log, "dt_series") else None,
        gamma_series=log.gamma_series if hasattr(log, "gamma_series") else None,
        residual_contact=log.residual_contact if hasattr(log, "residual_contact") else None,
        residual_momentum=log.residual_momentum if hasattr(log, "residual_momentum") else None,
        cluster_summary=cluster_summary,
        gap_median=log.gap_median if hasattr(log, "gap_median") else None,
        gap_frac_small=log.gap_frac_small if hasattr(log, "gap_frac_small") else None,
        V_ent=log.V_ent if hasattr(log, "V_ent") else None,
        bridge_ratio=log.bridge_ratio if hasattr(log, "bridge_ratio") else None,
        ratio_break=log.ratio_break if hasattr(log, "ratio_break") else None,
        beta_series=log.beta_series if hasattr(log, "beta_series") else None,
        rigid_speed2=log.rigid_speed2 if hasattr(log, "rigid_speed2") else None,
        relax_speed2=log.relax_speed2 if hasattr(log, "relax_speed2") else None,
        total_speed2=log.total_speed2 if hasattr(log, "total_speed2") else None,
    )

    if args.metrics_out:
        # Base metrics always available
        metrics = {"cluster_summary": cluster_summary}
        if isinstance(potential, HierarchicalSoftminPotential):
            hier_metrics = hier_metrics or compute_hierarchical_metrics(
                z_disk=log.positions_disk[-1],
                anchors_disk=potential.centers_disk,
                depths=potential.depths,
                parents=potential.parents,
            )
            metrics.update(hier_metrics)
            # layer dominance diagnostics
            if hasattr(potential, "layer_soft_stats"):
                metrics["layer_soft_stats"] = potential.layer_soft_stats(log.positions_disk[-1])
            if "labels" in cluster_summary:
                from .metrics import cluster_anchor_profiles

                metrics["cluster_profiles"] = cluster_anchor_profiles(
                    labels=np.array(cluster_summary["labels"]),
                    points_disk=log.positions_disk[-1],
                    anchors_disk=potential.centers_disk,
                    depths=potential.depths,
                    parents=potential.parents,
                )
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2))
        print(f"Saved metrics to {args.metrics_out}: {metrics}")
        # Optional summary plot for metrics_hier
        try:
            fig_m, axm = plt.subplots(1, 1, figsize=(6, 3))
            vals = []
            labels_bar = []
            # hierarchical metrics if available
            for key in ["sil_anchor", "tree_dist_corr", "mean_tree_pair"]:
                if key in metrics and metrics[key] is not None:
                    labels_bar.append(key)
                    vals.append(metrics[key])
            cs = metrics.get("cluster_summary", {})
            for key in ["best_silhouette", "max_frac", "top3_frac"]:
                if key in cs and cs[key] is not None:
                    labels_bar.append(key)
                    vals.append(cs[key])
            if labels_bar:
                axm.bar(labels_bar, vals, color="tab:blue")
                axm.set_title(f"Metrics summary (k={cs.get('best_k')})")
                axm.set_ylim(bottom=0)
                axm.grid(True, axis="y", alpha=0.3)
                fig_m.tight_layout()
                metrics_png = args.metrics_out.with_name(args.metrics_out.stem + ".png")
                fig_m.savefig(metrics_png, dpi=200, bbox_inches="tight")
                plt.close(fig_m)
                print(f"Saved metrics plot to {metrics_png}")
            else:
                plt.close(fig_m)
        except Exception as e:  # pragma: no cover
            print(f"Failed to save metrics plot: {e}")


if __name__ == "__main__":
    main()
