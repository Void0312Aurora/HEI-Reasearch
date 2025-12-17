#!/usr/bin/env python3
"""
群积分器 vs 传统积分器 对比测试
===============================

运行群积分器和传统积分器的模拟，生成可视化对比结果。

用法：
    # 快速测试（500步）
    python run_comparison.py
    
    # 长周期测试（4000步）
    python run_comparison.py --steps 4000
    
    # 只运行群积分器
    python run_comparison.py --group-only
    
    # 只运行传统积分器
    python run_comparison.py --legacy-only
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from hei.simulation import run_simulation, run_simulation_group, SimulationConfig
from hei.potential import build_hierarchical_potential
from hei.plot_results import plot_log
from hei.metrics import cluster_summary_kmeans


def plot_comparison(log_group, log_legacy, out_path: Path):
    """绘制两种积分器的对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    steps = len(log_group.energy)
    x = np.arange(steps)
    
    # 能量对比
    axes[0, 0].plot(x, log_group.energy, label="Group", color="tab:blue")
    axes[0, 0].plot(x, log_legacy.energy, label="Legacy", color="tab:orange", alpha=0.7)
    axes[0, 0].set_title("能量对比 (per point)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ||ξ|| 对比
    axes[0, 1].semilogy(x, log_group.xi_norm, label="Group", color="tab:blue")
    axes[0, 1].semilogy(x, log_legacy.xi_norm, label="Legacy", color="tab:orange", alpha=0.7)
    axes[0, 1].set_title("||ξ|| 对比 (log scale)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("||ξ||")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 梯度对比
    axes[0, 2].plot(x, log_group.grad_norm, label="Group", color="tab:blue")
    axes[0, 2].plot(x, log_legacy.grad_norm, label="Legacy", color="tab:orange", alpha=0.7)
    axes[0, 2].set_title("梯度范数对比")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("||∇V||")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Contact action 对比
    axes[1, 0].plot(x, log_group.z_series, label="Group", color="tab:blue")
    axes[1, 0].plot(x, log_legacy.z_series, label="Legacy", color="tab:orange", alpha=0.7)
    axes[1, 0].set_title("Contact Action Z")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Z")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 时间步长对比
    axes[1, 1].plot(x, log_group.dt_series, label="Group", color="tab:blue")
    axes[1, 1].plot(x, log_legacy.dt_series, label="Legacy", color="tab:orange", alpha=0.7)
    axes[1, 1].set_title("时间步长 dt")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("dt")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 阻尼系数对比
    axes[1, 2].plot(x, log_group.gamma_series, label="Group", color="tab:blue")
    axes[1, 2].plot(x, log_legacy.gamma_series, label="Legacy", color="tab:orange", alpha=0.7)
    axes[1, 2].set_title("阻尼系数 γ")
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].set_ylabel("γ")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"保存对比图到: {out_path}")


def print_summary(name: str, log, steps: int):
    """打印模拟结果摘要"""
    print(f"\n{'='*60}")
    print(f"{name} 结果摘要 ({steps} 步)")
    print(f"{'='*60}")
    
    e0, ef = log.energy[0], log.energy[-1]
    print(f"能量: {e0:.4f} → {ef:.4f} ({(ef/e0-1)*100:+.2f}%)")
    
    xi0, xif = log.xi_norm[0], log.xi_norm[-1]
    print(f"||ξ||: {xi0:.6f} → {xif:.6f}")
    
    g0, gf = log.grad_norm[0], log.grad_norm[-1]
    print(f"梯度: {g0:.2f} → {gf:.2f}")
    
    z0, zf = log.z_series[0], log.z_series[-1]
    print(f"Action Z: {z0:.2f} → {zf:.2f}")
    
    # 数值稳定性检查
    has_nan = any(np.isnan(e) or np.isinf(e) for e in log.energy)
    has_explosion = any(xi > 100 for xi in log.xi_norm)
    
    status = "✓ 稳定"
    if has_nan:
        status = "❌ 有 NaN/Inf"
    elif has_explosion:
        status = "⚠ 有爆炸"
    
    print(f"数值状态: {status}")
    
    # 惯性张量诊断
    if hasattr(log, 'inertia_eig_min'):
        I_min_final = log.inertia_eig_min[-1]
        I_max_final = log.inertia_eig_max[-1]
        print(f"惯性特征值: λ ∈ [{I_min_final:.2e}, {I_max_final:.2e}]")
    
    # 阻尼诊断
    if hasattr(log, 'gamma_series'):
        gamma_mean = np.mean(log.gamma_series)
        gamma_max = np.max(log.gamma_series)
        print(f"阻尼系数: γ_mean={gamma_mean:.4f}, γ_max={gamma_max:.4f}")
    
    return not (has_nan or has_explosion)


def main():
    parser = argparse.ArgumentParser(description="群积分器 vs 传统积分器对比测试")
    parser.add_argument("--steps", type=int, default=500, help="积分步数")
    parser.add_argument("--n-points", type=int, default=50, help="粒子数")
    parser.add_argument("--depth", type=int, default=3, help="层级深度")
    parser.add_argument("--branching", type=int, default=3, help="分支因子")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--group-only", action="store_true", help="只运行群积分器")
    parser.add_argument("--legacy-only", action="store_true", help="只运行传统积分器")
    
    args = parser.parse_args()
    
    print("="*60)
    print("群积分器 + Hyperboloid 混合架构测试")
    print("="*60)
    print(f"配置: {args.n_points} 个点, {args.steps} 步")
    print(f"层级: depth={args.depth}, branching={args.branching}")
    print()
    
    # 创建势能
    rng = np.random.default_rng(args.seed)
    pot = build_hierarchical_potential(
        n_points=args.n_points,
        depth=args.depth,
        branching=args.branching,
        rng=rng
    )
    
    # 配置
    cfg = SimulationConfig(
        n_points=args.n_points,
        steps=args.steps,
        max_rho=3.0,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_group = None
    log_legacy = None
    
    # 运行群积分器
    if not args.legacy_only:
        print("运行群积分器版本...")
        log_group = run_simulation_group(
            potential=pot,
            config=cfg,
            rng=np.random.default_rng(args.seed)
        )
        success_group = print_summary("群积分器", log_group, args.steps)
        
        # 生成可视化
        plot_log(
            energy=log_group.energy,
            xi_norm=log_group.xi_norm,
            potential=log_group.potential,
            kinetic=log_group.kinetic,
            z_series=log_group.z_series,
            grad_norm=log_group.grad_norm,
            positions_disk=log_group.positions_disk,
            centroid_disk=log_group.centroid_disk,
            p_proxy=log_group.p_proxy,
            q_proj=log_group.q_proj,
            p_vec=log_group.p_vec,
            out_path=output_dir / "group_integrator.png",
            track_points=5,
            dt_series=log_group.dt_series,
            gamma_series=log_group.gamma_series,
            residual_contact=log_group.residual_contact,
            residual_momentum=log_group.residual_momentum,
            gap_median=log_group.gap_median,
            gap_frac_small=log_group.gap_frac_small,
            V_ent=log_group.V_ent,
            bridge_ratio=log_group.bridge_ratio,
            ratio_break=log_group.ratio_break,
            beta_series=log_group.beta_series,
            rigid_speed2=log_group.rigid_speed2,
            relax_speed2=log_group.relax_speed2,
            total_speed2=log_group.total_speed2,
        )
    
    # 运行传统积分器
    if not args.group_only:
        print("\n运行传统积分器版本...")
        log_legacy = run_simulation(
            potential=pot,
            config=cfg,
            rng=np.random.default_rng(args.seed)
        )
        success_legacy = print_summary("传统积分器", log_legacy, args.steps)
        
        # 生成可视化
        plot_log(
            energy=log_legacy.energy,
            xi_norm=log_legacy.xi_norm,
            potential=log_legacy.potential,
            kinetic=log_legacy.kinetic,
            z_series=log_legacy.z_series,
            grad_norm=log_legacy.grad_norm,
            positions_disk=log_legacy.positions_disk,
            centroid_disk=log_legacy.centroid_disk,
            p_proxy=log_legacy.p_proxy,
            q_proj=log_legacy.q_proj,
            p_vec=log_legacy.p_vec,
            out_path=output_dir / "legacy_integrator.png",
            track_points=5,
            dt_series=log_legacy.dt_series,
            gamma_series=log_legacy.gamma_series,
            residual_contact=log_legacy.residual_contact,
            residual_momentum=log_legacy.residual_momentum,
            gap_median=log_legacy.gap_median,
            gap_frac_small=log_legacy.gap_frac_small,
            V_ent=log_legacy.V_ent,
            bridge_ratio=log_legacy.bridge_ratio,
            ratio_break=log_legacy.ratio_break,
            beta_series=log_legacy.beta_series,
            rigid_speed2=log_legacy.rigid_speed2,
            relax_speed2=log_legacy.relax_speed2,
            total_speed2=log_legacy.total_speed2,
        )
    
    # 对比图
    if log_group and log_legacy:
        plot_comparison(log_group, log_legacy, output_dir / "comparison.png")
        
        print(f"\n{'='*60}")
        print("对比总结")
        print(f"{'='*60}")
        
        # 聚类质量对比
        if log_group.positions_disk and log_legacy.positions_disk:
            cluster_group = cluster_summary_kmeans(log_group.positions_disk[-1])
            cluster_legacy = cluster_summary_kmeans(log_legacy.positions_disk[-1])
            
            print(f"\n聚类质量:")
            print(f"  群积分器: k={cluster_group.get('best_k')}, "
                  f"sil={cluster_group.get('best_silhouette', 0):.3f}")
            print(f"  传统积分器: k={cluster_legacy.get('best_k')}, "
                  f"sil={cluster_legacy.get('best_silhouette', 0):.3f}")
    
    print(f"\n所有结果已保存到: {output_dir}/")
    print(f"  - group_integrator.png (群积分器)")
    print(f"  - legacy_integrator.png (传统积分器)")
    if log_group and log_legacy:
        print(f"  - comparison.png (对比图)")


if __name__ == "__main__":
    main()

