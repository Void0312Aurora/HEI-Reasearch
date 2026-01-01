#!/usr/bin/env python3
"""
基准测试结果可视化

生成:
1. 计算曲线: tokens/s vs (B, L, evolution_steps)
2. 内存曲线: peak VRAM vs 配置
3. 能力曲线: PPL vs dim_q (欧氏 vs 双曲)
4. 消融分析: 各损失项贡献
5. 梯度流分析: 各模块梯度占比
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将只生成文本报告")


def load_results(result_dir: str) -> Dict[str, List]:
    """加载所有结果文件"""
    result_dir = Path(result_dir)
    results = {}
    
    for json_file in result_dir.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            results[json_file.stem] = json.load(f)
    
    return results


def generate_text_report(results: Dict, output_dir: Path):
    """生成文本报告"""
    report = []
    report.append("=" * 70)
    report.append("基准测试结果报告")
    report.append("=" * 70)
    report.append("")
    
    # Phase 1: 计算性能
    if 'phase1_compute' in results:
        report.append("## Phase 1: 计算性能")
        report.append("-" * 50)
        report.append(f"{'配置':<20} {'tok/s':>10} {'ms/step':>10} {'VRAM(GB)':>10}")
        report.append("-" * 50)
        
        for r in results['phase1_compute']:
            report.append(
                f"{r['config_name']:<20} "
                f"{r['tokens_per_sec_mean']:>10.0f} "
                f"{r['step_time_ms_mean']:>10.1f} "
                f"{r['peak_vram_gb_max']:>10.2f}"
            )
        report.append("")
    
    # Phase 2: dim_q曲线
    if 'phase2_dim_q' in results:
        report.append("## Phase 2: dim_q能力曲线")
        report.append("-" * 60)
        report.append(f"{'配置':<20} {'PPL':>10} {'E_pred':>10} {'hyp_dist':>10}")
        report.append("-" * 60)
        
        for r in results['phase2_dim_q']:
            report.append(
                f"{r['config_name']:<20} "
                f"{r['final_PPL']:>10.1f} "
                f"{r['final_E_pred']:>10.4f} "
                f"{r['final_hyp_dist']:>10.4f}"
            )
        report.append("")
    
    # Phase 3: 消融
    if 'phase3_ablation' in results:
        report.append("## Phase 3: 几何约束消融")
        report.append("-" * 70)
        report.append(f"{'配置':<15} {'PPL':>8} {'L_lyap':>10} {'L_atlas':>10} {'L_conn':>10} {'L_hyp':>10}")
        report.append("-" * 70)
        
        for r in results['phase3_ablation']:
            report.append(
                f"{r['config_name']:<15} "
                f"{r['final_PPL']:>8.1f} "
                f"{r['final_L_lyap']:>10.4f} "
                f"{r['final_L_atlas']:>10.4f} "
                f"{r['final_L_conn']:>10.4f} "
                f"{r['final_L_hyp']:>10.4f}"
            )
        report.append("")
        
        # 梯度分布
        report.append("### 梯度分布 (端口 vs 主体)")
        report.append("-" * 60)
        report.append(f"{'配置':<15} {'encoder':>10} {'decoder':>10} {'entity':>10} {'比例':>15}")
        report.append("-" * 60)
        
        for r in results['phase3_ablation']:
            enc = r.get('grad_norm_encoder', 0)
            dec = r.get('grad_norm_decoder', 0)
            ent = r.get('grad_norm_entity', 0)
            total = enc + dec + ent + 1e-8
            ratio = f"{(enc+dec)/total*100:.0f}% / {ent/total*100:.0f}%"
            report.append(
                f"{r['config_name']:<15} "
                f"{enc:>10.4f} "
                f"{dec:>10.4f} "
                f"{ent:>10.4f} "
                f"{ratio:>15}"
            )
        report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    with open(output_dir / 'report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def generate_plots(results: Dict, output_dir: Path):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("跳过图表生成（matplotlib未安装）")
        return
    
    # 图1: Phase 1 - B×L吞吐热力图
    if 'phase1_compute' in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 提取B×L数据
        bl_data = [r for r in results['phase1_compute'] if r['config_name'].startswith('B')]
        if bl_data:
            B_vals = sorted(set(r['config']['batch_size'] for r in bl_data))
            L_vals = sorted(set(r['config']['max_seq_len'] for r in bl_data))
            
            # 动态调整字体大小
            font_size = 8 if len(L_vals) < 10 else 6

            
            # tokens/s
            ax = axes[0]
            toks_matrix = []
            for B in B_vals:
                row = []
                for L in L_vals:
                    matching = [r for r in bl_data 
                                if r['config']['batch_size'] == B 
                                and r['config']['max_seq_len'] == L]
                    row.append(matching[0]['tokens_per_sec_mean'] if matching else 0)
                toks_matrix.append(row)
            
            im = ax.imshow(toks_matrix, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(len(L_vals)))
            ax.set_xticklabels(L_vals)
            ax.set_yticks(range(len(B_vals)))
            ax.set_yticklabels(B_vals)
            ax.set_xlabel('Sequence Length (L)')
            ax.set_ylabel('Batch Size (B)')
            ax.set_title('Throughput (tokens/s)')
            plt.colorbar(im, ax=ax)
            
            # 添加数值标注
            for i in range(len(B_vals)):
                for j in range(len(L_vals)):
                    ax.text(j, i, f'{toks_matrix[i][j]:.0f}', 
                            ha='center', va='center', fontsize=font_size)
            
            # VRAM
            ax = axes[1]
            vram_matrix = []
            for B in B_vals:
                row = []
                for L in L_vals:
                    matching = [r for r in bl_data 
                                if r['config']['batch_size'] == B 
                                and r['config']['max_seq_len'] == L]
                    row.append(matching[0]['peak_vram_gb_max'] if matching else 0)
                vram_matrix.append(row)
            
            im = ax.imshow(vram_matrix, aspect='auto', cmap='Blues')
            ax.set_xticks(range(len(L_vals)))
            ax.set_xticklabels(L_vals)
            ax.set_yticks(range(len(B_vals)))
            ax.set_yticklabels(B_vals)
            ax.set_xlabel('Sequence Length (L)')
            ax.set_ylabel('Batch Size (B)')
            ax.set_title('Peak VRAM (GB)')
            plt.colorbar(im, ax=ax)
            
            for i in range(len(B_vals)):

                for j in range(len(L_vals)):
                    ax.text(j, i, f'{vram_matrix[i][j]:.1f}', 
                            ha='center', va='center', fontsize=font_size)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'phase1_compute.png', dpi=150)
        plt.close()
        print(f"已生成: {output_dir / 'phase1_compute.png'}")
    
    # 图2: Phase 2 - dim_q曲线
    if 'phase2_dim_q' in results:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        hyp_data = [r for r in results['phase2_dim_q'] if 'hyp' in r['config_name']]
        euc_data = [r for r in results['phase2_dim_q'] if 'euc' in r['config_name']]
        
        # PPL曲线
        ax = axes[0]
        if hyp_data:
            dims = [r['config']['dim_q'] for r in hyp_data]
            ppls = [r['final_PPL'] for r in hyp_data]
            ax.plot(dims, ppls, 'o-', label='Hyperbolic', color='blue', linewidth=2)
        if euc_data:
            dims = [r['config']['dim_q'] for r in euc_data]
            ppls = [r['final_PPL'] for r in euc_data]
            ax.plot(dims, ppls, 's--', label='Euclidean', color='orange', linewidth=2)
        ax.set_xlabel('dim_q')
        ax.set_ylabel('Validation PPL')
        ax.set_title('PPL vs dim_q')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 双曲距离分布
        ax = axes[1]
        if hyp_data:
            dims = [r['config']['dim_q'] for r in hyp_data]
            dists = [r['final_hyp_dist'] for r in hyp_data]
            ax.bar(dims, dists, alpha=0.7, label='Hyperbolic Distance')
        ax.set_xlabel('dim_q')
        ax.set_ylabel('Mean Hyperbolic Distance')
        ax.set_title('Hyperbolic Distance Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'phase2_dim_q.png', dpi=150)
        plt.close()
        print(f"已生成: {output_dir / 'phase2_dim_q.png'}")
    
    # 图3: Phase 3 - 消融分析
    if 'phase3_ablation' in results:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        data = results['phase3_ablation']
        names = [r['config_name'] for r in data]
        
        # PPL对比
        ax = axes[0]
        ppls = [r['final_PPL'] for r in data]
        bars = ax.bar(names, ppls, color=['gray', 'green', 'blue', 'red', 'purple'])
        ax.set_ylabel('PPL')
        ax.set_title('Ablation: PPL Comparison')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        
        # 梯度分布堆叠图
        ax = axes[1]
        enc_grads = [r.get('grad_norm_encoder', 0) for r in data]
        dec_grads = [r.get('grad_norm_decoder', 0) for r in data]
        ent_grads = [r.get('grad_norm_entity', 0) for r in data]
        
        x = range(len(names))
        ax.bar(x, enc_grads, label='Encoder', color='steelblue')
        ax.bar(x, dec_grads, bottom=enc_grads, label='Decoder', color='lightblue')
        ax.bar(x, ent_grads, bottom=[e+d for e,d in zip(enc_grads, dec_grads)], 
               label='Entity', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Distribution by Module')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'phase3_ablation.png', dpi=150)
        plt.close()
        print(f"已生成: {output_dir / 'phase3_ablation.png'}")
    
    # 图4: 参数分布饼图
    if results:
        first_result = list(results.values())[0]
        if first_result and 'param_breakdown' in first_result[0]:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            breakdown = first_result[0]['param_breakdown']
            labels = ['Encoder', 'Decoder', 'Entity', 'Atlas', 'Connection']
            sizes = [
                breakdown.get('encoder', 0),
                breakdown.get('decoder', 0),
                breakdown.get('entity', 0),
                breakdown.get('atlas', 0),
                breakdown.get('connection', 0),
            ]
            
            # 过滤掉0值
            filtered = [(l, s) for l, s in zip(labels, sizes) if s > 0]
            if filtered:
                labels, sizes = zip(*filtered)
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                       colors=colors[:len(sizes)], startangle=90, pctdistance=0.85, labeldistance=1.1)
                ax.set_title('Parameter Distribution')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'param_distribution.png', dpi=150)
            plt.close()
            print(f"已生成: {output_dir / 'param_distribution.png'}")


def main():
    parser = argparse.ArgumentParser(description='基准测试结果可视化')
    parser.add_argument('result_dir', type=str, help='结果目录')
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"错误: 目录不存在 {result_dir}")
        sys.exit(1)
    
    print(f"加载结果: {result_dir}")
    results = load_results(result_dir)
    
    if not results:
        print("未找到结果文件")
        sys.exit(1)
    
    print(f"找到 {len(results)} 个结果文件: {list(results.keys())}")
    
    # 生成报告
    generate_text_report(results, result_dir)
    
    # 生成图表
    generate_plots(results, result_dir)
    
    print("\n可视化完成!")


if __name__ == '__main__':
    main()

