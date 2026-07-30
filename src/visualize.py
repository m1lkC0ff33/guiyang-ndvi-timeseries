"""
visualize.py — 制图与可视化

对应任务清单第 6 步：制图与可视化
============================================================

功能流程
--------
1. 读取 NDVI GeoTIFF 与分析结果 CSV
2. 生成以下图表：
   a. NDVI 空间分布图（多景对比）
   b. NDVI 时间序列趋势图（区域均值 ± 标准差）
   c. 中心城区 vs 郊区对比图（箱线图）
   d. 趋势空间分布图（逐点斜率散点图）
3. 所有图表保存至 outputs/figures/

输出文件
--------
outputs/figures/
├── ndvi_spatial_maps.png       # NDVI 空间分布（4 景对比）
├── ndvi_timeseries_trend.png    # 时间序列趋势图
├── region_comparison.png       # 区域对比箱线图
└── trend_spatial_map.png       # 趋势空间分布

运行方式
--------
    python src/visualize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互后端，无需显示器
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    PROCESSED_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    FIGURES_DIR,
    load_params,
)

# ---------------------------------------------------------------------------
# 全局样式设置
# ---------------------------------------------------------------------------
# 中文字体设置（Windows 上用 Microsoft YaHei，回退到默认）
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# NDVI 专用色标：红→黄→绿
NDVI_CMAP = plt.cm.RdYlGn
NDVI_VMIN = -0.2
NDVI_VMAX = 0.8

# 趋势色标：红（下降）→白（无变化）→绿（上升）
TREND_CMAP = plt.cm.RdYlGn

DPI = 150


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _safe_title(ax, title, **kwargs):
    """安全设置标题（中文字体不可用时回退到英文）"""
    ax.set_title(title, **kwargs)


def _relpath(path: Path) -> str:
    """
    安全获取相对项目根目录的路径字符串，用于日志打印。

    当 path 不在 PROJECT_ROOT 之下（如单元测试中的 tmp_path），
    直接返回绝对路径字符串，避免 relative_to() 抛出 ValueError。
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 图 1：NDVI 空间分布图
# ---------------------------------------------------------------------------
def plot_ndvi_spatial_maps(
    processed_dir: Path,
    quality_flags: pd.DataFrame | None,
    output_path: Path,
    n_panels: int = 4,
) -> None:
    """
    生成 NDVI 空间分布图（多景对比）。

    策略：
    - 从所有 NDVI 场景中选取 n_panels 景作为代表
    - 优先选不同年份的有效（非云污染）场景
    - 4 景并排显示，共享色标
    """
    # 查找所有 NDVI 文件
    ndvi_files = sorted(processed_dir.glob("*_NDVI.tif"))
    if not ndvi_files:
        # 也查找子目录
        for d in processed_dir.iterdir():
            if d.is_dir() and not d.name.startswith(("_", ".")):
                ndvi_files.extend(d.glob("*_NDVI.tif"))
        ndvi_files = sorted(set(ndvi_files))

    if not ndvi_files:
        print("[WARN] 未找到 NDVI 文件，跳过空间分布图")
        return

    # 从文件名提取年份
    def extract_year(path):
        parts = path.stem.split("_")
        if len(parts) >= 4:
            return parts[3][:4]
        return ""

    # 排除云污染场景
    contaminated = set()
    if quality_flags is not None and not quality_flags.empty:
        contaminated = set(
            quality_flags.loc[
                quality_flags["quality_flag"] == "cloud_contaminated",
                "scene_id",
            ]
        )

    # 按年份去重，每年选第一个有效场景
    seen_years = {}
    for f in ndvi_files:
        year = extract_year(f)
        scene_id = f.stem.replace("_NDVI", "")
        if scene_id in contaminated:
            continue
        if year and year not in seen_years:
            seen_years[year] = f

    # 选取代表性年份（均匀分布）
    all_years = sorted(seen_years.keys())
    if len(all_years) <= n_panels:
        selected = [seen_years[y] for y in all_years]
    else:
        indices = np.linspace(0, len(all_years) - 1, n_panels, dtype=int)
        selected = [seen_years[all_years[i]] for i in indices]

    n = len(selected)
    if n == 0:
        print("[WARN] 无有效 NDVI 场景可用于绘图")
        return

    # 创建子图
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, ndvi_path in zip(axes, selected):
        with rasterio.open(ndvi_path) as src:
            ndvi = src.read(1)

            # 下采样以加速绘图（取每 10 个像元）
            step = 10
            ndvi_sub = ndvi[::step, ::step]

        year = extract_year(ndvi_path)
        im = ax.imshow(
            ndvi_sub,
            cmap=NDVI_CMAP,
            vmin=NDVI_VMIN,
            vmax=NDVI_VMAX,
            aspect="equal",
        )
        ax.set_title(f"{year}", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # 共享色标
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.04, pad=0.08, shrink=0.6)
    cbar.set_label("NDVI", fontsize=12)

    fig.suptitle("NDVI Spatial Distribution", fontsize=16, fontweight="bold", y=1.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] NDVI 空间分布图: {_relpath(output_path)} ({n} 景)")


# ---------------------------------------------------------------------------
# 图 2：时间序列趋势图
# ---------------------------------------------------------------------------
def plot_timeseries_trend(
    annual_df: pd.DataFrame,
    region_trends: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """
    生成 NDVI 时间序列趋势图。

    - 逐区域绘制年均 NDVI ± 标准差
    - 叠加线性回归趋势线
    """
    if annual_df.empty:
        print("[WARN] 无逐年均值数据，跳过趋势图")
        return

    meta_cols = {"point_id", "region", "x", "y"}
    year_cols = sorted(
        [c for c in annual_df.columns if c not in meta_cols],
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    years = [int(y) for y in year_cols]

    regions = annual_df["region"].unique()
    colors = {"urban": "#e74c3c", "suburban": "#2ecc71"}
    labels = {"urban": "Urban", "suburban": "Suburban"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for region in regions:
        mask = annual_df["region"] == region
        region_data = annual_df.loc[mask]

        # 逐年均值与标准差
        means = [region_data[y].mean() for y in year_cols]
        stds = [region_data[y].std() for y in year_cols]

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)

        color = colors.get(region, "#3498db")
        label = labels.get(region, region)

        # 均值线 + 标准差带
        ax.plot(years, means, "o-", color=color, label=label,
                linewidth=2, markersize=6, zorder=3)
        ax.fill_between(
            years, means - stds, means + stds,
            alpha=0.2, color=color, zorder=1,
        )

        # 趋势线
        if region_trends is not None and not region_trends.empty:
            rt = region_trends[region_trends["region"] == region]
            if not rt.empty:
                slope = rt.iloc[0]["linear_slope"]
                # intercept 列可能缺失（旧版 CSV），缺失时从均值与斜率反推
                if "linear_intercept" in rt.columns:
                    intercept = rt.iloc[0]["linear_intercept"]
                else:
                    intercept = float(np.nanmean(means)) - slope * float(np.mean(years))
                if not np.isnan(slope):
                    trend_y = slope * np.array(years) + intercept
                    sig = ""
                    if rt.iloc[0].get("linear_significant", False):
                        sig = " *"
                    ax.plot(
                        years, trend_y, "--", color=color, alpha=0.6,
                        label=f"{label} trend ({slope:+.4f}/yr{sig})",
                        linewidth=1.5, zorder=2,
                    )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("NDVI", fontsize=12)
    ax.set_title("NDVI Time Series Trend", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.set_ylim(NDVI_VMIN, NDVI_VMAX + 0.2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 时间序列趋势图: {_relpath(output_path)}")


# ---------------------------------------------------------------------------
# 图 3：区域对比箱线图
# ---------------------------------------------------------------------------
def plot_region_comparison(
    annual_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    生成中心城区 vs 郊区的 NDVI 对比箱线图。

    每个年份一组箱线图，urban 和 suburban 并排。
    """
    if annual_df.empty:
        print("[WARN] 无逐年均值数据，跳过区域对比图")
        return

    meta_cols = {"point_id", "region", "x", "y"}
    year_cols = sorted(
        [c for c in annual_df.columns if c not in meta_cols],
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    years = [int(y) for y in year_cols]

    regions = sorted(annual_df["region"].unique())

    # 准备数据：每年每区域的 NDVI 值列表
    data_urban = []
    data_suburban = []
    for y in year_cols:
        u_vals = annual_df.loc[annual_df["region"] == "urban", y].dropna().tolist()
        s_vals = annual_df.loc[annual_df["region"] == "suburban", y].dropna().tolist()
        data_urban.append(u_vals)
        data_suburban.append(s_vals)

    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.35
    x = np.arange(len(years))

    bp1 = ax.boxplot(
        data_urban, positions=x - width / 2, widths=width,
        patch_artist=True, showfliers=True,
        boxprops=dict(facecolor="#e74c3c", alpha=0.6),
        medianprops=dict(color="black"),
    )

    bp2 = ax.boxplot(
        data_suburban, positions=x + width / 2, widths=width,
        patch_artist=True, showfliers=True,
        boxprops=dict(facecolor="#2ecc71", alpha=0.6),
        medianprops=dict(color="black"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=10)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("NDVI", fontsize=12)
    ax.set_title("Urban vs Suburban NDVI Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Urban", "Suburban"], fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 区域对比图: {_relpath(output_path)}")


# ---------------------------------------------------------------------------
# 图 4：趋势空间分布图
# ---------------------------------------------------------------------------
def plot_trend_spatial(
    trend_per_point: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    生成趋势斜率的空间散点图。

    - 每个点位置画一个圆点，颜色表示斜率
    - 显著趋势用边缘标记
    """
    if trend_per_point.empty:
        print("[WARN] 无趋势数据，跳过趋势空间图")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 斜率范围
    slopes = trend_per_point["linear_slope"].dropna()
    if len(slopes) == 0:
        print("[WARN] 无有效斜率值")
        return

    abs_max = max(abs(slopes.min()), abs(slopes.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # 颜色按区域
    region_markers = {"urban": "s", "suburban": "o"}

    for region in trend_per_point["region"].unique():
        mask = trend_per_point["region"] == region
        sub = trend_per_point.loc[mask]

        marker = region_markers.get(region, "o")

        # 显著趋势用大标记，不显著用小标记
        for sig in [True, False]:
            sig_mask = sub["linear_significant"] == sig
            if not sig_mask.any():
                continue

            data = sub.loc[sig_mask]
            scatter = ax.scatter(
                data["x"], data["y"],
                c=data["linear_slope"],
                cmap=TREND_CMAP,
                norm=norm,
                marker=marker,
                s=120 if sig else 60,
                edgecolors="black" if sig else "none",
                linewidths=1.5 if sig else 0,
                label=f"{region} ({'p<0.05' if sig else 'n.s.'})",
                zorder=3,
            )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("NDVI Trend Slope (/yr)", fontsize=11)

    ax.set_xlabel("Easting (m)", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=11)
    ax.set_title("NDVI Trend Spatial Distribution",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 趋势空间分布图: {_relpath(output_path)}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    """主入口"""
    print("=" * 70)
    print("制图与可视化 — Guiyang NDVI Time Series")
    print("=" * 70)
    print(f"图表输出目录: {FIGURES_DIR}")

    # 1. 读取分析数据
    print("\n--- 读取数据 ---")

    # 质量标记
    qf_path = TABLES_DIR / "quality_flags.csv"
    quality_flags = None
    if qf_path.exists():
        quality_flags = pd.read_csv(qf_path)
        print(f"[OK] 质量标记: {len(quality_flags)} 景")
    else:
        print(f"[WARN] 未找到 {qf_path.name}")

    # 逐点逐年均值
    annual_path = TABLES_DIR / "timeseries_annual_mean.csv"
    annual_df = pd.DataFrame()
    if annual_path.exists():
        annual_df = pd.read_csv(annual_path)
        print(f"[OK] 逐年均值: {len(annual_df)} 点")
    else:
        print(f"[WARN] 未找到 {annual_path.name}")

    # 逐点趋势
    trend_path = TABLES_DIR / "trend_per_point.csv"
    trend_per_point = pd.DataFrame()
    if trend_path.exists():
        trend_per_point = pd.read_csv(trend_path)
        print(f"[OK] 逐点趋势: {len(trend_per_point)} 点")
    else:
        print(f"[WARN] 未找到 {trend_path.name}")

    # 区域趋势
    region_trend_path = TABLES_DIR / "trend_per_region.csv"
    region_trends = None
    if region_trend_path.exists():
        region_trends = pd.read_csv(region_trend_path)
        print(f"[OK] 区域趋势: {len(region_trends)} 区域")

    # 2. 生成图表
    print("\n--- 生成图表 ---")

    # 图 1：NDVI 空间分布
    plot_ndvi_spatial_maps(
        PROCESSED_DIR, quality_flags,
        FIGURES_DIR / "ndvi_spatial_maps.png",
    )

    # 图 2：时间序列趋势
    plot_timeseries_trend(
        annual_df, region_trends,
        FIGURES_DIR / "ndvi_timeseries_trend.png",
    )

    # 图 3：区域对比
    plot_region_comparison(
        annual_df,
        FIGURES_DIR / "region_comparison.png",
    )

    # 图 4：趋势空间分布
    plot_trend_spatial(
        trend_per_point,
        FIGURES_DIR / "trend_spatial_map.png",
    )

    print("\n" + "=" * 70)
    print("可视化完成")
    print(f"图表保存在: {_relpath(FIGURES_DIR)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
