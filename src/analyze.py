"""
analyze.py — 统计分析

对应任务清单第 5 步：统计分析
============================================================

功能流程
--------
1. 读取时间序列 CSV（timeseries_point_values.csv）
2. 读取场景元数据（scene_metadata.csv, ndvi_metadata.csv）
3. 质量筛选：标记 T2 + NDVI 异常低的场景为云污染，排除后重新计算逐年均值
4. 趋势分析：
   a. 线性回归（NDVI ~ Year）：斜率 = 年变化率，R² = 拟合优度
   b. Mann-Kendall 趋势检验：非参数单调趋势显著性
5. 区域对比：中心城区 vs 郊区的趋势差异
6. 输出结果 CSV 至 outputs/tables/

输出文件
--------
outputs/tables/
├── quality_flags.csv           # 逐景质量标记
├── trend_per_point.csv          # 逐点趋势分析（斜率、p值、显著性）
├── trend_per_region.csv         # 区域趋势分析
└── analysis_summary.csv         # 总体分析摘要

运行方式
--------
    python src/analyze.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    PROJECT_ROOT,
    TABLES_DIR,
    load_params,
)

# ---------------------------------------------------------------------------
# 质量筛选阈值
# ---------------------------------------------------------------------------
# T2 场景且 NDVI 均值低于此阈值 → 标记为云污染
CLOUD_CONTAMINATION_NDVI_THRESHOLD: float = 0.1
# 趋势显著性水平
SIGNIFICANCE_LEVEL: float = 0.05


# ---------------------------------------------------------------------------
# 读取数据
# ---------------------------------------------------------------------------
def load_all_data(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """
    读取所有分析所需的 CSV 文件。

    Returns
    -------
    dict with keys:
        - scene_metadata : 场景元数据（tier, cloud_cover 等）
        - ndvi_metadata  : NDVI 统计（mean, std 等）
        - point_values   : 逐点逐景 NDVI 原始值
        - annual_mean    : 逐点逐年 NDVI 均值
        - region_mean    : 区域逐年统计
    """
    data = {}

    # 场景元数据
    scene_meta_path = tables_dir / "scene_metadata.csv"
    if scene_meta_path.exists():
        data["scene_metadata"] = pd.read_csv(scene_meta_path)
        print(f"[OK] 读取场景元数据: {len(data['scene_metadata'])} 景")
    else:
        print(f"[WARN] 未找到 {scene_meta_path.name}")
        data["scene_metadata"] = pd.DataFrame()

    # NDVI 元数据
    ndvi_meta_path = tables_dir / "ndvi_metadata.csv"
    if ndvi_meta_path.exists():
        data["ndvi_metadata"] = pd.read_csv(ndvi_meta_path)
        print(f"[OK] 读取 NDVI 元数据: {len(data['ndvi_metadata'])} 景")
    else:
        print(f"[WARN] 未找到 {ndvi_meta_path.name}")
        data["ndvi_metadata"] = pd.DataFrame()

    # 逐点逐景 NDVI 值
    point_path = tables_dir / "timeseries_point_values.csv"
    if point_path.exists():
        data["point_values"] = pd.read_csv(point_path)
        print(f"[OK] 读取逐点 NDVI 值: {len(data['point_values'])} 点")
    else:
        print(f"[ERROR] 未找到 {point_path.name}，无法分析")
        data["point_values"] = pd.DataFrame()

    # 逐点逐年均值
    annual_path = tables_dir / "timeseries_annual_mean.csv"
    if annual_path.exists():
        data["annual_mean"] = pd.read_csv(annual_path)
        print(f"[OK] 读取逐年均值: {len(data['annual_mean'])} 点")
    else:
        data["annual_mean"] = pd.DataFrame()

    # 区域逐年统计
    region_path = tables_dir / "timeseries_region_mean.csv"
    if region_path.exists():
        data["region_mean"] = pd.read_csv(region_path)
        print(f"[OK] 读取区域统计: {len(data['region_mean'])} 条记录")
    else:
        data["region_mean"] = pd.DataFrame()

    return data


# ---------------------------------------------------------------------------
# 质量筛选
# ---------------------------------------------------------------------------
def flag_cloud_contaminated_scenes(
    scene_metadata: pd.DataFrame,
    ndvi_metadata: pd.DataFrame,
    threshold: float = CLOUD_CONTAMINATION_NDVI_THRESHOLD,
) -> pd.DataFrame:
    """
    标记云污染场景。

    判定规则：
    - tier == T2 且 ndvi_mean < threshold → cloud_contaminated
    - 否则 → valid

    Returns
    -------
    pd.DataFrame with columns: scene_id, tier, ndvi_mean, quality_flag
    """
    # 合并场景元数据与 NDVI 统计
    if scene_metadata.empty or ndvi_metadata.empty:
        return pd.DataFrame(columns=["scene_id", "tier", "ndvi_mean", "quality_flag"])

    merged = scene_metadata[["scene_id", "tier"]].merge(
        ndvi_metadata[["scene_id", "mean"]].rename(columns={"mean": "ndvi_mean"}),
        on="scene_id",
        how="left",
    )

    # 标记
    flags = []
    for _, row in merged.iterrows():
        tier = str(row.get("tier", ""))
        ndvi_mean = row.get("ndvi_mean", np.nan)

        if tier == "T2" and not np.isnan(ndvi_mean) and ndvi_mean < threshold:
            flags.append("cloud_contaminated")
        else:
            flags.append("valid")

    merged["quality_flag"] = flags
    return merged[["scene_id", "tier", "ndvi_mean", "quality_flag"]]


def recalculate_annual_means(
    point_values: pd.DataFrame,
    quality_flags: pd.DataFrame,
) -> pd.DataFrame:
    """
    排除云污染场景后重新计算逐年 NDVI 均值。

    Parameters
    ----------
    point_values : pd.DataFrame
        逐点逐景 NDVI 原始值（含 point_id, region, x, y, <scene_id> 列）
    quality_flags : pd.DataFrame
        每景的质量标记

    Returns
    -------
    pd.DataFrame: 逐点逐年 NDVI 均值（排除污染场景后）
    """
    if point_values.empty:
        return pd.DataFrame()

    # 识别场景列（非元数据列）
    meta_cols = {"point_id", "region", "x", "y"}
    scene_cols = [c for c in point_values.columns if c not in meta_cols]

    # 构建场景 ID → 年份 映射
    scene_to_year: dict[str, str] = {}
    for col in scene_cols:
        # scene_id 格式：LC08_L2SP_127041_20180607_20200831_02_T1
        parts = col.split("_")
        if len(parts) >= 4:
            date_str = parts[3]
            scene_to_year[col] = date_str[:4] if len(date_str) >= 4 else ""

    # 标记有效场景
    valid_scenes = set(
        quality_flags.loc[quality_flags["quality_flag"] == "valid", "scene_id"]
    )
    contaminated_scenes = set(
        quality_flags.loc[
            quality_flags["quality_flag"] == "cloud_contaminated", "scene_id"
        ]
    )

    # 按年份分组有效场景
    years = sorted(set(scene_to_year.values()))
    df_annual = point_values[["point_id", "region", "x", "y"]].copy()

    for year in years:
        year_valid_scenes = [
            s for s in scene_cols
            if scene_to_year.get(s, "") == year and s in valid_scenes
        ]
        if year_valid_scenes:
            df_annual[year] = point_values[year_valid_scenes].mean(axis=1)
        else:
            df_annual[year] = np.nan

    n_excluded = len(contaminated_scenes)
    if n_excluded > 0:
        print(f"[INFO] 排除 {n_excluded} 个云污染场景: "
              f"{', '.join(sorted(contaminated_scenes))}")

    return df_annual


# ---------------------------------------------------------------------------
# 趋势分析方法
# ---------------------------------------------------------------------------
def linear_trend(
    years: np.ndarray, values: np.ndarray
) -> dict[str, float]:
    """
    线性回归趋势分析。

    y = slope * x + intercept

    Parameters
    ----------
    years : np.ndarray
        自变量（年份）
    values : np.ndarray
        因变量（NDVI 值）

    Returns
    -------
    dict with:
        - slope : 回归斜率（NDVI/年）
        - intercept : 截距
        - r_squared : 决定系数 R²
        - p_value : 斜率显著性 p 值（t 检验）
        - std_err : 斜率标准误
    """
    # 过滤 NaN
    mask = ~np.isnan(values)
    x = years[mask].astype(float)
    y = values[mask].astype(float)

    n = len(x)
    if n < 3:
        return {
            "slope": np.nan, "intercept": np.nan,
            "r_squared": np.nan, "p_value": np.nan,
            "std_err": np.nan, "n": n,
        }

    # 最小二乘拟合
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # 预测值与残差
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # R²
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # t 检验 for slope
    if n > 2 and ss_res > 0:
        std_err_slope = math.sqrt(ss_res / (n - 2)) / math.sqrt(
            float(np.sum((x - np.mean(x)) ** 2))
        )
        t_stat = slope / std_err_slope if std_err_slope > 0 else np.nan
        # 双尾 p 值（t 分布的近似，用正态分布）
        if not np.isnan(t_stat):
            p_value = 2 * (1 - _norm_cdf(abs(t_stat)))
        else:
            p_value = np.nan
    else:
        std_err_slope = np.nan
        p_value = np.nan

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": float(r_squared),
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "std_err": float(std_err_slope) if not np.isnan(std_err_slope) else np.nan,
        "n": n,
    }


def mann_kendall_test(values: np.ndarray) -> dict[str, float]:
    """
    Mann-Kendall 非参数趋势检验。

    检验序列是否存在单调趋势（上升或下降），
    不要求数据正态分布，适合小样本。

    Parameters
    ----------
    values : np.ndarray
        时间序列值（按时间顺序排列）

    Returns
    -------
    dict with:
        - s : MK 统计量 S
        - z : 标准化统计量 Z
        - p_value : 双尾 p 值
        - trend : 趋势方向 ('increasing' / 'decreasing' / 'no_trend')
        - tau : Kendall's tau（趋势强度）
    """
    # 过滤 NaN
    y = values[~np.isnan(values)]
    n = len(y)

    if n < 3:
        return {
            "s": 0, "z": 0, "p_value": 1.0,
            "trend": "insufficient_data", "tau": np.nan, "n": n,
        }

    # 计算 S 统计量
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(y[j] - y[i])

    # 计算方差（考虑 ties）
    # 统计重复值
    unique_vals, counts = np.unique(y, return_counts=True)
    tied_groups = counts[counts > 1]

    var_s = n * (n - 1) * (2 * n + 5) / 18
    if len(tied_groups) > 0:
        tie_correction = sum(t * (t - 1) * (2 * t + 5) for t in tied_groups)
        var_s -= tie_correction / 18

    # Z 统计量
    if s > 0:
        z = (s - 1) / math.sqrt(var_s) if var_s > 0 else 0
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s) if var_s > 0 else 0
    else:
        z = 0

    # 双尾 p 值
    p_value = 2 * (1 - _norm_cdf(abs(z)))

    # 趋势方向
    if p_value < SIGNIFICANCE_LEVEL:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no_trend"

    # Kendall's tau
    tau = s / (n * (n - 1) / 2) if n > 1 else np.nan

    return {
        "s": int(s),
        "z": float(z),
        "p_value": float(p_value),
        "trend": trend,
        "tau": float(tau),
        "n": n,
    }


def _norm_cdf(z: float) -> float:
    """
    标准正态分布累积分布函数（CDF）近似。

    使用 error function 计算，无需 scipy 依赖。
    """
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# 逐点趋势分析
# ---------------------------------------------------------------------------
def analyze_per_point(
    annual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    对每个采样点做趋势分析。

    Parameters
    ----------
    annual_df : pd.DataFrame
        逐点逐年 NDVI 均值（列: point_id, region, x, y, <year>...）

    Returns
    -------
    pd.DataFrame: 每行的趋势分析结果
    """
    meta_cols = {"point_id", "region", "x", "y"}
    year_cols = sorted(
        [c for c in annual_df.columns if c not in meta_cols],
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    years = np.array([int(y) for y in year_cols])

    records = []
    for _, row in annual_df.iterrows():
        values = np.array([row[y] for y in year_cols], dtype=float)

        # 线性回归
        lr = linear_trend(years, values)

        # Mann-Kendall
        mk = mann_kendall_test(values)

        # 综合判断
        is_significant = (
            lr["p_value"] < SIGNIFICANCE_LEVEL if not np.isnan(lr["p_value"]) else False
        )
        mk_significant = mk["p_value"] < SIGNIFICANCE_LEVEL

        records.append({
            "point_id": row["point_id"],
            "region": row["region"],
            "x": row["x"],
            "y": row["y"],
            "n_years": lr["n"],
            # 线性回归
            "linear_slope": lr["slope"],
            "linear_intercept": lr["intercept"],
            "linear_r_squared": lr["r_squared"],
            "linear_p_value": lr["p_value"],
            "linear_std_err": lr["std_err"],
            "linear_significant": is_significant,
            # Mann-Kendall
            "mk_s": mk["s"],
            "mk_z": mk["z"],
            "mk_p_value": mk["p_value"],
            "mk_tau": mk["tau"],
            "mk_trend": mk["trend"],
            "mk_significant": mk_significant,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 区域趋势分析
# ---------------------------------------------------------------------------
def analyze_per_region(
    annual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    对每个区域做趋势分析。

    先按区域和年份计算 NDVI 均值，再对区域均值序列做趋势分析。
    """
    meta_cols = {"point_id", "region", "x", "y"}
    year_cols = sorted(
        [c for c in annual_df.columns if c not in meta_cols],
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    years = np.array([int(y) for y in year_cols])

    records = []
    for region in annual_df["region"].unique():
        mask = annual_df["region"] == region
        region_data = annual_df.loc[mask]

        # 逐年均值
        annual_means = np.array([
            region_data[y].mean() for y in year_cols
        ], dtype=float)

        # 线性回归
        lr = linear_trend(years, annual_means)

        # Mann-Kendall
        mk = mann_kendall_test(annual_means)

        is_significant = (
            lr["p_value"] < SIGNIFICANCE_LEVEL
            if not np.isnan(lr["p_value"]) else False
        )

        records.append({
            "region": region,
            "n_points": int(mask.sum()),
            "n_years": lr["n"],
            "ndvi_start": float(annual_means[0]) if len(annual_means) > 0 else np.nan,
            "ndvi_end": float(annual_means[-1]) if len(annual_means) > 0 else np.nan,
            "ndvi_mean": float(np.nanmean(annual_means)),
            # 线性回归
            "linear_slope": lr["slope"],
            "linear_intercept": lr["intercept"],
            "linear_r_squared": lr["r_squared"],
            "linear_p_value": lr["p_value"],
            "linear_significant": is_significant,
            # Mann-Kendall
            "mk_s": mk["s"],
            "mk_z": mk["z"],
            "mk_p_value": mk["p_value"],
            "mk_tau": mk["tau"],
            "mk_trend": mk["trend"],
            "mk_significant": mk["p_value"] < SIGNIFICANCE_LEVEL,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 区域对比
# ---------------------------------------------------------------------------
def compare_regions(
    region_trends: pd.DataFrame,
) -> dict[str, Any]:
    """
    对比中心城区与郊区的趋势差异。
    """
    if region_trends.empty:
        return {}

    summary = {}
    for region in region_trends["region"].unique():
        row = region_trends[region_trends["region"] == region].iloc[0]
        summary[region] = {
            "slope": row["linear_slope"],
            "r_squared": row["linear_r_squared"],
            "p_value": row["linear_p_value"],
            "mk_trend": row["mk_trend"],
            "mk_p_value": row["mk_p_value"],
            "ndvi_mean": row["ndvi_mean"],
        }

    # 计算差异
    if "urban" in summary and "suburban" in summary:
        summary["difference"] = {
            "slope_diff": summary["urban"]["slope"] - summary["suburban"]["slope"],
            "ndvi_mean_diff": (
                summary["urban"]["ndvi_mean"] - summary["suburban"]["ndvi_mean"]
            ),
        }

    return summary


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_results(
    quality_flags: pd.DataFrame,
    trend_per_point: pd.DataFrame,
    trend_per_region: pd.DataFrame,
    region_comparison: dict,
    output_dir: Path,
) -> None:
    """写出分析结果 CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 质量标记
    qf_path = output_dir / "quality_flags.csv"
    quality_flags.to_csv(qf_path, index=False, encoding="utf-8")
    n_contam = (quality_flags["quality_flag"] == "cloud_contaminated").sum()
    print(f"[OK] 质量标记: {qf_path.relative_to(PROJECT_ROOT)} "
          f"({n_contam} 个云污染场景)")

    # 2. 逐点趋势
    tp_path = output_dir / "trend_per_point.csv"
    trend_per_point.to_csv(tp_path, index=False, encoding="utf-8")
    n_sig = trend_per_point["linear_significant"].sum()
    print(f"[OK] 逐点趋势: {tp_path.relative_to(PROJECT_ROOT)} "
          f"({len(trend_per_point)} 点, {n_sig} 个显著)")

    # 3. 区域趋势
    tr_path = output_dir / "trend_per_region.csv"
    trend_per_region.to_csv(tr_path, index=False, encoding="utf-8")
    print(f"[OK] 区域趋势: {tr_path.relative_to(PROJECT_ROOT)}")

    # 4. 摘要
    summary_path = output_dir / "analysis_summary.csv"
    summary_records = []
    for region, stats in region_comparison.items():
        if region == "difference":
            summary_records.append({
                "comparison": "urban - suburban",
                "slope_difference": stats.get("slope_diff", np.nan),
                "ndvi_mean_difference": stats.get("ndvi_mean_diff", np.nan),
            })
        else:
            summary_records.append({
                "comparison": f"{region} trend",
                "slope_per_year": stats.get("slope", np.nan),
                "r_squared": stats.get("r_squared", np.nan),
                "p_value": stats.get("p_value", np.nan),
                "mk_trend": stats.get("mk_trend", ""),
                "mk_p_value": stats.get("mk_p_value", np.nan),
                "ndvi_mean": stats.get("ndvi_mean", np.nan),
            })
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"[OK] 分析摘要: {summary_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    """主入口"""
    params = load_params()
    tables_dir = TABLES_DIR

    print("=" * 70)
    print("统计分析 — Guiyang NDVI Time Series")
    print("=" * 70)
    print(f"输出目录: {tables_dir}")
    print(f"显著性水平: {SIGNIFICANCE_LEVEL}")

    # 1. 读取数据
    print("\n--- 读取数据 ---")
    data = load_all_data(tables_dir)

    if data["point_values"].empty:
        print("[ERROR] 无逐点 NDVI 数据，请先运行 timeseries.py")
        return 1

    # 2. 质量筛选
    print("\n--- 质量筛选 ---")
    quality_flags = flag_cloud_contaminated_scenes(
        data["scene_metadata"], data["ndvi_metadata"]
    )

    if not quality_flags.empty:
        n_contam = (quality_flags["quality_flag"] == "cloud_contaminated").sum()
        n_valid = (quality_flags["quality_flag"] == "valid").sum()
        print(f"有效场景: {n_valid} 景")
        print(f"云污染场景: {n_contam} 景")
        if n_contam > 0:
            contam = quality_flags[quality_flags["quality_flag"] == "cloud_contaminated"]
            for _, row in contam.iterrows():
                print(f"  ☁️ {row['scene_id']} (tier={row['tier']}, "
                      f"NDVI_mean={row['ndvi_mean']:.4f})")

    # 3. 重新计算逐年均值（排除污染场景）
    print("\n--- 重新计算逐年均值 ---")
    annual_clean = recalculate_annual_means(
        data["point_values"], quality_flags
    )

    if annual_clean.empty:
        print("[ERROR] 无法计算逐年均值")
        return 1

    # 4. 逐点趋势分析
    print("\n--- 逐点趋势分析 ---")
    trend_per_point = analyze_per_point(annual_clean)

    n_inc = (trend_per_point["linear_slope"] > 0).sum()
    n_dec = (trend_per_point["linear_slope"] < 0).sum()
    n_sig = trend_per_point["linear_significant"].sum()
    print(f"上升趋势: {n_inc} 点")
    print(f"下降趋势: {n_dec} 点")
    print(f"显著趋势 (p < {SIGNIFICANCE_LEVEL}): {n_sig} 点")

    # 5. 区域趋势分析
    print("\n--- 区域趋势分析 ---")
    trend_per_region = analyze_per_region(annual_clean)

    for _, row in trend_per_region.iterrows():
        sig = "✓ 显著" if row["linear_significant"] else "✗ 不显著"
        mk_sig = "✓" if row["mk_significant"] else "✗"
        print(f"  {row['region']}: "
              f"slope={row['linear_slope']:+.5f}/yr "
              f"(p={row['linear_p_value']:.4f} {sig}), "
              f"MK: {row['mk_trend']} (p={row['mk_p_value']:.4f} {mk_sig})")

    # 6. 区域对比
    print("\n--- 区域对比 ---")
    region_comparison = compare_regions(trend_per_region)

    if "urban" in region_comparison and "suburban" in region_comparison:
        u = region_comparison["urban"]
        s = region_comparison["suburban"]
        d = region_comparison.get("difference", {})
        print(f"  中心城区: slope={u['slope']:+.5f}/yr, mean NDVI={u['ndvi_mean']:.4f}")
        print(f"  郊区    : slope={s['slope']:+.5f}/yr, mean NDVI={s['ndvi_mean']:.4f}")
        print(f"  趋势差异: {d.get('slope_diff', 0):+.5f}/yr")
        print(f"  均值差异: {d.get('ndvi_mean_diff', 0):+.4f}")

    # 7. 写出结果
    print("\n--- 输出结果 ---")
    write_results(
        quality_flags, trend_per_point, trend_per_region,
        region_comparison, tables_dir,
    )

    print("\n" + "=" * 70)
    print("统计分析完成")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
