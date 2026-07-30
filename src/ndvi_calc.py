"""
ndvi_calc.py — 计算归一化植被指数 (NDVI)

对应任务清单第 3 步：计算 NDVI
============================================================

功能流程
--------
1. 扫描 data/processed/ 下经预处理的 scene 目录
2. 读取每景的 Band 4 (Red) 与 Band 5 (NIR) 反射率数据
3. 计算 NDVI = (NIR - Red) / (NIR + Red)
4. 处理无效像元（分母为 0、NaN）
5. 输出 NDVI 单波段 GeoTIFF
6. 汇总输出 NDVI 统计信息至 ndvi_metadata.csv

输出目录结构
-----------
data/processed/
├── LC08_L2SP_127041_20180607_20200831_02_T1/
│   ├── LC08_..._SR_B4_reflectance.tif
│   ├── LC08_..._SR_B5_reflectance.tif
│   └── LC08_..._NDVI.tif              (本脚本生成)
├── ...
└── ndvi_metadata.csv                  (本脚本生成)

运行方式
--------
    # 从项目根运行
    python src/ndvi_calc.py

    # 已计算过的景会自动跳过（增量处理），可重复运行
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# 允许从项目根或 src/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    PROCESSED_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    NDVI_VALID_MIN,
    NDVI_VALID_MAX,
    load_params,
)

# ---------------------------------------------------------------------------
# NDVI 计算核心
# ---------------------------------------------------------------------------
def calculate_ndvi(red_arr: np.ndarray, nir_arr: np.ndarray, 
                   denominator_min: float = 0.0001) -> tuple[np.ndarray, dict]:
    """
    计算 NDVI。

    Parameters
    ----------
    red_arr : np.ndarray
        Band 4 (Red) 反射率数组 (float32)
    nir_arr : np.ndarray
        Band 5 (NIR) 反射率数组 (float32)
    denominator_min : float
        分母 (NIR + Red) 的最小阈值，低于此值置为 NaN。
        避免在水体等低反射率区域产生不稳定的高 NDVI。

    Returns
    -------
    ndvi : np.ndarray
        NDVI 数组，float32
    stats : dict
        该景 NDVI 的统计信息：
        - mean, std, min, max
        - valid_count, nan_count
    """
    # 计算分母
    denominator = nir_arr + red_arr

    # 初始化 NDVI 数组为 NaN
    ndvi = np.full_like(red_arr, np.nan, dtype=np.float32)

    # 有效像元掩码：两个波段都非 NaN 且分母大于阈值
    valid_mask = (
        ~np.isnan(red_arr)
        & ~np.isnan(nir_arr)
        & (denominator > denominator_min)
    )

    # 对有效像元计算 NDVI
    ndvi[valid_mask] = (nir_arr[valid_mask] - red_arr[valid_mask]) / denominator[valid_mask]

    # 统计有效像元的 NDVI 值
    valid_ndvi = ndvi[valid_mask]
    
    stats = {
        "mean": float(np.mean(valid_ndvi)) if len(valid_ndvi) > 0 else 0.0,
        "std": float(np.std(valid_ndvi)) if len(valid_ndvi) > 0 else 0.0,
        "min": float(np.min(valid_ndvi)) if len(valid_ndvi) > 0 else 0.0,
        "max": float(np.max(valid_ndvi)) if len(valid_ndvi) > 0 else 0.0,
        "valid_count": int(np.sum(valid_mask)),
        "nan_count": int(np.sum(~valid_mask)),
    }

    return ndvi.astype(np.float32), stats


# ---------------------------------------------------------------------------
# 场景扫描
# ---------------------------------------------------------------------------
def find_processed_scenes(processed_dir: Path) -> list[dict]:
    """
    扫描 processed 目录，找出所有包含 B4_reflectance 和 B5_reflectance 对的场景。
    """
    scenes: list[dict] = []
    # 查找所有子目录
    for scene_dir in processed_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        # 排除 __pycache__ 等特殊目录
        if scene_dir.name.startswith("_") or scene_dir.name.startswith("."):
            continue
            
        # 查找 B4 和 B5 反射率文件
        b4_files = list(scene_dir.glob("*_SR_B4_reflectance.tif"))
        b5_files = list(scene_dir.glob("*_SR_B5_reflectance.tif"))
        
        if not b4_files or not b5_files:
            print(f"[WARN] {scene_dir.name} 缺少 B4 或 B5 反射率文件，跳过")
            continue
            
        b4_path = b4_files[0]
        b5_path = b5_files[0]
        scene_id = b4_path.name.replace("_SR_B4_reflectance.tif", "")
        
        # 检查 B4 和 B5 是否属于同一个 scene
        expected_b5 = b4_path.name.replace("_SR_B4_reflectance.tif", "_SR_B5_reflectance.tif")
        if expected_b5 != b5_path.name:
            print(f"[WARN] {scene_dir.name} B4 和 B5 文件名不匹配，跳过")
            continue

        scenes.append({
            "scene_id": scene_id,
            "scene_dir": scene_dir,
            "b4_path": b4_path,
            "b5_path": b5_path,
        })
        
    return scenes


# ---------------------------------------------------------------------------
# 单景 NDVI 计算
# ---------------------------------------------------------------------------
def process_scene_ndvi(scene: dict, ndvi_dir: Path, 
                       denominator_min: float) -> dict:
    """
    处理单景 NDVI 计算。
    """
    scene_id = scene["scene_id"]
    ndvi_path = ndvi_dir / f"{scene_id}_NDVI.tif"
    
    if ndvi_path.exists():
        status = "skipped"
        print(f"[SKIP] 已计算 NDVI: {scene_id}")
        return {
            "scene_id": scene_id,
            "ndvi_output": str(ndvi_path.relative_to(PROJECT_ROOT)),
            "status": status,
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "valid_count": 0, "nan_count": 0,
        }

    # 读取 B4 和 B5
    with rasterio.open(scene["b4_path"]) as src:
        red = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        
    with rasterio.open(scene["b5_path"]) as src:
        nir = src.read(1).astype(np.float32)

    # 计算 NDVI
    ndvi, stats = calculate_ndvi(red, nir, denominator_min)

    # 写入 NDVI
    profile.update(
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        tiled=True,
    )
    
    with rasterio.open(ndvi_path, "w", **profile) as dst:
        dst.write(ndvi.astype(np.float32), 1)

    status = "processed"
    print(f"[OK] {scene_id} | NDVI mean={stats['mean']:.4f} | valid={stats['valid_count']}")

    return {
        "scene_id": scene_id,
        "ndvi_output": str(ndvi_path.relative_to(PROJECT_ROOT)),
        "status": status,
        **stats,
    }


# ---------------------------------------------------------------------------
# 元数据 CSV 写出
# ---------------------------------------------------------------------------
NDVI_METADATA_FIELDS: tuple[str, ...] = (
    "scene_id",
    "ndvi_output",
    "status",
    "mean",
    "std",
    "min",
    "max",
    "valid_count",
    "nan_count",
)


def write_ndvi_metadata_csv(records: list[dict], csv_path: Path) -> None:
    """写出 NDVI 元数据 CSV"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NDVI_METADATA_FIELDS)
        writer.writeheader()
        for r in records:
            row = {
                k: ("" if isinstance(v, float) and v != v else v)
                for k, v in r.items()
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    """主入口"""
    params = load_params()
    processed_dir = Path(
        params.get("paths", {}).get("processed_dir", PROCESSED_DIR)
    )
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir

    denominator_min = float(
        params.get("ndvi", {}).get("denominator_min", 0.0001)
    )

    print("=" * 70)
    print("NDVI 计算 — Guiyang NDVI Time Series")
    print("=" * 70)
    print(f"预处理数据目录: {processed_dir}")
    print(f"分母最小阈值  : {denominator_min}")

    if not processed_dir.exists():
        print(f"[ERROR] 预处理数据目录不存在: {processed_dir}")
        print("        请先运行 python src/preprocess.py")
        return 1

    # 扫描已预处理的场景
    scenes = find_processed_scenes(processed_dir)
    print(f"\n发现 {len(scenes)} 个已预处理场景")
    print("-" * 70)

    if not scenes:
        print("[WARN] 未发现任何可计算的场景")
        return 0

    # 逐景计算
    records: list[dict] = []
    for scene in scenes:
        record = process_scene_ndvi(scene, processed_dir, denominator_min)
        records.append(record)

    # 写 NDVI 元数据 CSV（输出到 outputs/tables/）
    ndvi_metadata_csv = TABLES_DIR / "ndvi_metadata.csv"
    write_ndvi_metadata_csv(records, ndvi_metadata_csv)
    print("-" * 70)
    print(f"[OK] NDVI 元数据汇总: {ndvi_metadata_csv.relative_to(PROJECT_ROOT)}")

    # 统计
    n_processed = sum(1 for r in records if r["status"] == "processed")
    n_skipped = sum(1 for r in records if r["status"] == "skipped")
    
    # 计算所有场景的平均 NDVI
    valid_means = [r["mean"] for r in records if r["status"] == "processed"]
    if valid_means:
        overall_mean = float(np.mean(valid_means))
        print(f"\n完成: {n_processed} 景新计算, {n_skipped} 景跳过")
        print(f"所有新计算场景 NDVI 均值: {overall_mean:.4f}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
