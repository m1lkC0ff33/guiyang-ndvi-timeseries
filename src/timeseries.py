"""
timeseries.py — 时间序列提取

对应任务清单第 4 步：提取时间序列
============================================================

功能流程
--------
1. 扫描 NDVI GeoTIFF 列表（由 ndvi_calc.py 生成，位于 processed/ 根目录或子目录）
2. 确定采样点来源（按优先级）：
   a) 用户提供的矢量文件（含 point_id, region 字段）
   b) 研究区边界矢量文件 → 在边界内随机生成
   c) 无任何矢量时 → 在第一景 NDVI 影像的有效像元内随机生成
3. 用 rasterio 按点坐标提取每景 NDVI 值（自动处理 CRS 转换）
4. 整理为逐点 × 逐景的结构化表格
5. 计算逐年均值与区域统计量
6. 输出 CSV 表格

输出文件
--------
data/processed/
├── timeseries_point_values.csv      # 逐点逐景 NDVI 原始值
├── timeseries_annual_mean.csv        # 逐点逐年 NDVI 均值
└── timeseries_region_mean.csv        # 区域逐年 NDVI 均值 + 标准差

运行方式
--------
    python src/timeseries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    PROCESSED_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    load_params,
    resolve_boundary_path,
)


# ---------------------------------------------------------------------------
# NDVI 场景扫描
# ---------------------------------------------------------------------------
def find_ndvi_scenes(processed_dir: Path) -> list[dict]:
    """
    扫描 processed 目录，找出所有 NDVI GeoTIFF。

    查找位置：
    - processed_dir 根目录下的 *_NDVI.tif（ndvi_calc.py 的输出位置）
    - processed_dir 子目录下的 *_NDVI.tif（向后兼容）
    """
    scenes: list[dict] = []

    # 1. 根目录下的 NDVI 文件（主要位置）
    ndvi_files = list(processed_dir.glob("*_NDVI.tif"))

    # 2. 子目录下的 NDVI 文件（向后兼容）
    for scene_dir in processed_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        if scene_dir.name.startswith("_") or scene_dir.name.startswith("."):
            continue
        ndvi_files.extend(scene_dir.glob("*_NDVI.tif"))

    # 去重（同一个文件可能被匹配两次）
    ndvi_files = sorted(set(ndvi_files), key=lambda p: p.name)

    for ndvi_path in ndvi_files:
        scene_id = ndvi_path.name.replace("_NDVI.tif", "")

        # 从 scene_id 提取年份
        # 格式：LC08_L2SP_127041_20180607_20200831_02_T1
        parts = scene_id.split("_")
        date_str = parts[3] if len(parts) >= 4 else ""
        year = date_str[:4] if len(date_str) >= 4 else ""
        month = date_str[4:6] if len(date_str) >= 6 else ""
        day = date_str[6:8] if len(date_str) >= 8 else ""
        acq_date = f"{year}-{month}-{day}" if year else ""

        # 传感器信息
        sensor = "LC08" if scene_id.startswith("LC08") else "LC09"

        scenes.append({
            "scene_id": scene_id,
            "ndvi_path": ndvi_path,
            "sensor": sensor,
            "year": year,
            "acquisition_date": acq_date,
        })

    scenes.sort(key=lambda s: s["acquisition_date"])
    return scenes


# ---------------------------------------------------------------------------
# 采样点生成
# ---------------------------------------------------------------------------
def generate_points_from_boundary(
    boundary_path: Path,
    n_points_per_region: int = 10,
    urban_ratio: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    在研究区边界内自动生成采样点。

    策略：
    - 在边界外接矩形内随机生成点，保留落在边界内的点
    - 按到质心的距离排序，最近的归为 urban，其余归为 suburban

    返回的 DataFrame 的 attrs["crs"] 记录点的 CRS（边界矢量的 CRS）。
    """
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        print("[WARN] 边界矢量无 CRS 信息，默认设为 WGS84 (EPSG:4326)")
        gdf = gdf.set_crs("EPSG:4326")

    points_crs = str(gdf.crs)
    bounds = gdf.total_bounds  # [xmin, ymin, xmax, ymax]
    union_geom = gdf.geometry.union_all()
    centroid = union_geom.centroid

    total_points = n_points_per_region * 2
    rng = np.random.RandomState(seed)

    points: list[Point] = []
    attempts = 0
    max_attempts = total_points * 100

    while len(points) < total_points and attempts < max_attempts:
        x = rng.uniform(bounds[0], bounds[2])
        y = rng.uniform(bounds[1], bounds[3])
        pt = Point(x, y)
        if union_geom.covers(pt):
            points.append(pt)
        attempts += 1

    if len(points) < total_points:
        print(f"[WARN] 仅生成 {len(points)}/{total_points} 个采样点")

    # 按到质心距离排序，最近的归 urban
    points.sort(key=lambda p: p.distance(centroid))
    n_urban = int(len(points) * urban_ratio)

    records = []
    for i, pt in enumerate(points):
        region = "urban" if i < n_urban else "suburban"
        records.append({
            "point_id": f"P{i + 1:03d}",
            "region": region,
            "x": float(pt.x),
            "y": float(pt.y),
        })

    df = pd.DataFrame(records)
    df.attrs["crs"] = points_crs
    return df


def generate_points_from_ndvi_extent(
    ndvi_path: Path,
    n_points_per_region: int = 10,
    urban_ratio: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    在单景 NDVI 影像的有效像元内随机生成采样点。

    回退方案（仅 1 景可用时）：在有效像元内随机采样。
    多景可用时优先用 generate_points_valid_in_all_scenes。
    """
    with rasterio.open(ndvi_path) as src:
        ndvi_arr = src.read(1)
        transform = src.transform
        crs = src.crs

    valid_mask = ~np.isnan(ndvi_arr)
    valid_rows, valid_cols = np.where(valid_mask)

    if len(valid_rows) == 0:
        raise ValueError(f"NDVI 影像中没有有效像元: {ndvi_path}")

    print(f"[INFO] NDVI 影像有效像元数: {len(valid_rows)} ({ndvi_path.name})")

    center_row = float(np.mean(valid_rows))
    center_col = float(np.mean(valid_cols))
    center_x, center_y = xy(transform, center_row, center_col)

    rng = np.random.RandomState(seed)
    total_points = n_points_per_region * 2
    n_valid = len(valid_rows)

    if n_valid < total_points:
        print(f"[WARN] 有效像元数 ({n_valid}) 小于所需采样点数 ({total_points})，将允许重复")
        indices = rng.choice(n_valid, total_points, replace=True)
    else:
        indices = rng.choice(n_valid, total_points, replace=False)

    points_xy: list[tuple[float, float]] = []
    for idx in indices:
        r, c = valid_rows[idx], valid_cols[idx]
        x, y = xy(transform, r, c)
        points_xy.append((float(x), float(y)))

    points_xy.sort(
        key=lambda p: ((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) ** 0.5
    )
    n_urban = int(len(points_xy) * urban_ratio)

    records = []
    for i, (x, y) in enumerate(points_xy):
        region = "urban" if i < n_urban else "suburban"
        records.append({
            "point_id": f"P{i + 1:03d}",
            "region": region,
            "x": x,
            "y": y,
        })

    df = pd.DataFrame(records)
    df.attrs["crs"] = str(crs)
    return df


def generate_points_valid_in_all_scenes(
    scenes: list[dict],
    n_points_per_region: int = 10,
    urban_ratio: float = 0.4,
    seed: int = 42,
    min_valid_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    在所有 NDVI 场景的共同有效区域内生成采样点。

    当数据跨越多个 path/row 时，确保每个采样点在尽可能多的场景中
    都有有效值，避免出现某些年份整组点为 NaN 的情况。

    策略：
    1. 计算所有场景外接矩形的交集（共同地理范围）
    2. 在共同范围内随机生成候选点
    3. 对每个候选点，检查在所有场景中是否为有效（非 NaN）
    4. 只保留有效率 >= min_valid_ratio 的点
    5. 用共同范围的中心作为城区中心代理，按距离划分 urban/suburban

    Parameters
    ----------
    scenes : list[dict]
        NDVI 场景列表
    n_points_per_region : int
        每区域采样点数
    urban_ratio : float
        中心城区点占比
    seed : int
        随机种子
    min_valid_ratio : float
        最低有效率阈值（0~1），点在多少比例的场景中有效才被保留
    """
    from rasterio.windows import Window

    if not scenes:
        raise ValueError("场景列表为空")

    # 1. 计算所有场景外接矩形的交集
    bounds_list = []
    crs_set = set()
    for scene in scenes:
        with rasterio.open(scene["ndvi_path"]) as src:
            b = src.bounds  # left, bottom, right, top
            bounds_list.append((b.left, b.right, b.bottom, b.top))
            crs_set.add(str(src.crs))

    if len(crs_set) > 1:
        print(f"[WARN] 场景 CRS 不一致: {crs_set}，使用第一个场景的 CRS")
    points_crs = crs_set.pop() if crs_set else "EPSG:4326"

    common_left = max(b[0] for b in bounds_list)
    common_right = min(b[1] for b in bounds_list)
    common_bottom = max(b[2] for b in bounds_list)
    common_top = min(b[3] for b in bounds_list)

    if common_left >= common_right or common_bottom >= common_top:
        raise ValueError(
            "场景之间没有共同范围（外接矩形无交集），"
            "无法生成对所有场景都有效的采样点"
        )

    print(f"[INFO] {len(scenes)} 景场景的共同范围:")
    print(f"       x: [{common_left:.0f}, {common_right:.0f}]")
    print(f"       y: [{common_bottom:.0f}, {common_top:.0f}]")
    print(f"       CRS: {points_crs}")

    center_x = (common_left + common_right) / 2
    center_y = (common_bottom + common_top) / 2

    # 2. 打开所有场景（保持打开，用于逐点检查）
    srcs = [rasterio.open(s["ndvi_path"]) for s in scenes]
    n_scenes = len(srcs)
    min_valid_count = int(np.ceil(n_scenes * min_valid_ratio))

    total_points = n_points_per_region * 2
    rng = np.random.RandomState(seed)

    points_xy: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = total_points * 200

    print(f"[INFO] 生成 {total_points} 个有效率 >= {min_valid_ratio} 的采样点...")

    while len(points_xy) < total_points and attempts < max_attempts:
        x = rng.uniform(common_left, common_right)
        y = rng.uniform(common_bottom, common_top)

        # 检查该点在所有场景中的有效性
        valid_count = 0
        for src in srcs:
            r, c = rowcol(src.transform, x, y)
            r, c = int(r), int(c)
            if 0 <= r < src.height and 0 <= c < src.width:
                # 用窗口读取单个像元（高效）
                window = Window(c, r, 1, 1)
                pixel = src.read(1, window=window)
                if not np.isnan(pixel[0, 0]):
                    valid_count += 1

        if valid_count >= min_valid_count:
            points_xy.append((x, y))

        attempts += 1

    # 关闭所有文件
    for src in srcs:
        src.close()

    if len(points_xy) < total_points:
        print(f"[WARN] 仅找到 {len(points_xy)}/{total_points} 个满足条件的点")
        print(f"       (尝试 {attempts} 次, 阈值有效率 {min_valid_ratio})")
        if len(points_xy) < 4:
            raise ValueError(
                f"无法在共同范围内找到足够的有效采样点 "
                f"(仅 {len(points_xy)} 个)，请降低 min_valid_ratio 或检查数据覆盖"
            )

    # 按到中心距离排序，最近的归 urban
    points_xy.sort(
        key=lambda p: ((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) ** 0.5
    )
    n_urban = int(len(points_xy) * urban_ratio)

    records = []
    for i, (x, y) in enumerate(points_xy):
        region = "urban" if i < n_urban else "suburban"
        records.append({
            "point_id": f"P{i + 1:03d}",
            "region": region,
            "x": x,
            "y": y,
        })

    df = pd.DataFrame(records)
    df.attrs["crs"] = points_crs
    return df


def load_sampling_points_from_file(points_file: Path) -> pd.DataFrame:
    """
    从用户提供的矢量文件加载采样点。

    要求文件包含 point_id 和 region 字段（缺失时会自动补全）。
    """
    import geopandas as gpd

    gdf = gpd.read_file(points_file)
    if gdf.crs is None:
        print("[WARN] 采样点文件无 CRS 信息，默认设为 WGS84 (EPSG:4326)")
        gdf = gdf.set_crs("EPSG:4326")

    if "region" not in gdf.columns:
        print("[WARN] 采样点文件缺少 'region' 字段，默认全部标记为 'urban'")
        gdf["region"] = "urban"
    if "point_id" not in gdf.columns:
        gdf["point_id"] = [f"P{i + 1:03d}" for i in range(len(gdf))]

    df = pd.DataFrame({
        "point_id": gdf["point_id"].astype(str),
        "region": gdf["region"].astype(str),
        "x": gdf.geometry.x.astype(float),
        "y": gdf.geometry.y.astype(float),
    })
    df.attrs["crs"] = str(gdf.crs)
    print(f"[OK] 加载 {len(df)} 个采样点 (从 {points_file})")
    return df


def load_or_generate_sampling_points(
    points_file: Path | None,
    boundary_path: Path | None,
    n_points: int,
    scenes: list[dict] | None,
) -> pd.DataFrame:
    """
    按优先级加载或生成采样点：

    1. 用户提供采样点矢量文件 → 加载
    2. 研究区边界矢量文件 → 在边界内随机生成
    3. 多景 NDVI → 在所有场景共同有效区域内生成（推荐，支持多 path/row）
    4. 单景 NDVI → 在有效像元内随机生成
    5. 都没有 → 报错

    Parameters
    ----------
    points_file : Path or None
        用户提供的采样点文件（GeoJSON/SHP）
    boundary_path : Path or None
        研究区边界矢量文件
    n_points : int
        每区域采样点数（自动生成时使用）
    scenes : list[dict] or None
        NDVI 场景列表（多景时优先用共同区域生成）
    """
    # 1. 用户采样点文件
    if points_file is not None and points_file.exists():
        return load_sampling_points_from_file(points_file)

    # 2. 边界矢量文件
    if boundary_path is not None and boundary_path.exists():
        print(f"[INFO] 未找到采样点文件，将在边界内随机生成 {n_points * 2} 个点")
        return generate_points_from_boundary(boundary_path, n_points)

    # 3. 多景 NDVI → 共同有效区域生成（推荐）
    if scenes and len(scenes) > 1:
        print(f"[INFO] 未找到边界矢量，在 {len(scenes)} 景 NDVI 的共同有效区域内生成采样点")
        return generate_points_valid_in_all_scenes(scenes, n_points)

    # 4. 单景 NDVI → 有效像元生成
    if scenes and len(scenes) == 1:
        print(f"[INFO] 仅 1 景 NDVI，在有效像元内生成采样点")
        return generate_points_from_ndvi_extent(scenes[0]["ndvi_path"], n_points)

    # 5. 都没有
    raise FileNotFoundError(
        "无法生成采样点：未找到采样点文件、边界矢量、或 NDVI 影像。\n"
        "请先运行 python src/ndvi_calc.py 生成 NDVI 影像，"
        "或提供采样点/边界矢量文件。"
    )


# ---------------------------------------------------------------------------
# NDVI 值按点提取
# ---------------------------------------------------------------------------
def extract_ndvi_at_points_batch(
    ndvi_path: Path,
    points_df: pd.DataFrame,
    points_crs: str,
) -> dict[str, float]:
    """
    批量提取：一次性读取整景 NDVI，对所有点进行采样。

    Parameters
    ----------
    ndvi_path : Path
        NDVI GeoTIFF 文件路径
    points_df : pd.DataFrame
        采样点数据（包含 point_id, x, y 列）
    points_crs : str
        采样点坐标的 CRS（如 'EPSG:4326' 或 'EPSG:32648'）

    Returns
    -------
    dict: point_id -> ndvi_value (float or NaN)
    """
    values: dict[str, float] = {}

    with rasterio.open(ndvi_path) as src:
        ndvi_arr = src.read(1)
        transform = src.transform
        img_crs = str(src.crs)

        # 预先准备所有点的坐标，若 CRS 不同则批量转换
        xs = points_df["x"].tolist()
        ys = points_df["y"].tolist()

        if img_crs != points_crs:
            xs_proj, ys_proj = warp_transform(points_crs, img_crs, xs, ys)
        else:
            xs_proj, ys_proj = xs, ys

        for i, row in points_df.iterrows():
            point_id = row["point_id"]
            x, y = xs_proj[i], ys_proj[i]

            r, c = rowcol(transform, x, y)
            r, c = int(r), int(c)

            if 0 <= r < src.height and 0 <= c < src.width:
                values[point_id] = float(ndvi_arr[r, c])
            else:
                values[point_id] = float("nan")

    return values


# ---------------------------------------------------------------------------
# 时间序列表格构建
# ---------------------------------------------------------------------------
def build_timeseries_tables(
    points_df: pd.DataFrame,
    scenes: list[dict],
    points_crs: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    构建三个层级的时间序列表格。

    Parameters
    ----------
    points_df : pd.DataFrame
        采样点（含 point_id, region, x, y）
    scenes : list[dict]
        NDVI 场景列表
    points_crs : str
        采样点坐标的 CRS

    Returns
    -------
    df_point_values : 逐点逐景 NDVI 原始值
    df_annual_mean : 逐点逐年 NDVI 均值
    df_region_mean : 区域逐年 NDVI 均值与标准差
    """
    # 1. 逐点逐景原始值表
    columns_base = ["point_id", "region", "x", "y"]
    df_point_values = points_df[columns_base].copy()

    for scene in scenes:
        scene_id = scene["scene_id"]
        ndvi_path = scene["ndvi_path"]

        print(f"  提取 {scene_id} NDVI 值...")
        values = extract_ndvi_at_points_batch(ndvi_path, points_df, points_crs)
        df_point_values[scene_id] = df_point_values["point_id"].map(values)

    # 2. 逐点逐年均值表
    years = sorted(set(s["year"] for s in scenes if s["year"]))
    df_annual = points_df[["point_id", "region", "x", "y"]].copy()

    for year in years:
        year_scenes = [s["scene_id"] for s in scenes if s["year"] == year]
        if year_scenes:
            df_annual[year] = df_point_values[year_scenes].mean(axis=1)
        else:
            df_annual[year] = np.nan

    # 3. 区域逐年统计
    region_records: list[dict] = []
    for year in years:
        for region in df_annual["region"].unique():
            mask = df_annual["region"] == region
            vals = df_annual.loc[mask, year].dropna()
            region_records.append({
                "region": region,
                "year": year,
                "mean": float(vals.mean()) if len(vals) > 0 else np.nan,
                "std": float(vals.std()) if len(vals) > 1 else np.nan,
                "count": int(len(vals)),
            })

    df_region_mean = pd.DataFrame(region_records)

    return df_point_values, df_annual, df_region_mean


# ---------------------------------------------------------------------------
# CSV 写出
# ---------------------------------------------------------------------------
def write_timeseries_csvs(
    df_point: pd.DataFrame,
    df_annual: pd.DataFrame,
    df_region: pd.DataFrame,
    output_dir: Path,
) -> None:
    """写出三个时间序列 CSV"""
    output_dir.mkdir(parents=True, exist_ok=True)

    point_csv = output_dir / "timeseries_point_values.csv"
    annual_csv = output_dir / "timeseries_annual_mean.csv"
    region_csv = output_dir / "timeseries_region_mean.csv"

    df_point.to_csv(point_csv, index=False, encoding="utf-8")
    n_scenes = len(df_point.columns) - 4
    print(f"[OK] 逐点逐景表: {point_csv.relative_to(PROJECT_ROOT)} ({len(df_point)} 点 × {n_scenes} 景)")

    df_annual.to_csv(annual_csv, index=False, encoding="utf-8")
    n_years = len(df_annual.columns) - 4
    print(f"[OK] 逐点逐年表: {annual_csv.relative_to(PROJECT_ROOT)} ({len(df_annual)} 点 × {n_years} 年)")

    df_region.to_csv(region_csv, index=False, encoding="utf-8")
    print(f"[OK] 区域逐年表: {region_csv.relative_to(PROJECT_ROOT)} ({len(df_region)} 条记录)")


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

    n_points = int(params.get("sampling", {}).get("n_points", 10))

    print("=" * 70)
    print("时间序列提取 — Guiyang NDVI Time Series")
    print("=" * 70)
    print(f"处理数据目录: {processed_dir}")
    print(f"每区域采样点: {n_points}")

    # 1. 先扫描 NDVI 场景（用于无边界时的回退）
    scenes = find_ndvi_scenes(processed_dir)
    if not scenes:
        print("[ERROR] 未发现任何 NDVI 数据，请先运行 python src/ndvi_calc.py")
        return 1

    years = sorted(set(s["year"] for s in scenes))
    print(f"\n发现 {len(scenes)} 景 NDVI 数据")
    print(f"年份范围: {years[0]} - {years[-1]} ({len(years)} 年)")
    print("-" * 70)

    # 2. 加载/生成采样点
    boundary_path = resolve_boundary_path()
    if boundary_path is None:
        bp = params.get("paths", {}).get("boundary_path")
        if bp:
            boundary_path = PROJECT_ROOT / bp

    points_file_str = params.get("paths", {}).get("sampling_points")
    points_file = Path(points_file_str) if points_file_str else None
    if points_file and not points_file.is_absolute():
        points_file = PROJECT_ROOT / points_file

    try:
        points_df = load_or_generate_sampling_points(
            points_file, boundary_path, n_points, scenes
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 1

    points_crs = points_df.attrs.get("crs", "EPSG:4326")
    n_urban = (points_df["region"] == "urban").sum()
    n_suburban = (points_df["region"] == "suburban").sum()
    print(f"\n采样点: {len(points_df)} 个 (中心城区 {n_urban}, 郊区 {n_suburban})")
    print(f"采样点 CRS: {points_crs}")

    # 3. 构建时间序列表格
    print("\n提取 NDVI 值...")
    df_point, df_annual, df_region = build_timeseries_tables(
        points_df, scenes, points_crs
    )

    # 4. 写出 CSV
    print("\n写出时间序列表格...")
    write_timeseries_csvs(df_point, df_annual, df_region, TABLES_DIR)

    # 5. 简要统计
    print("\n" + "=" * 70)
    print("时间序列提取完成")
    print("=" * 70)

    if not df_region.empty:
        print("\n区域逐年 NDVI 均值预览:")
        print(df_region.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
