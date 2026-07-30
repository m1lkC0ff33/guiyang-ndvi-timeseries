"""
preprocess.py — 数据读取与预处理

对应任务清单第 2 步：数据读取与预处理
============================================================

功能流程
--------
1. 扫描 data/raw/ 下所有 Landsat SR_B4 / SR_B5 / MTL 三件套
2. 从文件名解析 scene_id、传感器、path/row、日期
3. 从 MTL 读取元数据（云量、反射率缩放参数、CRS）
4. 读取 Band 4 (Red) 与 Band 5 (NIR)，DN → 反射率
5. 标记无效像元（DN=0 填充、超出合理反射率范围）
6. 按贵阳市边界裁剪（若边界文件存在，否则跳过裁剪）
7. 输出预处理后的浮点 GeoTIFF 至 data/processed/<scene_id>/
8. 汇总输出 scene_metadata.csv 至 data/processed/

输出目录结构
-----------
data/processed/
├── LC08_L2SP_127041_20180607_20200831_02_T1/
│   ├── LC08_..._SR_B4_reflectance.tif   (float32, 已缩放)
│   └── LC08_..._SR_B5_reflectance.tif
├── LC08_L2SP_127041_20180826_20200831_02_T1/
│   └── ...
└── scene_metadata.csv                    (所有景元数据汇总)

运行方式
--------
    # 从项目根运行
    python src/preprocess.py

    # 或在 src/ 目录下运行
    python preprocess.py

    # 已处理过的景会自动跳过（按输出文件存在判断），可重复运行
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
import yaml
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask

# 允许从项目根或 src/ 目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DEFAULT_PROFILE_UPDATE,
    DEFAULT_SR_ADD,
    DEFAULT_SR_MULT,
    NIR_BAND_FILE,
    PROCESSED_DIR,
    PROJECT_ROOT,
    PARAMS_YAML,
    RAW_DIR,
    RED_BAND_FILE,
    SR_FILL_DN,
    SR_VALID_MAX,
    SR_VALID_MIN,
    load_params,
    resolve_boundary_path,
)

# ---------------------------------------------------------------------------
# Landsat Collection 2 文件名解析
#   示例：LC08_L2SP_127041_20180607_20200831_02_T1
#         └┬─┘ └┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └┬┘ └┬┘
#         sensor level  path/row acq   proc   col tier
# ---------------------------------------------------------------------------
LANDSAT_SCENE_PATTERN = re.compile(
    r"^(LC0[89])_"        # sensor: LC08 or LC09
    r"L2SP_"              # level: Level-2 Surface Reflectance
    r"(\d{3})(\d{3})_"    # path (3 digits) + row (3 digits)
    r"(\d{8})_"           # acquisition date YYYYMMDD
    r"(\d{8})_"           # processing date YYYYMMDD
    r"(\d{2})_"           # collection number
    r"(T[12])$"           # tier: T1 or T2
)


def parse_scene_id(scene_id: str) -> dict[str, Any]:
    """
    解析 Landsat Collection 2 scene ID 为结构化字典。

    Parameters
    ----------
    scene_id : str
        形如 ``LC08_L2SP_127041_20180607_20200831_02_T1`` 的标识符

    Returns
    -------
    dict with keys: sensor, level, path, row, acquisition_date,
                    processing_date, collection, tier

    Raises
    ------
    ValueError
        scene_id 不符合 Landsat Collection 2 命名约定
    """
    m = LANDSAT_SCENE_PATTERN.match(scene_id)
    if not m:
        raise ValueError(f"无效的 Landsat scene_id 格式: {scene_id!r}")
    sensor, path, row, acq, proc, col, tier = m.groups()
    return {
        "sensor": sensor,
        "level": "L2SP",
        "path": int(path),
        "row": int(row),
        "acquisition_date": f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}",
        "processing_date": f"{proc[:4]}-{proc[4:6]}-{proc[6:8]}",
        "collection": int(col),
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# MTL 元数据解析
#   MTL 是 ODL (Object Description Language) 格式，扁平化为 dict 即可
# ---------------------------------------------------------------------------
def parse_mtl(mtl_path: Path) -> dict[str, Any]:
    """
    解析 Landsat Collection 2 MTL 文件，提取本流程关心的字段。

    返回字段
    --------
    spacecraft         : 'LANDSAT_8' / 'LANDSAT_9'
    sensor_id          : 'OLI_TIRS'
    wrs_path, wrs_row  : int
    date_acquired      : 'YYYY-MM-DD'
    scene_center_time  : 'HH:MM:SS.sssZ'
    cloud_cover        : float, 整景云量百分比
    cloud_cover_land   : float, 陆地部分云量
    sun_azimuth        : float
    sun_elevation      : float
    reflectance_mult_b4, reflectance_add_b4 : float, B4 缩放
    reflectance_mult_b5, reflectance_add_b5 : float, B5 缩放
    map_projection     : 'UTM'
    utm_zone           : int
    datum              : 'WGS84'
    """
    fields: dict[str, str] = {}
    with open(mtl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过 GROUP / END_GROUP / END / 空行
            if not line or line.startswith("GROUP") or line.startswith("END"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"')
            if key:
                fields[key] = value

    def _get_float(key: str, default: float) -> float:
        try:
            return float(fields.get(key, default))
        except (TypeError, ValueError):
            return default

    def _get_int(key: str, default: int) -> int:
        try:
            return int(fields.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "spacecraft": fields.get("SPACECRAFT_ID", ""),
        "sensor_id": fields.get("SENSOR_ID", ""),
        "wrs_path": _get_int("WRS_PATH", 0),
        "wrs_row": _get_int("WRS_ROW", 0),
        "date_acquired": fields.get("DATE_ACQUIRED", ""),
        "scene_center_time": fields.get("SCENE_CENTER_TIME", ""),
        "cloud_cover": _get_float("CLOUD_COVER", float("nan")),
        "cloud_cover_land": _get_float("CLOUD_COVER_LAND", float("nan")),
        "sun_azimuth": _get_float("SUN_AZIMUTH", float("nan")),
        "sun_elevation": _get_float("SUN_ELEVATION", float("nan")),
        "reflectance_mult_b4": _get_float(
            "REFLECTANCE_MULT_BAND_4", DEFAULT_SR_MULT
        ),
        "reflectance_add_b4": _get_float(
            "REFLECTANCE_ADD_BAND_4", DEFAULT_SR_ADD
        ),
        "reflectance_mult_b5": _get_float(
            "REFLECTANCE_MULT_BAND_5", DEFAULT_SR_MULT
        ),
        "reflectance_add_b5": _get_float(
            "REFLECTANCE_ADD_BAND_5", DEFAULT_SR_ADD
        ),
        "map_projection": fields.get("MAP_PROJECTION", ""),
        "utm_zone": _get_int("UTM_ZONE", 0),
        "datum": fields.get("DATUM", ""),
    }


# ---------------------------------------------------------------------------
# 影像读取与反射率缩放
# ---------------------------------------------------------------------------
def load_band_as_reflectance(
    band_path: Path, mult: float, add: float
) -> tuple[np.ndarray, dict]:
    """
    读取 Landsat SR 波段并转为浮点反射率。

    处理流程
    --------
    1. 以 float32 读取原始 DN
    2. 应用缩放：SR = DN * mult + add
    3. 将填充像元（DN=0）置为 NaN
    4. 将超出合理范围 [SR_VALID_MIN, SR_VALID_MAX] 的像元置为 NaN
    5. 返回数组与更新后的 rasterio profile（dtype=float32, nodata=NaN）

    Returns
    -------
    reflectance : np.ndarray, float32
    profile : dict, 已更新 dtype 与 nodata
    """
    with rasterio.open(band_path) as src:
        dn = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    fill_mask = dn == SR_FILL_DN
    reflectance = dn * mult + add
    # 填充像元先置 NaN
    reflectance[fill_mask] = np.nan
    # 超出有效范围的像元也置 NaN（注意排除已经是 NaN 的填充像元）
    out_of_range = (reflectance < SR_VALID_MIN) | (reflectance > SR_VALID_MAX)
    reflectance[out_of_range & ~np.isnan(reflectance)] = np.nan

    profile.update(dtype="float32", nodata=np.nan)
    return reflectance, profile


# ---------------------------------------------------------------------------
# 边界裁剪
# ---------------------------------------------------------------------------
def load_boundary_gdf(boundary_path: Optional[Path]):
    """
    读取边界矢量。返回 GeoDataFrame 或 None。

    延迟导入 geopandas，避免在无该依赖的纯解析场景下报错。
    """
    if boundary_path is None or not boundary_path.exists():
        return None
    import geopandas as gpd

    gdf = gpd.read_file(boundary_path)
    # 若有多个要素，合并为单一几何
    if len(gdf) > 1:
        gdf = gpd.GeoDataFrame(
            {"geometry": [gdf.geometry.unary_union]},
            crs=gdf.crs,
        )
    return gdf


def clip_to_boundary(
    arr: np.ndarray, profile: dict, boundary_gdf
) -> tuple[np.ndarray, dict]:
    """
    按边界裁剪影像：边界外置 NaN，并裁剪到边界外接矩形（crop=True）。

    若 boundary_gdf 为 None，直接返回原数组与 profile。
    若边界 CRS 与影像 CRS 不一致，自动重投影边界到影像 CRS。
    """
    if boundary_gdf is None:
        return arr, profile

    img_crs = profile.get("crs")
    if boundary_gdf.crs != img_crs:
        boundary_gdf = boundary_gdf.to_crs(img_crs)

    geom = boundary_gdf.geometry.unary_union.__geo_interface__
    # 用 MemoryFile 包装数组后调用 rasterio.mask
    # 这样既能 crop 边界外接矩形，又能把边界外像元设为 nodata
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(arr.astype(np.float32), 1)
        with memfile.open() as src:
            out, out_transform = rio_mask(
                src, [geom], crop=True, nodata=np.nan, filled=True
            )
            out = out[0]  # (1, H, W) -> (H, W)
            new_profile = src.profile.copy()
            new_profile.update(
                height=out.shape[0],
                width=out.shape[1],
                transform=out_transform,
            )
    return out, new_profile


# ---------------------------------------------------------------------------
# 场景扫描与配对
# ---------------------------------------------------------------------------
def find_scene_pairs(raw_dir: Path) -> list[dict]:
    """
    扫描 raw 目录，找出所有 SR_B4 + SR_B5 + MTL 三件套。

    自动排除：
      - 重复下载产生的 ``SR_B4 (1).TIF`` / ``SR_B4 (2).TIF``（命名不符合 Landsat 规范）
      - 缺失 B5 或 MTL 配对的单波段文件
      - 文件名不符合 Landsat Collection 2 规范的文件
    """
    scenes: list[dict] = []
    b4_files = sorted(raw_dir.glob("*_SR_B4.TIF"))
    for b4 in b4_files:
        # 取去掉 _SR_B4.TIF 后缀的 scene_id
        scene_id = b4.name[: -len("_SR_B4.TIF")]
        b5 = raw_dir / f"{scene_id}_SR_B5.TIF"
        mtl = raw_dir / f"{scene_id}_MTL.txt"

        if not b5.exists():
            print(f"[WARN] 缺失 B5 配对，跳过: {b4.name}")
            continue
        if not mtl.exists():
            print(f"[WARN] 缺失 MTL 配对，跳过: {b4.name}")
            continue

        try:
            info = parse_scene_id(scene_id)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        scenes.append(
            {
                "scene_id": scene_id,
                "b4_path": b4,
                "b5_path": b5,
                "mtl_path": mtl,
                **info,
            }
        )
    return scenes


# ---------------------------------------------------------------------------
# 单景处理
# ---------------------------------------------------------------------------
METADATA_FIELDS: tuple[str, ...] = (
    "scene_id",
    "sensor",
    "spacecraft",
    "path",
    "row",
    "acquisition_date",
    "scene_center_time",
    "cloud_cover",
    "cloud_cover_land",
    "sun_azimuth",
    "sun_elevation",
    "tier",
    "utm_zone",
    "reflectance_mult_b4",
    "reflectance_add_b4",
    "reflectance_mult_b5",
    "reflectance_add_b5",
    "b4_output",
    "b5_output",
    "status",
)


def preprocess_scene(
    scene: dict, output_dir: Path, boundary_gdf
) -> dict:
    """
    处理单景影像：读取 → 缩放 → 裁剪 → 写出。

    始终读取 MTL 用于元数据 CSV 输出；
    若输出文件已存在，跳过影像重处理（增量处理）。

    Returns
    -------
    dict : 该景的元数据记录，用于汇总 CSV
    """
    scene_id = scene["scene_id"]
    out_dir = output_dir / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_b4 = out_dir / f"{scene_id}_SR_B4_reflectance.tif"
    out_b5 = out_dir / f"{scene_id}_SR_B5_reflectance.tif"

    # MTL 始终读取（用于元数据 CSV）
    mtl = parse_mtl(scene["mtl_path"])

    if out_b4.exists() and out_b5.exists():
        status = "skipped"
        print(f"[SKIP] 已处理: {scene_id}")
    else:
        b4_arr, b4_profile = load_band_as_reflectance(
            scene["b4_path"],
            mtl["reflectance_mult_b4"],
            mtl["reflectance_add_b4"],
        )
        b5_arr, b5_profile = load_band_as_reflectance(
            scene["b5_path"],
            mtl["reflectance_mult_b5"],
            mtl["reflectance_add_b5"],
        )
        b4_arr, b4_profile = clip_to_boundary(b4_arr, b4_profile, boundary_gdf)
        b5_arr, b5_profile = clip_to_boundary(b5_arr, b5_profile, boundary_gdf)

        b4_profile.update(DEFAULT_PROFILE_UPDATE)
        b5_profile.update(DEFAULT_PROFILE_UPDATE)
        with rasterio.open(out_b4, "w", **b4_profile) as dst:
            dst.write(b4_arr.astype(np.float32), 1)
        with rasterio.open(out_b5, "w", **b5_profile) as dst:
            dst.write(b5_arr.astype(np.float32), 1)
        status = "processed"
        print(
            f"[OK] {scene_id} | shape={b4_arr.shape} | "
            f"cloud={mtl['cloud_cover']:.2f}%"
        )

    return {
        "scene_id": scene_id,
        "sensor": scene["sensor"],
        "spacecraft": mtl["spacecraft"],
        "path": scene["path"],
        "row": scene["row"],
        "acquisition_date": scene["acquisition_date"],
        "scene_center_time": mtl["scene_center_time"],
        "cloud_cover": mtl["cloud_cover"],
        "cloud_cover_land": mtl["cloud_cover_land"],
        "sun_azimuth": mtl["sun_azimuth"],
        "sun_elevation": mtl["sun_elevation"],
        "tier": scene["tier"],
        "utm_zone": mtl["utm_zone"],
        "reflectance_mult_b4": mtl["reflectance_mult_b4"],
        "reflectance_add_b4": mtl["reflectance_add_b4"],
        "reflectance_mult_b5": mtl["reflectance_mult_b5"],
        "reflectance_add_b5": mtl["reflectance_add_b5"],
        "b4_output": str(out_b4.relative_to(PROJECT_ROOT)),
        "b5_output": str(out_b5.relative_to(PROJECT_ROOT)),
        "status": status,
    }


# ---------------------------------------------------------------------------
# 元数据 CSV 写出
# ---------------------------------------------------------------------------
def write_metadata_csv(records: list[dict], csv_path: Path) -> None:
    """将所有景的元数据记录写出为 CSV。"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for r in records:
            # NaN 在 CSV 中以空字符串表示，避免 'nan' 污染
            row = {
                k: ("" if isinstance(v, float) and v != v else v)
                for k, v in r.items()
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    """主入口。返回 0 表示成功。"""
    # 加载 YAML 配置（若存在）
    params = load_params()
    raw_dir = Path(params.get("paths", {}).get("raw_dir", RAW_DIR))
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    processed_dir = Path(
        params.get("paths", {}).get("processed_dir", PROCESSED_DIR)
    )
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir

    print("=" * 70)
    print("Landsat 数据预处理 — Guiyang NDVI Time Series")
    print("=" * 70)
    print(f"原始数据目录: {raw_dir}")
    print(f"输出目录    : {processed_dir}")

    if not raw_dir.exists():
        print(f"[ERROR] 原始数据目录不存在: {raw_dir}")
        return 1

    # 边界文件
    boundary_path = resolve_boundary_path()
    if boundary_path is None:
        # 回退到 params.yaml 配置
        bp = params.get("paths", {}).get("boundary_path")
        if bp:
            boundary_path = PROJECT_ROOT / bp
    boundary_gdf = load_boundary_gdf(boundary_path)
    if boundary_gdf is None:
        print(f"[WARN] 未找到边界矢量，将跳过裁剪步骤")
        print(f"       预期位置: {PROCESSED_DIR}/guiyang_boundary.geojson 或 .shp")
    else:
        print(f"边界文件    : {boundary_path} (CRS={boundary_gdf.crs})")

    # 扫描所有配对
    scenes = find_scene_pairs(raw_dir)
    print(f"\n发现 {len(scenes)} 景配对影像 (SR_B4 + SR_B5 + MTL)")
    print("-" * 70)

    if not scenes:
        print("[WARN] 未发现任何可处理影像")
        return 0

    # 逐景处理
    records: list[dict] = []
    for scene in scenes:
        record = preprocess_scene(scene, processed_dir, boundary_gdf)
        records.append(record)

    # 写元数据 CSV
    metadata_csv = processed_dir / "scene_metadata.csv"
    write_metadata_csv(records, metadata_csv)
    print("-" * 70)
    print(f"[OK] 元数据汇总: {metadata_csv.relative_to(PROJECT_ROOT)}")

    # 简要统计
    n_processed = sum(1 for r in records if r["status"] == "processed")
    n_skipped = sum(1 for r in records if r["status"] == "skipped")
    print(f"\n完成: {n_processed} 景新处理, {n_skipped} 景跳过, "
          f"共 {len(records)} 景")
    return 0


if __name__ == "__main__":
    sys.exit(main())
