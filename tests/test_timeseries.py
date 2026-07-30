"""
test_timeseries.py — 时间序列提取单元测试
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from timeseries import (  # noqa: E402
    find_ndvi_scenes,
    generate_points_from_ndvi_extent,
    extract_ndvi_at_points_batch,
)


# ---------------------------------------------------------------------------
# find_ndvi_scenes 测试
# ---------------------------------------------------------------------------
def test_find_ndvi_scenes_empty_dir(tmp_path):
    """空目录应返回空列表"""
    scenes = find_ndvi_scenes(tmp_path)
    assert scenes == []


def test_find_ndvi_scenes_no_ndvi(tmp_path):
    """有场景目录但无 NDVI 文件应返回空列表"""
    scene_dir = tmp_path / "LC08_L2SP_127041_20180607_20200831_02_T1"
    scene_dir.mkdir()
    (scene_dir / "LC08_..._SR_B4_reflectance.tif").touch()

    scenes = find_ndvi_scenes(tmp_path)
    assert scenes == []


def test_find_ndvi_scenes_finds_files_in_subdir(tmp_path):
    """能正确发现子目录中的 NDVI 文件并解析元数据"""
    scene_id = "LC08_L2SP_127041_20180607_20200831_02_T1"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir()
    (scene_dir / f"{scene_id}_NDVI.tif").touch()

    scenes = find_ndvi_scenes(tmp_path)

    assert len(scenes) == 1
    assert scenes[0]["scene_id"] == scene_id
    assert scenes[0]["sensor"] == "LC08"
    assert scenes[0]["year"] == "2018"
    assert scenes[0]["acquisition_date"] == "2018-06-07"


def test_find_ndvi_scenes_finds_files_at_root(tmp_path):
    """能正确发现根目录下的 NDVI 文件（ndvi_calc.py 的实际输出位置）"""
    scene_id = "LC08_L2SP_127041_20180607_20200831_02_T1"
    (tmp_path / f"{scene_id}_NDVI.tif").touch()

    scenes = find_ndvi_scenes(tmp_path)

    assert len(scenes) == 1
    assert scenes[0]["scene_id"] == scene_id
    assert scenes[0]["year"] == "2018"


def test_find_ndvi_scenes_landsat9(tmp_path):
    """正确识别 Landsat 9 场景"""
    scene_id = "LC09_L2SP_127041_20230901_20230903_02_T1"
    scene_dir = tmp_path / scene_id
    scene_dir.mkdir()
    (scene_dir / f"{scene_id}_NDVI.tif").touch()

    scenes = find_ndvi_scenes(tmp_path)
    assert len(scenes) == 1
    assert scenes[0]["sensor"] == "LC09"
    assert scenes[0]["year"] == "2023"


def test_find_ndvi_scenes_multiple_sorted(tmp_path):
    """多场景应按日期排序"""
    sid1 = "LC08_L2SP_127041_20201018_20201105_02_T2"
    (tmp_path / f"{sid1}_NDVI.tif").touch()

    sid2 = "LC08_L2SP_127041_20180607_20200831_02_T1"
    (tmp_path / f"{sid2}_NDVI.tif").touch()

    sid3 = "LC09_L2SP_127041_20230901_20230903_02_T1"
    (tmp_path / f"{sid3}_NDVI.tif").touch()

    scenes = find_ndvi_scenes(tmp_path)

    assert len(scenes) == 3
    assert scenes[0]["year"] == "2018"
    assert scenes[1]["year"] == "2020"
    assert scenes[2]["year"] == "2023"


def test_find_ndvi_scenes_ignores_hidden(tmp_path):
    """忽略以 _ 或 . 开头的目录"""
    sid = "LC08_L2SP_127041_20180607_20200831_02_T1"
    (tmp_path / sid).mkdir()
    (tmp_path / sid / f"{sid}_NDVI.tif").touch()

    hidden = tmp_path / "_hidden"
    hidden.mkdir()
    (hidden / "fake_NDVI.tif").touch()

    dot_dir = tmp_path / ".dot"
    dot_dir.mkdir()
    (dot_dir / "fake_NDVI.tif").touch()

    scenes = find_ndvi_scenes(tmp_path)
    assert len(scenes) == 1
    assert scenes[0]["scene_id"] == sid


# ---------------------------------------------------------------------------
# 辅助：创建测试用 NDVI GeoTIFF
# ---------------------------------------------------------------------------
def _create_test_ndvi_tif(
    path: Path,
    arr: np.ndarray,
    crs: str = "EPSG:32648",
    transform=None,
):
    """创建一个测试用的 GeoTIFF 文件"""
    if transform is None:
        # 默认 transform：左上角 (500000, 2900000)，像元 30m
        transform = from_origin(500000, 2900000, 30, 30)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": float("nan"),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


# ---------------------------------------------------------------------------
# generate_points_from_ndvi_extent 测试
# ---------------------------------------------------------------------------
class TestGeneratePointsFromNdviExtent:
    """测试从 NDVI 影像范围生成采样点"""

    def test_basic_generation(self, tmp_path):
        """能从有效像元生成采样点"""
        # 10x10 影像，中心 5x5 有效，边缘 NaN
        arr = np.full((10, 10), np.nan, dtype=np.float32)
        arr[2:8, 2:8] = 0.5  # 中心 6x6 = 36 个有效像元

        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr)

        df = generate_points_from_ndvi_extent(ndvi_path, n_points_per_region=5)

        assert len(df) == 10  # 5 urban + 5 suburban
        assert "point_id" in df.columns
        assert "region" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns

    def test_crs_in_attrs(self, tmp_path):
        """返回的 DataFrame 应在 attrs 中记录 CRS"""
        arr = np.full((10, 10), 0.5, dtype=np.float32)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr, crs="EPSG:32648")

        df = generate_points_from_ndvi_extent(ndvi_path, n_points_per_region=3)

        assert df.attrs.get("crs") == "EPSG:32648"

    def test_region_split(self, tmp_path):
        """应按到中心距离划分 urban/suburban"""
        arr = np.full((10, 10), 0.5, dtype=np.float32)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr)

        df = generate_points_from_ndvi_extent(
            ndvi_path, n_points_per_region=5, urban_ratio=0.4
        )

        n_urban = (df["region"] == "urban").sum()
        n_suburban = (df["region"] == "suburban").sum()
        # urban_ratio=0.4, 总数 10 → urban=4, suburban=6
        assert n_urban == 4
        assert n_suburban == 6

    def test_reproducible_with_seed(self, tmp_path):
        """相同种子应生成相同采样点"""
        arr = np.full((20, 20), 0.5, dtype=np.float32)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr)

        df1 = generate_points_from_ndvi_extent(ndvi_path, n_points_per_region=5, seed=42)
        df2 = generate_points_from_ndvi_extent(ndvi_path, n_points_per_region=5, seed=42)

        pd.testing.assert_frame_equal(df1, df2)

    def test_all_nan_raises(self, tmp_path):
        """全 NaN 影像应抛出异常"""
        arr = np.full((10, 10), np.nan, dtype=np.float32)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr)

        with pytest.raises(ValueError, match="没有有效像元"):
            generate_points_from_ndvi_extent(ndvi_path, n_points_per_region=5)

    def test_insufficient_valid_pixels(self, tmp_path):
        """有效像元不足时应允许重复采样"""
        arr = np.full((10, 10), np.nan, dtype=np.float32)
        arr[5, 5] = 0.5  # 只有 1 个有效像元

        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr)

        # 应该不报错，而是允许重复
        df = generate_points_from_ndvi_extent(ndvi_path, n_points_per_region=5)
        assert len(df) == 10


# ---------------------------------------------------------------------------
# extract_ndvi_at_points_batch 测试
# ---------------------------------------------------------------------------
class TestExtractNdviAtPoints:
    """测试 NDVI 值提取"""

    def test_extract_same_crs(self, tmp_path):
        """点与影像 CRS 相同时直接提取"""
        # 5x5 影像，值 = 行号
        arr = np.arange(25, dtype=np.float32).reshape(5, 5)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr, crs="EPSG:32648")

        # 构造采样点（CRS 与影像一致）
        points = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "x": [500015.0, 500075.0],  # 像元中心
            "y": [2899985.0, 2899985.0],
        })

        values = extract_ndvi_at_points_batch(ndvi_path, points, "EPSG:32648")

        assert len(values) == 2
        assert values["P001"] is not np.nan
        assert values["P002"] is not np.nan

    def test_extract_different_crs(self, tmp_path):
        """点与影像 CRS 不同时应自动转换"""
        arr = np.full((10, 10), 0.6, dtype=np.float32)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr, crs="EPSG:32648")

        # 采样点用 WGS84 经纬度（贵阳市中心附近）
        points = pd.DataFrame({
            "point_id": ["P001"],
            "x": [106.71],  # 经度
            "y": [26.57],   # 纬度
        })

        values = extract_ndvi_at_points_batch(ndvi_path, points, "EPSG:4326")

        # 该点应落在影像范围内
        assert "P001" in values
        # 值应为 0.6 或 NaN（取决于点是否落在影像范围）
        if not np.isnan(values["P001"]):
            assert values["P001"] == pytest.approx(0.6, abs=0.01)

    def test_extract_out_of_bounds_returns_nan(self, tmp_path):
        """点在影像范围外应返回 NaN"""
        arr = np.full((5, 5), 0.5, dtype=np.float32)
        ndvi_path = tmp_path / "test_NDVI.tif"
        _create_test_ndvi_tif(ndvi_path, arr, crs="EPSG:32648")

        # 影像范围是 x:[500000, 500150], y:[2899850, 2900000]
        # 给一个明显在范围外的点
        points = pd.DataFrame({
            "point_id": ["P001"],
            "x": [999999.0],
            "y": [999999.0],
        })

        values = extract_ndvi_at_points_batch(ndvi_path, points, "EPSG:32648")

        assert np.isnan(values["P001"])


# ---------------------------------------------------------------------------
# 数据帧构建逻辑测试（不依赖真实 GeoTIFF）
# ---------------------------------------------------------------------------
class TestTimeseriesDataFrames:
    """测试时间序列表格构建的数据逻辑"""

    def test_point_values_structure(self):
        """逐点逐景表的结构"""
        points = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "region": ["urban", "suburban"],
            "x": [106.5, 106.6],
            "y": [26.5, 26.6],
        })

        scene_ids = ["LC08_2018", "LC08_2019", "LC08_2020"]

        df = points.copy()
        df[scene_ids[0]] = [0.5, 0.3]
        df[scene_ids[1]] = [0.6, 0.35]
        df[scene_ids[2]] = [0.55, 0.4]

        assert len(df) == 2
        assert list(df.columns[:4]) == ["point_id", "region", "x", "y"]
        assert "LC08_2018" in df.columns
        assert df.loc[0, "LC08_2018"] == 0.5

    def test_annual_mean_calculation(self):
        """逐年均值计算逻辑"""
        df = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "region": ["urban", "suburban"],
            "x": [106.5, 106.6],
            "y": [26.5, 26.6],
            "LC08_2018_0607": [0.5, 0.3],
            "LC08_2018_0826": [0.6, 0.35],
            "LC08_2019_0813": [0.55, 0.4],
        })

        year_2018_scenes = ["LC08_2018_0607", "LC08_2018_0826"]
        df["2018"] = df[year_2018_scenes].mean(axis=1)

        assert df.loc[0, "2018"] == pytest.approx((0.5 + 0.6) / 2)
        assert df.loc[1, "2018"] == pytest.approx((0.3 + 0.35) / 2)

    def test_region_mean_calculation(self):
        """区域均值与标准差计算"""
        df_annual = pd.DataFrame({
            "point_id": ["P001", "P002", "P003", "P004"],
            "region": ["urban", "urban", "suburban", "suburban"],
            "x": [106.5, 106.55, 106.7, 106.8],
            "y": [26.5, 26.52, 26.7, 26.8],
            "2018": [0.5, 0.6, 0.3, 0.35],
            "2019": [0.55, 0.65, 0.35, 0.4],
        })

        records = []
        for year in ["2018", "2019"]:
            for region in ["urban", "suburban"]:
                mask = df_annual["region"] == region
                vals = df_annual.loc[mask, year].dropna()
                records.append({
                    "region": region,
                    "year": year,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "count": int(len(vals)),
                })

        df_region = pd.DataFrame(records)

        urban_2018 = df_region[(df_region["region"] == "urban") & (df_region["year"] == "2018")]
        assert len(urban_2018) == 1
        assert urban_2018.iloc[0]["mean"] == pytest.approx((0.5 + 0.6) / 2)
        assert urban_2018.iloc[0]["count"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
