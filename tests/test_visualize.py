"""
test_visualize.py — 可视化单元测试

验证绘图函数能正常运行并生成 PNG 文件。
使用合成数据，不依赖真实 GeoTIFF。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from visualize import (  # noqa: E402
    plot_timeseries_trend,
    plot_region_comparison,
    plot_trend_spatial,
    plot_ndvi_spatial_maps,
)


# ---------------------------------------------------------------------------
# 辅助：创建测试用 NDVI GeoTIFF
# ---------------------------------------------------------------------------
def _create_test_ndvi(path: Path, arr: np.ndarray, crs: str = "EPSG:32648"):
    """创建测试用 NDVI GeoTIFF"""
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
# plot_ndvi_spatial_maps 测试
# ---------------------------------------------------------------------------
class TestPlotNdviSpatialMaps:
    def test_basic_plot(self, tmp_path):
        """能生成 NDVI 空间分布图"""
        # 创建 2 个 NDVI 文件
        for i, year in enumerate(["2017", "2020"]):
            arr = np.full((50, 50), 0.3 + i * 0.1, dtype=np.float32)
            arr[10:40, 10:40] = 0.5 + i * 0.1
            sid = f"LC08_L2SP_127041_{year}0101_20200101_02_T1"
            _create_test_ndvi(tmp_path / f"{sid}_NDVI.tif", arr)

        output = tmp_path / "ndvi_maps.png"
        plot_ndvi_spatial_maps(tmp_path, None, output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_excludes_contaminated(self, tmp_path):
        """排除云污染场景"""
        # 有效场景
        sid_valid = "LC08_L2SP_127041_20170101_20200101_02_T1"
        arr = np.full((30, 30), 0.5, dtype=np.float32)
        _create_test_ndvi(tmp_path / f"{sid_valid}_NDVI.tif", arr)

        # 污染场景
        sid_contam = "LC08_L2SP_127041_20150101_20200101_02_T2"
        arr_contam = np.full((30, 30), 0.01, dtype=np.float32)
        _create_test_ndvi(tmp_path / f"{sid_contam}_NDVI.tif", arr_contam)

        quality = pd.DataFrame({
            "scene_id": [sid_valid, sid_contam],
            "quality_flag": ["valid", "cloud_contaminated"],
        })

        output = tmp_path / "ndvi_maps.png"
        plot_ndvi_spatial_maps(tmp_path, quality, output)

        assert output.exists()

    def test_no_files(self, tmp_path):
        """无 NDVI 文件时不报错"""
        output = tmp_path / "ndvi_maps.png"
        plot_ndvi_spatial_maps(tmp_path, None, output)
        # 不应生成文件
        assert not output.exists()


# ---------------------------------------------------------------------------
# plot_timeseries_trend 测试
# ---------------------------------------------------------------------------
class TestPlotTimeseriesTrend:
    def test_basic_plot(self, tmp_path):
        """能生成时间序列趋势图"""
        annual = pd.DataFrame({
            "point_id": ["P001", "P002", "P003", "P004"],
            "region": ["urban", "urban", "suburban", "suburban"],
            "x": [100.0, 100.5, 101.0, 101.5],
            "y": [26.0, 26.1, 26.2, 26.3],
            "2017": [0.35, 0.40, 0.45, 0.42],
            "2018": [0.38, 0.42, 0.47, 0.44],
            "2019": [0.40, 0.45, 0.50, 0.46],
            "2020": [0.42, 0.47, 0.52, 0.48],
        })

        output = tmp_path / "trend.png"
        plot_timeseries_trend(annual, None, output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_with_trend_line(self, tmp_path):
        """带趋势线"""
        annual = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "region": ["urban", "suburban"],
            "x": [100.0, 101.0],
            "y": [26.0, 27.0],
            "2017": [0.3, 0.5],
            "2018": [0.35, 0.52],
            "2019": [0.4, 0.54],
            "2020": [0.45, 0.56],
        })

        region_trends = pd.DataFrame({
            "region": ["urban", "suburban"],
            "linear_slope": [0.05, 0.02],
            "linear_intercept": [-100.0, -40.0],
            "linear_significant": [True, False],
        })

        output = tmp_path / "trend.png"
        plot_timeseries_trend(annual, region_trends, output)

        assert output.exists()

    def test_empty_data(self, tmp_path):
        """空数据不报错"""
        output = tmp_path / "trend.png"
        plot_timeseries_trend(pd.DataFrame(), None, output)
        assert not output.exists()


# ---------------------------------------------------------------------------
# plot_region_comparison 测试
# ---------------------------------------------------------------------------
class TestPlotRegionComparison:
    def test_basic_plot(self, tmp_path):
        """能生成区域对比箱线图"""
        np.random.seed(42)
        n_points = 10
        annual = pd.DataFrame({
            "point_id": [f"P{i:03d}" for i in range(n_points)],
            "region": ["urban"] * 5 + ["suburban"] * 5,
            "x": np.random.uniform(100, 101, n_points),
            "y": np.random.uniform(26, 27, n_points),
            "2017": np.random.uniform(0.3, 0.6, n_points),
            "2018": np.random.uniform(0.3, 0.6, n_points),
            "2019": np.random.uniform(0.3, 0.6, n_points),
        })

        output = tmp_path / "comparison.png"
        plot_region_comparison(annual, output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_empty_data(self, tmp_path):
        """空数据不报错"""
        output = tmp_path / "comparison.png"
        plot_region_comparison(pd.DataFrame(), output)
        assert not output.exists()


# ---------------------------------------------------------------------------
# plot_trend_spatial 测试
# ---------------------------------------------------------------------------
class TestPlotTrendSpatial:
    def test_basic_plot(self, tmp_path):
        """能生成趋势空间图"""
        trend = pd.DataFrame({
            "point_id": [f"P{i:03d}" for i in range(10)],
            "region": ["urban"] * 5 + ["suburban"] * 5,
            "x": np.linspace(500000, 510000, 10),
            "y": np.linspace(2900000, 2910000, 10),
            "linear_slope": np.linspace(-0.01, 0.02, 10),
            "linear_significant": [True, False] * 5,
        })

        output = tmp_path / "trend_map.png"
        plot_trend_spatial(trend, output)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_all_positive_trend(self, tmp_path):
        """所有点都正趋势"""
        trend = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "region": ["urban", "suburban"],
            "x": [500000.0, 501000.0],
            "y": [2900000.0, 2901000.0],
            "linear_slope": [0.01, 0.02],
            "linear_significant": [True, True],
        })

        output = tmp_path / "trend_map.png"
        plot_trend_spatial(trend, output)

        assert output.exists()

    def test_empty_data(self, tmp_path):
        """空数据不报错"""
        output = tmp_path / "trend_map.png"
        plot_trend_spatial(pd.DataFrame(), output)
        assert not output.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
