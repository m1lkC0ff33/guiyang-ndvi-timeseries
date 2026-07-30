"""
test_analyze.py — 统计分析单元测试
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from analyze import (  # noqa: E402
    linear_trend,
    mann_kendall_test,
    flag_cloud_contaminated_scenes,
    recalculate_annual_means,
    analyze_per_point,
    analyze_per_region,
    _norm_cdf,
)


# ---------------------------------------------------------------------------
# _norm_cdf 测试
# ---------------------------------------------------------------------------
class TestNormCdf:
    def test_zero(self):
        """z=0 时 CDF=0.5"""
        assert _norm_cdf(0) == pytest.approx(0.5, abs=1e-6)

    def test_positive(self):
        """z=1.96 时 CDF≈0.975（95% 单尾）"""
        assert _norm_cdf(1.96) == pytest.approx(0.975, abs=0.001)

    def test_negative(self):
        """z=-1.96 时 CDF≈0.025"""
        assert _norm_cdf(-1.96) == pytest.approx(0.025, abs=0.001)

    def test_symmetry(self):
        """正态分布对称性: CDF(z) + CDF(-z) = 1"""
        for z in [0.5, 1.0, 1.5, 2.0, 2.5]:
            assert _norm_cdf(z) + _norm_cdf(-z) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# linear_trend 测试
# ---------------------------------------------------------------------------
class TestLinearTrend:
    def test_perfect_positive_trend(self):
        """完美正趋势：slope>0, R²=1, p≈0"""
        years = np.array([2015, 2016, 2017, 2018, 2019], dtype=float)
        values = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)

        result = linear_trend(years, values)

        assert result["slope"] == pytest.approx(0.1, abs=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)
        assert result["p_value"] < 0.01
        assert result["n"] == 5

    def test_perfect_negative_trend(self):
        """完美负趋势：slope<0"""
        years = np.array([2015, 2016, 2017, 2018, 2019], dtype=float)
        values = np.array([0.7, 0.6, 0.5, 0.4, 0.3], dtype=float)

        result = linear_trend(years, values)

        assert result["slope"] == pytest.approx(-0.1, abs=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_no_trend(self):
        """无趋势：slope≈0"""
        years = np.array([2015, 2016, 2017, 2018, 2019], dtype=float)
        values = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)

        result = linear_trend(years, values)

        assert result["slope"] == pytest.approx(0.0, abs=1e-6)
        assert result["p_value"] > 0.05  # 不显著

    def test_with_nan(self):
        """含 NaN 的序列"""
        years = np.array([2015, 2016, 2017, 2018, 2019], dtype=float)
        values = np.array([0.3, np.nan, 0.5, 0.6, 0.7], dtype=float)

        result = linear_trend(years, values)

        assert result["n"] == 4  # NaN 被过滤
        assert result["slope"] > 0

    def test_insufficient_data(self):
        """数据不足（<3 个有效点）"""
        years = np.array([2015, 2016], dtype=float)
        values = np.array([0.3, 0.5], dtype=float)

        result = linear_trend(years, values)

        assert math.isnan(result["slope"])
        assert result["n"] == 2

    def test_noisy_trend(self):
        """有噪声的正趋势"""
        years = np.array([2015, 2016, 2017, 2018, 2019, 2020], dtype=float)
        values = np.array([0.30, 0.38, 0.45, 0.43, 0.55, 0.58], dtype=float)

        result = linear_trend(years, values)

        assert result["slope"] > 0
        assert 0.7 < result["r_squared"] < 1.0


# ---------------------------------------------------------------------------
# mann_kendall_test 测试
# ---------------------------------------------------------------------------
class TestMannKendall:
    def test_increasing_trend(self):
        """显著上升趋势"""
        values = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)

        result = mann_kendall_test(values)

        assert result["s"] > 0
        assert result["z"] > 0
        assert result["p_value"] < 0.05
        assert result["trend"] == "increasing"
        assert result["tau"] > 0

    def test_decreasing_trend(self):
        """显著下降趋势"""
        values = np.array([0.7, 0.6, 0.5, 0.4, 0.3], dtype=float)

        result = mann_kendall_test(values)

        assert result["s"] < 0
        assert result["z"] < 0
        assert result["p_value"] < 0.05
        assert result["trend"] == "decreasing"

    def test_no_trend(self):
        """无趋势"""
        values = np.array([0.5, 0.48, 0.52, 0.49, 0.51], dtype=float)

        result = mann_kendall_test(values)

        assert result["p_value"] > 0.05
        assert result["trend"] == "no_trend"

    def test_with_nan(self):
        """含 NaN"""
        values = np.array([0.3, np.nan, 0.5, 0.6, 0.7], dtype=float)

        result = mann_kendall_test(values)

        assert result["n"] == 4
        assert result["s"] > 0

    def test_insufficient_data(self):
        """数据不足"""
        values = np.array([0.3, 0.5], dtype=float)

        result = mann_kendall_test(values)

        assert result["trend"] == "insufficient_data"

    def test_tau_range(self):
        """Kendall's tau 在 [-1, 1] 范围内"""
        values = np.array([0.3, 0.5, 0.2, 0.6, 0.4, 0.7], dtype=float)

        result = mann_kendall_test(values)

        assert -1 <= result["tau"] <= 1


# ---------------------------------------------------------------------------
# flag_cloud_contaminated_scenes 测试
# ---------------------------------------------------------------------------
class TestFlagCloudContaminated:
    def test_flags_t2_low_ndvi(self):
        """T2 + NDVI < 0.1 → cloud_contaminated"""
        scene_meta = pd.DataFrame({
            "scene_id": ["S1", "S2", "S3"],
            "tier": ["T1", "T2", "T2"],
        })
        ndvi_meta = pd.DataFrame({
            "scene_id": ["S1", "S2", "S3"],
            "mean": [0.45, 0.005, 0.42],
        })

        flags = flag_cloud_contaminated_scenes(scene_meta, ndvi_meta)

        assert len(flags) == 3
        assert flags.loc[flags["scene_id"] == "S1", "quality_flag"].iloc[0] == "valid"
        assert flags.loc[flags["scene_id"] == "S2", "quality_flag"].iloc[0] == "cloud_contaminated"
        assert flags.loc[flags["scene_id"] == "S3", "quality_flag"].iloc[0] == "valid"

    def test_t2_normal_ndvi_valid(self):
        """T2 但 NDVI 正常 → valid"""
        scene_meta = pd.DataFrame({
            "scene_id": ["S1"],
            "tier": ["T2"],
        })
        ndvi_meta = pd.DataFrame({
            "scene_id": ["S1"],
            "mean": [0.45],
        })

        flags = flag_cloud_contaminated_scenes(scene_meta, ndvi_meta)

        assert flags.iloc[0]["quality_flag"] == "valid"

    def test_empty_input(self):
        """空输入返回空 DataFrame"""
        flags = flag_cloud_contaminated_scenes(
            pd.DataFrame(), pd.DataFrame()
        )
        assert flags.empty


# ---------------------------------------------------------------------------
# recalculate_annual_means 测试
# ---------------------------------------------------------------------------
class TestRecalculateAnnualMeans:
    def test_excludes_contaminated(self):
        """排除污染场景后重新计算"""
        point_values = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "region": ["urban", "suburban"],
            "x": [100.0, 101.0],
            "y": [26.0, 27.0],
            "LC08_L2SP_127041_20150601_20200101_02_T1": [0.4, 0.5],  # 2015, valid
            "LC08_L2SP_127041_20160601_20200101_02_T2": [0.01, 0.02],  # 2016, contaminated
            "LC08_L2SP_127041_20170601_20200101_02_T1": [0.45, 0.55],  # 2017, valid
        })

        quality = pd.DataFrame({
            "scene_id": [
                "LC08_L2SP_127041_20150601_20200101_02_T1",
                "LC08_L2SP_127041_20160601_20200101_02_T2",
                "LC08_L2SP_127041_20170601_20200101_02_T1",
            ],
            "quality_flag": ["valid", "cloud_contaminated", "valid"],
        })

        annual = recalculate_annual_means(point_values, quality)

        assert "2015" in annual.columns
        assert "2016" in annual.columns
        assert "2017" in annual.columns
        # 2016 只有污染场景 → 全 NaN
        assert np.isnan(annual["2016"].iloc[0])
        # 2015 和 2017 有效
        assert annual["2015"].iloc[0] == pytest.approx(0.4)
        assert annual["2017"].iloc[0] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# analyze_per_point 测试
# ---------------------------------------------------------------------------
class TestAnalyzePerPoint:
    def test_basic_structure(self):
        """输出结构正确"""
        annual = pd.DataFrame({
            "point_id": ["P001", "P002"],
            "region": ["urban", "suburban"],
            "x": [100.0, 101.0],
            "y": [26.0, 27.0],
            "2015": [0.3, 0.4],
            "2016": [0.35, 0.42],
            "2017": [0.4, 0.44],
            "2018": [0.45, 0.46],
            "2019": [0.5, 0.48],
        })

        result = analyze_per_point(annual)

        assert len(result) == 2
        assert "linear_slope" in result.columns
        assert "linear_p_value" in result.columns
        assert "mk_trend" in result.columns
        # P001 上升趋势明显
        assert result.loc[result["point_id"] == "P001", "linear_slope"].iloc[0] > 0

    def test_increasing_trend_significant(self):
        """强上升趋势应显著"""
        annual = pd.DataFrame({
            "point_id": ["P001"],
            "region": ["urban"],
            "x": [100.0],
            "y": [26.0],
            "2015": [0.30],
            "2016": [0.35],
            "2017": [0.40],
            "2018": [0.45],
            "2019": [0.50],
        })

        result = analyze_per_point(annual)

        assert result.iloc[0]["linear_significant"] == True
        assert result.iloc[0]["mk_trend"] == "increasing"


# ---------------------------------------------------------------------------
# analyze_per_region 测试
# ---------------------------------------------------------------------------
class TestAnalyzePerRegion:
    def test_basic_structure(self):
        """区域趋势输出结构"""
        annual = pd.DataFrame({
            "point_id": ["P001", "P002", "P003"],
            "region": ["urban", "urban", "suburban"],
            "x": [100.0, 100.5, 101.0],
            "y": [26.0, 26.1, 27.0],
            "2015": [0.3, 0.32, 0.5],
            "2016": [0.35, 0.37, 0.52],
            "2017": [0.4, 0.42, 0.54],
            "2018": [0.45, 0.47, 0.56],
            "2019": [0.5, 0.52, 0.58],
        })

        result = analyze_per_region(annual)

        assert len(result) == 2
        assert "urban" in result["region"].values
        assert "suburban" in result["region"].values
        assert "linear_slope" in result.columns
        assert "mk_trend" in result.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
