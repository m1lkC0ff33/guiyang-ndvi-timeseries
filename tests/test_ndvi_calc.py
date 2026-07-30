"""
test_ndvi_calc.py — NDVI 计算单元测试

验证核心计算逻辑，不依赖真实 GeoTIFF 数据。

运行方式
--------
    pytest tests/test_ndvi_calc.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from ndvi_calc import calculate_ndvi  # noqa: E402


# ---------------------------------------------------------------------------
# NDVI 核心计算测试
# ---------------------------------------------------------------------------
class TestCalculateNDVI:
    """calculate_ndvi 函数的单元测试"""

    def test_basic_calculation(self):
        """测试基本计算：NIR > Red 时 NDVI 为正"""
        # 5x5 数组，NIR 反射率 0.5，Red 反射率 0.1
        red = np.full((5, 5), 0.1, dtype=np.float32)
        nir = np.full((5, 5), 0.5, dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        # 理论值：(0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 ≈ 0.6667
        expected_ndvi = (0.5 - 0.1) / (0.5 + 0.1)
        assert np.allclose(ndvi, expected_ndvi, atol=1e-6)
        assert stats["mean"] == pytest.approx(expected_ndvi, abs=1e-6)
        assert stats["valid_count"] == 25
        assert stats["nan_count"] == 0

    def test_negative_ndvi_for_water(self):
        """测试水体：NIR < Red 时 NDVI 为负"""
        red = np.full((5, 5), 0.1, dtype=np.float32)
        nir = np.full((5, 5), 0.05, dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        expected = (0.05 - 0.1) / (0.05 + 0.1)  # ≈ -0.3333
        assert np.allclose(ndvi, expected, atol=1e-6)
        assert stats["mean"] == pytest.approx(expected, abs=1e-6)

    def test_zero_denominator_handled(self):
        """测试分母为 0 的情况：两个波段都是 0 时应置为 NaN"""
        red = np.zeros((5, 5), dtype=np.float32)
        nir = np.zeros((5, 5), dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        # 全部应为 NaN
        assert np.all(np.isnan(ndvi))
        assert stats["valid_count"] == 0
        assert stats["nan_count"] == 25

    def test_small_denominator_threshold(self):
        """测试小分母阈值：分母小于阈值时应置为 NaN"""
        # 分母 = 0.00005 < 0.0001
        red = np.full((5, 5), 0.00002, dtype=np.float32)
        nir = np.full((5, 5), 0.00003, dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir, denominator_min=0.0001)
        
        # 分母 0.00005 < 0.0001，应全部为 NaN
        assert np.all(np.isnan(ndvi))
        assert stats["valid_count"] == 0

    def test_nan_input_handled(self):
        """测试输入含 NaN 的情况"""
        red = np.array([[0.1, np.nan, 0.1], [0.1, 0.1, 0.1]], dtype=np.float32)
        nir = np.array([[0.5, 0.5, np.nan], [0.5, 0.5, 0.5]], dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        # 2 个像元应为 NaN（red[0,1]=NaN, nir[0,2]=NaN），其余 4 个有效
        assert np.isnan(ndvi[0, 1])
        assert np.isnan(ndvi[0, 2])
        assert not np.isnan(ndvi[0, 0])  # 这个应有值
        assert stats["valid_count"] == 4
        assert stats["nan_count"] == 2

    def test_mixed_case(self):
        """测试混合情况：正常植被 + 水体 + 无效像元"""
        red = np.array([
            [0.1, 0.1, np.nan],   # 植被 / 植被 / 无效
            [0.1, 0.1, 0.0],     # 植被 / 植被 / 水体(分母=0)
        ], dtype=np.float32)
        nir = np.array([
            [0.5, 0.3, 0.5],     # 植被 / 植被(稀疏) / 无效
            [0.05, 0.5, 0.0],    # 水体 / 植被 / 水体(分母=0)
        ], dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        # 预期：
        # [0,0] = (0.5-0.1)/(0.5+0.1) = 0.6667
        # [0,1] = (0.3-0.1)/(0.3+0.1) = 0.5
        # [0,2] = NaN (red 为 NaN)
        # [1,0] = (0.05-0.1)/(0.05+0.1) = -0.3333
        # [1,1] = (0.5-0.1)/(0.5+0.1) = 0.6667
        # [1,2] = NaN (分母为 0)
        
        expected = np.array([
            [0.6667, 0.5000, np.nan],
            [-0.3333, 0.6667, np.nan],
        ], dtype=np.float32)
        
        assert np.allclose(ndvi, expected, atol=1e-3, equal_nan=True)
        assert stats["valid_count"] == 4
        assert stats["nan_count"] == 2

    def test_identical_bands_gives_zero(self):
        """测试 NIR = Red 时 NDVI = 0"""
        red = np.full((3, 3), 0.3, dtype=np.float32)
        nir = np.full((3, 3), 0.3, dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        # (0.3 - 0.3) / (0.3 + 0.3) = 0
        assert np.allclose(ndvi, 0.0, atol=1e-6)
        assert stats["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_stats_consistency(self):
        """测试统计信息的一致性"""
        # 混合 NDVI 值
        red = np.array([0.1, 0.2, 0.3, 0.1, 0.2], dtype=np.float32)
        nir = np.array([0.5, 0.6, 0.7, 0.3, 0.4], dtype=np.float32)
        
        ndvi, stats = calculate_ndvi(red, nir)
        
        # 手动计算预期值
        expected_ndvi = (nir - red) / (nir + red)
        assert np.allclose(ndvi, expected_ndvi, atol=1e-6)
        assert stats["mean"] == pytest.approx(np.mean(expected_ndvi), abs=1e-6)
        assert stats["min"] == pytest.approx(np.min(expected_ndvi), abs=1e-6)
        assert stats["max"] == pytest.approx(np.max(expected_ndvi), abs=1e-6)
        assert stats["std"] == pytest.approx(np.std(expected_ndvi), abs=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
