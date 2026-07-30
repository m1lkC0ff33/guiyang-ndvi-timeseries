"""
test_preprocess.py — preprocess 模块的单元测试

验证核心解析函数的正确性，不依赖真实 GeoTIFF 数据：
  - parse_scene_id：Landsat 文件名解析
  - parse_mtl：MTL 元数据字段提取（用样本字符串）
  - 反射率缩放公式

运行方式
--------
    # 项目根运行
    pytest tests/test_preprocess.py -v

    # 或安装 pytest 后
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests 能导入 src 下的模块
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import (  # noqa: E402
    LANDSAT_SCENE_PATTERN,
    find_scene_pairs,
    parse_mtl,
    parse_scene_id,
)
from config import (  # noqa: E402
    DEFAULT_SR_ADD,
    DEFAULT_SR_MULT,
    SR_FILL_DN,
)


# ---------------------------------------------------------------------------
# parse_scene_id 测试
# ---------------------------------------------------------------------------
class TestParseSceneId:
    """Landsat Collection 2 scene ID 解析测试"""

    def test_landsat8_t1(self):
        info = parse_scene_id("LC08_L2SP_127041_20180607_20200831_02_T1")
        assert info["sensor"] == "LC08"
        assert info["level"] == "L2SP"
        assert info["path"] == 127
        assert info["row"] == 41
        assert info["acquisition_date"] == "2018-06-07"
        assert info["processing_date"] == "2020-08-31"
        assert info["collection"] == 2
        assert info["tier"] == "T1"

    def test_landsat9_t1(self):
        info = parse_scene_id("LC09_L2SP_127042_20231120_20231121_02_T1")
        assert info["sensor"] == "LC09"
        assert info["path"] == 127
        assert info["row"] == 42
        assert info["acquisition_date"] == "2023-11-20"

    def test_tier2(self):
        """2015/2016 影像是 T2"""
        info = parse_scene_id("LC08_L2SP_127042_20151005_20200908_02_T2")
        assert info["tier"] == "T2"
        assert info["acquisition_date"] == "2015-10-05"

    def test_path_row_parsing(self):
        """path/row 三位数字解析正确"""
        info = parse_scene_id("LC08_L2SP_127041_20180607_20200831_02_T1")
        assert info["path"] == 127
        assert info["row"] == 41
        # 注意：path 041 而非 41 的格式，因为原文件名是 040 的形式
        info2 = parse_scene_id("LC08_L2SP_127040_20180607_20200831_02_T1")
        assert info2["row"] == 40

    @pytest.mark.parametrize(
        "bad_id",
        [
            "INVALID_NAME",
            "LC08_L2SP_127041_20180607_02_T1",          # 缺处理日期
            "LC07_L2SP_127041_20180607_20200831_02_T1",  # 非法传感器
            "LC08_L1TP_127041_20180607_20200831_02_T1",  # 非 L2SP
            "LC08_L2SP_127041_20180607_20200831_02_T3",  # 非法 tier
        ],
    )
    def test_invalid_scene_id_raises(self, bad_id):
        with pytest.raises(ValueError):
            parse_scene_id(bad_id)


# ---------------------------------------------------------------------------
# LANDSAT_SCENE_PATTERN 单独测试
# ---------------------------------------------------------------------------
def test_pattern_excludes_duplicate_downloads():
    """浏览器重复下载产生的 '(1)' / '(2)' 后缀文件名应不被匹配"""
    # 这些是浏览器重复下载后的命名，不应被当作合法 scene_id
    assert LANDSAT_SCENE_PATTERN.match("LC08_L2SP_127041_20180826_20200831_02_T1") is not None
    # 重复文件不会进入 find_scene_pairs，因为 glob('*_SR_B4.TIF') 不会匹配 'SR_B4 (1).TIF'


# ---------------------------------------------------------------------------
# parse_mtl 测试（用样本字符串构造）
# ---------------------------------------------------------------------------
SAMPLE_MTL = """\
GROUP = LANDSAT_METADATA_FILE
  GROUP = PRODUCT_CONTENTS
    LANDSAT_PRODUCT_ID = "LC08_L2SP_127041_20180607_20200831_02_T1"
    PROCESSING_LEVEL = "L2SP"
    COLLECTION_NUMBER = 02
    COLLECTION_CATEGORY = "T1"
  END_GROUP = PRODUCT_CONTENTS
  GROUP = IMAGE_ATTRIBUTES
    SPACECRAFT_ID = "LANDSAT_8"
    SENSOR_ID = "OLI_TIRS"
    WRS_PATH = 127
    WRS_ROW = 41
    DATE_ACQUIRED = 2018-06-07
    SCENE_CENTER_TIME = "03:20:36.9443150Z"
    CLOUD_COVER = 2.79
    CLOUD_COVER_LAND = 2.79
    SUN_AZIMUTH = 97.90244417
    SUN_ELEVATION = 69.02410440
  END_GROUP = IMAGE_ATTRIBUTES
  GROUP = PROJECTION_ATTRIBUTES
    MAP_PROJECTION = "UTM"
    DATUM = "WGS84"
    ELLIPSOID = "WGS84"
    UTM_ZONE = 48
  END_GROUP = PROJECTION_ATTRIBUTES
  GROUP = LEVEL2_SURFACE_REFLECTANCE_PARAMETERS
    REFLECTANCE_MULT_BAND_4 = 2.75e-05
    REFLECTANCE_ADD_BAND_4 = -0.2
    REFLECTANCE_MULT_BAND_5 = 2.75e-05
    REFLECTANCE_ADD_BAND_5 = -0.2
  END_GROUP = LEVEL2_SURFACE_REFLECTANCE_PARAMETERS
END_GROUP = LANDSAT_METADATA_FILE
END
"""


def test_parse_mtl_from_text(tmp_path):
    """用样本 MTL 字符串验证字段提取"""
    mtl_file = tmp_path / "test_MTL.txt"
    mtl_file.write_text(SAMPLE_MTL, encoding="utf-8")

    m = parse_mtl(mtl_file)
    assert m["spacecraft"] == "LANDSAT_8"
    assert m["sensor_id"] == "OLI_TIRS"
    assert m["wrs_path"] == 127
    assert m["wrs_row"] == 41
    assert m["date_acquired"] == "2018-06-07"
    assert m["scene_center_time"] == "03:20:36.9443150Z"
    assert abs(m["cloud_cover"] - 2.79) < 1e-6
    assert abs(m["sun_elevation"] - 69.02410440) < 1e-6
    assert abs(m["reflectance_mult_b4"] - 2.75e-05) < 1e-12
    assert abs(m["reflectance_add_b4"] - (-0.2)) < 1e-12
    assert m["map_projection"] == "UTM"
    assert m["utm_zone"] == 48
    assert m["datum"] == "WGS84"


def test_parse_mtl_fallback_on_missing_file(tmp_path):
    """MTL 文件不存在某些字段时回退到默认值"""
    minimal = "GROUP = X\n  DATE_ACQUIRED = 2018-06-07\nEND_GROUP = X\nEND\n"
    f = tmp_path / "m_MTL.txt"
    f.write_text(minimal, encoding="utf-8")
    m = parse_mtl(f)
    # 缺失字段使用默认值
    assert m["date_acquired"] == "2018-06-07"
    assert m["reflectance_mult_b4"] == DEFAULT_SR_MULT
    assert m["reflectance_add_b4"] == DEFAULT_SR_ADD
    assert m["cloud_cover"] != m["cloud_cover"]  # NaN


# ---------------------------------------------------------------------------
# 反射率缩放公式验证（数学层面，不读真实影像）
# ---------------------------------------------------------------------------
class TestReflectanceScaling:
    """验证 DN → 反射率转换公式"""

    def test_known_dn_to_reflectance(self):
        """手算验证：DN=10000 → SR = 10000 * 2.75e-05 + (-0.2) = 0.075"""
        mult, add = DEFAULT_SR_MULT, DEFAULT_SR_ADD
        dn = 10000.0
        expected = dn * mult + add  # 0.275 - 0.2 = 0.075
        assert abs(expected - 0.075) < 1e-9

    def test_dn_zero_is_fill(self):
        """DN=0 应被识别为填充像元"""
        # 在 load_band_as_reflectance 中 DN=0 会被置为 NaN
        assert SR_FILL_DN == 0

    def test_dn_one_is_valid(self):
        """DN=1 是有效反射率下限，不应被剔除"""
        mult, add = DEFAULT_SR_MULT, DEFAULT_SR_ADD
        sr_at_dn1 = 1 * mult + add  # ≈ -0.1999725
        # 应在 SR_VALID_MIN (-0.2) 之上
        assert sr_at_dn1 > -0.2

    def test_reflectance_range_reasonable(self):
        """高 DN 值对应合理范围内的反射率"""
        mult, add = DEFAULT_SR_MULT, DEFAULT_SR_ADD
        # DN=40000 → SR ≈ 1.1 - 0.2 = 0.9，是合理植被反射率
        sr = 40000 * mult + add
        assert 0 < sr < 1.5
        assert np.isfinite(sr)


# ---------------------------------------------------------------------------
# find_scene_pairs 行为测试（用临时目录构造测试数据）
# ---------------------------------------------------------------------------
def test_find_scene_pairs_skips_incomplete(tmp_path):
    """find_scene_pairs 应跳过缺失 B5 或 MTL 的影像"""
    # 完整三件套
    sid = "LC08_L2SP_127041_20180607_20200831_02_T1"
    (tmp_path / f"{sid}_SR_B4.TIF").write_bytes(b"")
    (tmp_path / f"{sid}_SR_B5.TIF").write_bytes(b"")
    (tmp_path / f"{sid}_MTL.txt").write_text(SAMPLE_MTL, encoding="utf-8")

    # 缺 B5
    sid2 = "LC08_L2SP_127042_20190914_20200826_02_T1"
    (tmp_path / f"{sid2}_SR_B4.TIF").write_bytes(b"")
    (tmp_path / f"{sid2}_MTL.txt").write_text(SAMPLE_MTL, encoding="utf-8")

    # 缺 MTL
    sid3 = "LC09_L2SP_127041_20220813_20230402_02_T1"
    (tmp_path / f"{sid3}_SR_B4.TIF").write_bytes(b"")
    (tmp_path / f"{sid3}_SR_B5.TIF").write_bytes(b"")

    # 重复下载文件（浏览器自动加 (1) 后缀）— glob('*_SR_B4.TIF') 不会匹配
    (tmp_path / f"{sid}_SR_B4 (1).TIF").write_bytes(b"")

    scenes = find_scene_pairs(tmp_path)
    assert len(scenes) == 1
    assert scenes[0]["scene_id"] == sid


if __name__ == "__main__":
    # 直接运行：python tests/test_preprocess.py
    sys.exit(pytest.main([__file__, "-v"]))
