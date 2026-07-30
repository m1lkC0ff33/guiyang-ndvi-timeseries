"""
config.py — 集中管理路径、参数、常量。

所有其他脚本统一从此模块导入，便于维护与修改。
路径以项目根目录为基准自动推导，无需手动配置绝对路径。
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 项目根目录与主要目录
#   假设本文件位于 <PROJECT_ROOT>/src/config.py
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_DIR: Path = PROJECT_ROOT / "src"
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
TABLES_DIR: Path = OUTPUTS_DIR / "tables"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
PARAMS_YAML: Path = CONFIG_DIR / "params.yaml"

# ---------------------------------------------------------------------------
# 边界矢量（用户提供，存放于 data/processed/）
#   优先 .geojson，其次 .shp；若都不存在则裁剪步骤会被跳过
# ---------------------------------------------------------------------------
BOUNDARY_CANDIDATES: tuple[str, ...] = (
    "guiyang_boundary.geojson",
    "guiyang_boundary.shp",
    "guiyang.geojson",
    "guiyang.shp",
)


def resolve_boundary_path() -> Path | None:
    """在 data/processed/ 下查找边界矢量文件，返回首个匹配或 None。"""
    for name in BOUNDARY_CANDIDATES:
        p = PROCESSED_DIR / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Landsat 文件命名约定（Collection 2 Level-2 Surface Reflectance）
#   示例： LC08_L2SP_127041_20180607_20200831_02_T1_SR_B4.TIF
# ---------------------------------------------------------------------------
RED_BAND_FILE: str = "SR_B4"   # Band 4 (Red, 640-670 nm for L8 / 636-673 nm for L9)
NIR_BAND_FILE: str = "SR_B5"   # Band 5 (NIR, 850-880 nm for L8 / 851-879 nm for L9)
MTL_SUFFIX: str = "MTL"        # 元数据 .txt 文件后缀

# ---------------------------------------------------------------------------
# 反射率缩放参数（Landsat Collection 2 Level-2 SR 默认值）
#   实际使用时会从 MTL 读取覆盖这两个默认值
#   公式：SR = DN * MULT + ADD
# ---------------------------------------------------------------------------
DEFAULT_SR_MULT: float = 2.75e-05
DEFAULT_SR_ADD: float = -0.2
SR_FILL_DN: int = 0                  # DN=0 表示填充像元（fill）
SR_VALID_MIN: float = -0.2           # 合理反射率下限（与 SR_ADD 一致）
SR_VALID_MAX: float = 1.6            # 合理反射率上限

# ---------------------------------------------------------------------------
# NDVI 有效范围
# ---------------------------------------------------------------------------
NDVI_VALID_MIN: float = -1.0
NDVI_VALID_MAX: float = 1.0

# ---------------------------------------------------------------------------
# 时间范围
# ---------------------------------------------------------------------------
YEAR_START: int = 2015
YEAR_END: int = 2025
YEARS: list[int] = list(range(YEAR_START, YEAR_END + 1))

# ---------------------------------------------------------------------------
# 输出 GeoTIFF 默认编码参数
# ---------------------------------------------------------------------------
DEFAULT_PROFILE_UPDATE: dict = {
    "dtype": "float32",
    "nodata": float("nan"),
    "compress": "deflate",
    "tiled": True,
}


def load_params() -> dict:
    """
    从 params.yaml 读取可配置项。

    若文件不存在则返回空 dict，调用方需自行回退到默认值。
    """
    if PARAMS_YAML.exists():
        with open(PARAMS_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


__all__ = [
    "PROJECT_ROOT", "SRC_DIR", "CONFIG_DIR", "DATA_DIR",
    "RAW_DIR", "PROCESSED_DIR", "OUTPUTS_DIR", "FIGURES_DIR", "TABLES_DIR",
    "TESTS_DIR", "PARAMS_YAML", "resolve_boundary_path",
    "RED_BAND_FILE", "NIR_BAND_FILE", "MTL_SUFFIX",
    "DEFAULT_SR_MULT", "DEFAULT_SR_ADD", "SR_FILL_DN",
    "SR_VALID_MIN", "SR_VALID_MAX",
    "NDVI_VALID_MIN", "NDVI_VALID_MAX",
    "YEAR_START", "YEAR_END", "YEARS",
    "DEFAULT_PROFILE_UPDATE", "load_params",
]
