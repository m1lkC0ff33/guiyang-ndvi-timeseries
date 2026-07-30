# Guiyang NDVI Time Series Analysis (2015–2025)

## Overview

This project analyzes vegetation dynamics in Guiyang, China, over a 10-year period using Landsat satellite imagery. The analysis uses the Normalized Difference Vegetation Index (NDVI) to assess changes in vegetation health and coverage between urban and suburban areas.

**Key findings**: [To be filled after analysis]

## Data

- **Source**: USGS EarthExplorer
- **Sensor**: Landsat 8 (2015–2021), Landsat 9 (2022–2025)
- **Level**: Level-2 (atmospherically corrected surface reflectance)
- **Temporal range**: 2015-01 to 2025-07
- **Spatial extent**: Guiyang administrative region
- **Cloud cover**: < 20% per scene
- **Acquisition**: 2–3 scenes per year (summer/autumn, vegetation growing season)

## Methodology

1. **Data acquisition**: Download Landsat Level-2 imagery from USGS EarthExplorer.
2. **Preprocessing**: Extract Red (Band 4) and NIR (Band 5) bands; mask to Guiyang boundary.
3. **NDVI calculation**: Compute NDVI = (NIR - Red) / (NIR + Red) for each scene.
4. **Time series extraction**: Sample NDVI values at urban and suburban sites; compute annual means.
5. **Statistical analysis**: Trend detection and comparison between regions.
6. **Visualization**: Generate time series plots and spatial distribution maps.

## Repository Structure

```
guiyang-ndvi-timeseries/
├── data/
│   ├── raw/                  # 原始 Landsat 影像（.TIF，被 .gitignore 排除）
│   │   └── *.TIF, *_MTL.txt   # SR_B4 (Red), SR_B5 (NIR), MTL 元数据
│   └── processed/            # 预处理产物
│       ├── <scene_id>/        #   每景一个子目录
│       │   ├── *_SR_B4_reflectance.tif   # 已缩放的反射率 (float32)
│       │   └── *_SR_B5_reflectance.tif
│       ├── guiyang_boundary.geojson      # 贵阳市边界矢量（用户提供）
│       └── scene_metadata.csv            # 所有景的元数据汇总
├── src/
│   ├── __init__.py
│   ├── config.py             # 路径与参数集中管理
│   ├── download.py           # 数据下载元数据记录（非自动下载工具）
│   ├── preprocess.py         # 数据读取与预处理（缩放、裁剪）
│   ├── ndvi_calc.py          # NDVI 计算
│   ├── timeseries.py         # 时间序列采样
│   ├── analyze.py            # 统计分析（趋势、显著性）
│   └── visualize.py          # 制图与可视化
├── config/
│   └── params.yaml           # 路径、参数 YAML 配置
├── notebooks/                # Jupyter Notebooks（开发探索用）
├── outputs/
│   ├── figures/              # 最终成果图（PNG/PDF）
│   ├── tables/               # 统计结果表格（CSV）
│   └── report.md             # 英文分析报告
├── tests/
│   └── test_preprocess.py    # preprocess 模块单元测试
├── requirements.txt
├── README.md
└── LICENSE
```

## Usage

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 数据放置

将 Landsat 原始影像放入 `data/raw/`，命名遵循 USGS 默认格式：

```
data/raw/LC08_L2SP_127041_20180607_20200831_02_T1_SR_B4.TIF
data/raw/LC08_L2SP_127041_20180607_20200831_02_T1_SR_B5.TIF
data/raw/LC08_L2SP_127041_20180607_20200831_02_T1_MTL.txt
```

### 3. 准备边界矢量

将贵阳市行政边界文件命名为 `guiyang_boundary.geojson`（或 `.shp`），放入 `data/processed/`。
推荐来源：[阿里 DataV 行政区划选择器](http://datav.aliyun.com/portal/school/atlas/area_selector)。

### 4. 运行预处理

```bash
python src/preprocess.py
```

预处理后 `data/processed/` 会包含：
- 每景的反射率浮点 GeoTIFF（已应用缩放、按边界裁剪）
- `scene_metadata.csv`：所有景的元数据汇总（传感器、日期、云量、缩放参数）

### 5. 后续步骤

```bash
python src/ndvi_calc.py      # 计算 NDVI
python src/timeseries.py     # 提取时间序列
python src/analyze.py        # 统计分析
python src/visualize.py      # 生成成果图
```

## License

MIT — see [LICENSE](LICENSE)
