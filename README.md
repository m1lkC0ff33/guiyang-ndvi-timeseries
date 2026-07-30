# Analysis Report: Vegetation Change in Guiyang City Based on Landsat NDVI Time Series (2015–2025)

## 1. Background and Objective

Guiyang, the capital of Guizhou Province in southwestern China, has experienced rapid urbanization over the past decade. Urban expansion often exerts pressure on surrounding vegetation ecosystems, making it essential to monitor vegetation dynamics quantitatively. The Normalized Difference Vegetation Index (NDVI), derived from red and near-infrared reflectance, is a widely adopted proxy for vegetation vigor and canopy density.

This study aims to: (1) construct a multi-year NDVI time series for Guiyang using Landsat 8/9 imagery; (2) quantify the inter-annual trend of vegetation cover from 2017 to 2025; and (3) compare vegetation trajectories between the urban core and suburban areas to assess whether urbanization has differentially affected vegetation.

## 2. Data and Methods

### 2.1 Data Source

A total of 19 Landsat Collection 2 Level-2 Surface Reflectance scenes were acquired from the USGS EarthExplorer platform, covering WRS-2 path/row 127/41 and 127/42 (2015-10-05 to 2025-05-09). The dataset comprises 14 Landsat 8 and 5 Landsat 9 scenes, with MTL-reported cloud cover ranging from 0.03% to 20.36%. Only the red (Band 4) and near-infrared (Band 5) bands were used.

### 2.2 Preprocessing

Raw digital numbers were converted to surface reflectance using per-scene scaling parameters parsed from the MTL metadata (`SR = DN × mult + add`). Fill pixels (DN = 0) and out-of-range values were masked as NaN. Scenes were optionally clipped to the Guiyang administrative boundary.

NDVI was computed as `(NIR − Red) / (NIR + Red)`, with a denominator threshold of 0.0001 to suppress unstable values over water bodies.

### 2.3 Quality Screening

Three Tier-2 scenes (2015, 2016, 2020) exhibited anomalously low NDVI means (−0.010 to +0.006), despite MTL cloud cover below 12%. Spectral diagnosis confirmed cloud contamination — the red and near-infrared reflectance were both elevated (~0.67) and nearly identical, a hallmark of cloud scattering. These scenes were flagged as `cloud_contaminated` and excluded from trend analysis, leaving 16 valid scenes spanning 2017–2025 (9 effective years).

### 2.4 Trend Analysis

A stratified sampling design placed 20 points (8 urban, 12 suburban) within the common valid extent of all scenes. For each point, annual NDVI means were computed.

Vegetation trends were assessed using two complementary methods:
- **Linear regression** (ordinary least squares) to estimate the rate of change (slope, units: NDVI/year) and its statistical significance (t-test, α = 0.05).
- **Mann-Kendall test**, a non-parametric monotonic trend test robust to non-normality and small samples.

## 3. Results

### 3.1 Overall Trend

Both regions exhibited a slight positive NDVI trend over 2017–2025, but neither was statistically significant:

| Region | Slope (/yr) | R² | p-value | Significant? | MK Trend | MK p-value |
|--------|-------------|------|---------|--------------|----------|------------|
| Urban | +0.00537 | 0.041 | 0.583 | No | no_trend | 0.754 |
| Suburban | +0.00610 | 0.051 | 0.539 | No | no_trend | 0.602 |

At the point level, 16 of 20 sampling points showed positive slopes (range: −0.014 to +0.029 /yr), but only 2 reached the p < 0.05 significance threshold.

### 3.2 Urban vs. Suburban Comparison

| Metric | Urban | Suburban | Difference |
|--------|-------|----------|------------|
| Mean NDVI | 0.447 | 0.384 | +0.063 |
| Trend slope (/yr) | +0.00537 | +0.00610 | −0.00074 |

The suburban area exhibited a marginally stronger greening rate than the urban core (+0.00074/yr), although the difference is small and not formally tested for significance. The urban core maintained a higher absolute NDVI (0.447 vs. 0.384).

### 3.3 Cloud Contamination Impact

The three excluded Tier-2 scenes had NDVI means near zero (−0.010 to +0.006), starkly lower than valid Tier-1 scenes (typically 0.35–0.53). Their inclusion would have severely biased the trend downward — underscoring the importance of spectral quality screening beyond metadata-based cloud cover filtering.

## 4. Conclusion

This study analyzed 16 valid Landsat scenes (2017–2025) to assess vegetation change in Guiyang. The key quantitative findings are:

1. **Slight, non-significant greening**: NDVI increased at approximately **+0.0054/yr** in the urban core and **+0.0061/yr** in suburban areas, but neither trend reached statistical significance (p > 0.53), and the Mann-Kendall test confirmed no monotonic trend.

2. **Suburban greening slightly outpaces urban**: the suburban slope exceeded the urban slope by **0.00074/yr**, suggesting vegetation recovery in suburban zones marginally outpaces urban vegetation change, though the difference is modest.

3. **Urban NDVI exceeds suburban**: the urban core maintained a **0.063 higher** mean NDVI than suburbs, likely reflecting established urban green infrastructure (parks, street trees) rather than natural vegetation.

4. **Cloud screening is essential**: three Tier-2 scenes with MTL cloud cover < 12% were spectrally diagnosed as cloud-contaminated; their exclusion was critical to avoid spurious downward trends.

In summary, Guiyang's vegetation cover remained relatively stable over the study period, with no statistically significant degradation or improvement. The slight positive tendency in both regions may reflect recent urban greening policies, but longer time series and additional explanatory variables (climate, land-use change) would be needed to confirm drivers.