# Feature Selection Summary

## Inputs
- CRSP source: `/Users/atharvavaidya/Desktop/Sem-3/DSCI/Code/exact/financial_loss_functions/data/raw/2023_sp_500_select_50/combined_predictors_raw.csv`
- Lags used: `[10, 30, 50, 60]`
- Non-lagged CRSP features: `8`
- Lagged CRSP features: `32`
- Macro features used: `102`
- Model rows after lag/macro merge and NaN drop: `3984`

## Macro Features Dropped
- Count dropped: `1`

| feature | reason |
|---|---|
| `SP500` | `configured_no_logic` |

## Correlation Signals (Spearman)
- Correlations are between non-lagged engineered CRSP features and lagged engineered CRSP features.
- Positive correlations found: `106`
- Negative correlations found: `150`
- Low correlations found: `192`

### Top Positive Correlations
| non_lagged_feature | lagged_feature | correlation |
|---|---|---:|
| `CUM_RET` | `CUM_RET_LAG_10` | `0.9972` |
| `LOG_CUM_RET` | `LOG_CUM_RET_LAG_10` | `0.9972` |
| `LOG_CUM_RET` | `CUM_RET_LAG_10` | `0.9972` |
| `CUM_RET` | `LOG_CUM_RET_LAG_10` | `0.9972` |
| `LOG_CUM_RET` | `LOG_CUM_RET_LAG_30` | `0.9927` |
| `LOG_CUM_RET` | `CUM_RET_LAG_30` | `0.9927` |
| `CUM_RET` | `LOG_CUM_RET_LAG_30` | `0.9927` |
| `CUM_RET` | `CUM_RET_LAG_30` | `0.9927` |
| `LOG_CUM_RET` | `LOG_CUM_RET_LAG_50` | `0.9912` |
| `LOG_CUM_RET` | `CUM_RET_LAG_50` | `0.9912` |
| `CUM_RET` | `LOG_CUM_RET_LAG_50` | `0.9912` |
| `CUM_RET` | `CUM_RET_LAG_50` | `0.9912` |
| `LOG_CUM_RET` | `LOG_CUM_RET_LAG_60` | `0.9906` |
| `LOG_CUM_RET` | `CUM_RET_LAG_60` | `0.9906` |
| `CUM_RET` | `LOG_CUM_RET_LAG_60` | `0.9906` |

### Top Negative Correlations
| non_lagged_feature | lagged_feature | correlation |
|---|---|---:|
| `LOG_CUM_RET` | `TURNOVER_LAG_10` | `-0.5724` |
| `CUM_RET` | `TURNOVER_LAG_10` | `-0.5724` |
| `CUM_RET` | `TURNOVER_LAG_60` | `-0.5677` |
| `LOG_CUM_RET` | `TURNOVER_LAG_60` | `-0.5677` |
| `TURNOVER` | `CUM_RET_LAG_10` | `-0.5676` |
| `TURNOVER` | `LOG_CUM_RET_LAG_10` | `-0.5676` |
| `LOG_CUM_RET` | `TURNOVER_LAG_50` | `-0.5670` |
| `CUM_RET` | `TURNOVER_LAG_50` | `-0.5670` |
| `LOG_CUM_RET` | `TURNOVER_LAG_30` | `-0.5646` |
| `CUM_RET` | `TURNOVER_LAG_30` | `-0.5646` |
| `TURNOVER` | `CUM_RET_LAG_30` | `-0.5574` |
| `TURNOVER` | `LOG_CUM_RET_LAG_30` | `-0.5574` |
| `TURNOVER` | `CUM_RET_LAG_50` | `-0.5472` |
| `TURNOVER` | `LOG_CUM_RET_LAG_50` | `-0.5472` |
| `TURNOVER` | `CUM_RET_LAG_60` | `-0.5412` |

### Lowest-Magnitude Correlations
| non_lagged_feature | lagged_feature | abs_correlation |
|---|---|---:|
| `BA_SPREAD` | `sprtrn_LAG_60` | `0.0001` |
| `RET` | `VOL_CHANGE_LAG_10` | `0.0002` |
| `sprtrn` | `sprtrn_LAG_50` | `0.0006` |
| `BA_SPREAD` | `VOL_CHANGE_LAG_10` | `0.0006` |
| `sprtrn` | `CUM_RET_LAG_60` | `0.0007` |
| `sprtrn` | `LOG_CUM_RET_LAG_60` | `0.0007` |
| `VOL_CHANGE` | `TURNOVER_LAG_30` | `0.0007` |
| `RET` | `ILLIQUIDITY_LAG_50` | `0.0009` |
| `RET` | `VOL_CHANGE_LAG_50` | `0.0010` |
| `BA_SPREAD` | `RET_LAG_30` | `0.0010` |
| `sprtrn` | `LOG_CUM_RET_LAG_10` | `0.0010` |
| `sprtrn` | `CUM_RET_LAG_10` | `0.0010` |
| `sprtrn` | `LOG_CUM_RET_LAG_50` | `0.0011` |
| `sprtrn` | `CUM_RET_LAG_50` | `0.0011` |
| `LOG_CUM_RET` | `RET_LAG_10` | `0.0012` |

## Model Metrics
| metric | value |
|---|---:|
| `r2` | `-0.289842` |
| `mae` | `0.009501` |

## Top Features by Combined Ranking
| feature | mean_rank |
|---|---:|
| `LOG_CUM_RET_LAG_10` | `16.00` |
| `LOG_CUM_RET_LAG_30` | `19.25` |
| `TURNOVER_LAG_10` | `25.25` |
| `LOG_CUM_RET_LAG_50` | `25.25` |
| `EXUSUK` | `29.33` |
| `BA_SPREAD_LAG_30` | `29.50` |
| `BA_SPREAD_LAG_60` | `29.75` |
| `CUM_RET_LAG_10` | `30.00` |
| `TURNOVER_LAG_30` | `30.50` |
| `EXJPUS` | `31.67` |
| `IPNCONGD` | `32.33` |
| `CUM_RET_LAG_30` | `35.00` |
| `TB3MS` | `36.67` |
| `ACOGNO` | `37.33` |
| `VOL_CHANGE_LAG_50` | `39.25` |
| `TB6MS` | `39.67` |
| `ILLIQUIDITY_LAG_30` | `41.50` |
| `BUSLOANS` | `41.67` |
| `LOG_CUM_RET_LAG_60` | `42.00` |
| `CUM_RET_LAG_50` | `43.75` |
| `sprtrn_LAG_60` | `43.75` |
| `sprtrn_LAG_10` | `44.00` |
| `CUM_RET_LAG_60` | `44.25` |
| `ILLIQUIDITY_LAG_10` | `44.75` |
| `VOL_CHANGE_LAG_10` | `45.25` |
| `INVEST` | `45.33` |
| `RET_LAG_60` | `45.50` |
| `REALLN` | `47.00` |
| `UNRATE` | `47.67` |
| `sprtrn_LAG_50` | `48.25` |
| `CUUR0000SAD` | `48.33` |
| `AMBSL` | `49.00` |
| `VOL_CHANGE_LAG_30` | `49.00` |
| `AMDMUO` | `49.33` |
| `UMCSENT` | `50.00` |
| `EXSZUS` | `50.33` |
| `GS1` | `52.33` |
| `RET_LAG_50` | `53.00` |
| `ILLIQUIDITY_LAG_60` | `53.75` |
| `AAA` | `54.33` |
| `CUSR0000SAC` | `56.67` |
| `TURNOVER_LAG_50` | `57.50` |
| `PPICRM` | `58.00` |
| `NONBORRES` | `58.00` |
| `CPITRNSL` | `58.00` |
| `TURNOVER_LAG_60` | `59.00` |
| `DTCOLNVHFNM` | `60.00` |
| `IPNMAT` | `60.33` |
| `PPICMM` | `60.33` |
| `BA_SPREAD_LAG_10` | `60.50` |