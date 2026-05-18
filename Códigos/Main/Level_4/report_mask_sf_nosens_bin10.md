# FRB Isotropy Analysis Report

Consolidated output of the isotropy pipeline.

## Run Configuration

```text
run_tag                : mask_sf_nosens_bin10
catalog_path           : /home/brunowesley/projetos/FRB-isotropy-tests/FRB_catalogs/SkyPosition.csv
catalog_size           : 2732
use_gal_mask           : 1
use_selection_function : 1
run_sensitivity        : 0
gal_cut_deg            : 20
bin_size_deg           : 10
n_bins                 : 17
n_rand_factor          : 20
n_mocks                : 1000
n_ensemble             : 20
n_mocks_per_ensemble   : 50
nside_sf               : 64
smooth_sigma_deg       : 3
perturbation_scale     : 1
nside_jackknife        : 4
svd_eigenvalue_cut     : 0.01
```

## Full Covariance Diagnostic

```text
chi2                     : 28.1942
chi2_red                 : 1.76214
p_chi2_analytic          : 0.029971
p_chi2_empirical         : 0.0529471
p_chi2_empirical_floor   : 0.000999001
sigma_equiv_empirical    : 1.61693
hartlap_factor           : 0.981982
rms_normalized_deviation : 1.25735
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd               : 25.2584
chi2_svd_red           : 1.57865
p_chi2_svd_analytic    : 0.0653981
p_chi2_svd_empirical   : 0.0809191
sigma_svd_empirical    : 1.39892
svd_modes_kept         : 16/17
svd_retained_condition : 40.079
```

## Covariance Diagnostics

```text
mocks                : 1000
bins                 : 17
covariance_rank      : 17/17
covariance_condition : 1034.42
effective_modes      : 4.22822
h0_w_shape           : (1000, 17)
h0_abs_shape         : (1000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.470588
w_ks_pvalue        : 0.0449529
w_ks_empirical_p   : 0.943057
w_ad_stat          : 3.45366
w_ad_pvalue        : 0.0130003
w_ad_empirical_p   : 0.894106
abs_ks_stat        : 0.444444
abs_ks_pvalue      : 0.351707
abs_ks_empirical_p : 0.598402
abs_ad_stat        : 0.174453
abs_ad_pvalue      : 0.25
abs_ad_empirical_p : 0.522478
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 0.0025038
empirical_pvalue                : 0.330669
```

## Jackknife Summary

```text
n_regions        : 131
median_sigma_w   : 0.0136659
min_sigma_w      : 0.010073
max_sigma_w      : 0.0860615
median_sigma_abs : 0.007624
min_sigma_abs    : 0.00534694
max_sigma_abs    : 0.0257612
```

## Chi-square Bin Audit

```text
 bin_index  theta_deg     w_obs   h0_mean   h0_std     delta      pull  diagonal_chi2_contribution
         0   6.667948 -0.011592 -0.000737 0.002627 -0.010855 -4.131860                   16.764659
         8  89.828558 -0.003601 -0.000380 0.001964 -0.003221 -1.639766                    2.640385
         3  37.164116  0.002177 -0.000423 0.001598  0.002600  1.626758                    2.598660
         6  68.870459  0.001476 -0.000313 0.001604  0.001789  1.115617                    1.222176
         9 100.355737  0.001805 -0.000328 0.002227  0.002133  0.957583                    0.900443
        14 153.205332  0.003633 -0.000303 0.004382  0.003935  0.898086                    0.792026
        11 121.557490 -0.002380 -0.000276 0.002822 -0.002104 -0.745656                    0.545985
        12 132.129861  0.001598 -0.000386 0.003335  0.001984  0.594857                    0.347479
         7  79.334326 -0.001305 -0.000415 0.001717 -0.000890 -0.518407                    0.263903
        16 173.035552  0.004347 -0.000583 0.009943  0.004930  0.495829                    0.241417
         4  47.673709 -0.001089 -0.000455 0.001600 -0.000634 -0.396452                    0.154342
        10 111.011515 -0.001199 -0.000497 0.002476 -0.000703 -0.283742                    0.079059
```

## Chi-square SVD Mode Audit

```text
 mode_index   eigenvalue  relative_eigenvalue  kept_by_svd_cut  delta_projection  raw_chi2_contribution  svd_chi2_contribution  fractional_svd_chi2_contribution  dominant_theta_deg  dominant_loading
          6 7.252549e-06             0.073441             True          0.010425              14.716122              14.716122                          0.582623            6.667948         -0.938589
         11 3.199766e-06             0.032401             True          0.003520               3.803456               3.803456                          0.150582           68.870459          0.741093
         16 9.546747e-08             0.000967            False          0.000534               2.935861               0.000000                          0.000000           37.164116         -0.363982
          8 4.993843e-06             0.050569             True          0.002643               1.374040               1.374040                          0.054399          100.355737          0.671988
          9 3.910053e-06             0.039594             True         -0.002282               1.308205               1.308205                          0.051793           89.828558          0.741446
          4 1.088940e-05             0.110268             True          0.003396               1.040297               1.040297                          0.041186          132.129861          0.809681
         12 3.164576e-06             0.032045             True          0.001793               0.997434               0.997434                          0.039489           47.673709         -0.583501
          2 1.887978e-05             0.191180             True          0.003965               0.817749               0.817749                          0.032375          153.205332          0.950711
         13 2.774683e-06             0.028097             True         -0.000998               0.352824               0.352824                          0.013969           37.164116         -0.563940
         10 3.599455e-06             0.036449             True          0.001009               0.277774               0.277774                          0.010997           16.219550         -0.840214
          7 5.761271e-06             0.058340             True          0.001209               0.249303               0.249303                          0.009870          111.011515          0.640616
          0 9.875380e-05             1.000000             True         -0.004880               0.236814               0.236814                          0.009376          173.035552         -0.995879
```

## Selection Function Validation

```text
             run_tag       survey   weight    peak_ra   peak_dec  active_coverage_pct  sky_fraction  entropy_bits
mask_sf_nosens_bin10        CHIME 0.934186 315.000000  89.269029            39.821370      0.398214     13.541732
mask_sf_nosens_bin10       FRBCAT 0.028154 326.953125 -40.228185            24.538167      0.245382     12.551346
mask_sf_nosens_bin10        CRAFT 0.012066 351.562500 -32.089951            18.023682      0.180237     11.974846
mask_sf_nosens_bin10       PARKES 0.008044  76.500000 -67.934929            13.120524      0.131205     11.619174
mask_sf_nosens_bin10        ALERT 0.005484 332.578125  15.713861            13.720703      0.137207     11.505026
mask_sf_nosens_bin10     MeerTRAP 0.002925 297.187500 -54.340912            29.962158      0.299622     12.960114
mask_sf_nosens_bin10         FAST 0.001828   9.843750  15.094787            20.174154      0.201742     12.054359
mask_sf_nosens_bin10   Pan-STARRS 0.001463   2.812500 -36.423574            22.399902      0.223999     12.236365
mask_sf_nosens_bin10        ATLAS 0.001097 308.671875 -22.024313            15.924072      0.159241     11.707566
mask_sf_nosens_bin10       UTMOST 0.000731 334.396552 -46.571847            10.249837      0.102498     10.988954
mask_sf_nosens_bin10 VLA-realfast 0.000731  10.546875  41.810315             3.322347      0.033223      9.374281
mask_sf_nosens_bin10   GaiaAlerts 0.000731 308.671875 -22.024313            11.417643      0.114176     11.203424
```

## Intersurvey Correlations

```text
  survey_1   survey_2     value
     ATLAS GaiaAlerts  0.816463
Pan-STARRS GaiaAlerts  0.353409
Pan-STARRS      ATLAS  0.288475
     CRAFT   MeerTRAP  0.060324
     CRAFT     FRBCAT  0.051176
    FRBCAT     PARKES  0.037015
  MeerTRAP     FRBCAT  0.035310
     CRAFT     PARKES  0.031017
     ALERT     FRBCAT  0.021857
   Tianlai      CHIME  0.018448
       ZTF      CHIME  0.018448
    FRBCAT      CHIME -0.011912
```

## Intersurvey Overlap

```text
    survey_1   survey_2      value
         ZTF      CHIME 100.000000
VLA-realfast      CHIME 100.000000
     ASAS-SN      CHIME 100.000000
     Tianlai      CHIME 100.000000
       ALERT      CHIME 100.000000
      TRAPUM      CHIME 100.000000
  VLITE-Fast      CHIME 100.000000
       GBNCC      CHIME 100.000000
      MASTER      CHIME 100.000000
VLA-realfast     FRBCAT 100.000000
       ATLAS GaiaAlerts  66.666667
  Pan-STARRS      CHIME  50.000000
```

## Saved Files

```text
sf_validation: mask_sf_nosens_bin10_sf_validation.csv
intersurvey_corr: mask_sf_nosens_bin10_intersurvey_corr.csv
intersurvey_overlap: mask_sf_nosens_bin10_intersurvey_overlap.csv
stats_summary: mask_sf_nosens_bin10_stats_summary.csv
jackknife_w_errors: mask_sf_nosens_bin10_jackknife_w_errors.csv
jackknife_abs_errors: mask_sf_nosens_bin10_jackknife_abs_errors.csv
covariance_matrix: mask_sf_nosens_bin10_covariance_matrix.csv
chi2_bin_diagnostics: mask_sf_nosens_bin10_chi2_bin_diagnostics.csv
svd_mode_contributions: mask_sf_nosens_bin10_svd_mode_contributions.csv
report: report_mask_sf_nosens_bin10.md
```
