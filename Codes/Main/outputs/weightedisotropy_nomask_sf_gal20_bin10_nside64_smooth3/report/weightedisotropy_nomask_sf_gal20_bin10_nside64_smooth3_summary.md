# FRB Isotropy Analysis Report

Consolidated output of the isotropy pipeline.

## Run Configuration

```text
run_tag                : weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3
catalog_path           : /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/SkyPosition.csv
catalog_size           : 4480
use_gal_mask           : False
use_selection_function : True
gal_cut_deg            : 20
bin_size_deg           : 10
n_bins                 : 18
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
chi2                     : 252.435
chi2_red                 : 14.8491
p_chi2_analytic          : 6.63775e-44
p_chi2_empirical         : 0.000999001
p_chi2_empirical_floor   : 0.000999001
sigma_equiv_empirical    : 3.09053
hartlap_factor           : 0.980981
rms_normalized_deviation : 3.82573
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd               : 252.367
chi2_svd_red           : 14.8451
p_chi2_svd_analytic    : 6.85362e-44
p_chi2_svd_empirical   : 0.000999001
sigma_svd_empirical    : 3.09053
svd_modes_kept         : 17/18
svd_retained_condition : 70.8585
```

## Covariance Diagnostics

```text
mocks                : 1000
bins                 : 18
covariance_rank      : 18/18
covariance_condition : 1444.66
effective_modes      : 3.55938
h0_w_shape           : (1000, 18)
h0_abs_shape         : (1000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.5
w_ks_pvalue        : 0.0207475
w_ks_empirical_p   : 0.79021
w_ad_stat          : 4.9128
w_ad_pvalue        : 0.00372844
w_ad_empirical_p   : 0.324675
abs_ks_stat        : 0.555556
abs_ks_pvalue      : 0.125874
abs_ks_empirical_p : 0.0879121
abs_ad_stat        : 3.30014
abs_ad_pvalue      : 0.014893
abs_ad_empirical_p : 0.001998
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 0.106626
empirical_pvalue                : 0.000999001
```

## Jackknife Summary

```text
n_regions        : 178
median_sigma_w   : 0.0232447
min_sigma_w      : 0.0136531
max_sigma_w      : 0.21
median_sigma_abs : 0.0295185
min_sigma_abs    : 0.0153717
max_sigma_abs    : 0.178673
```

## Chi-square Bin Audit

```text
 bin_index  theta_deg     w_obs   h0_mean   h0_std     delta      pull  diagonal_chi2_contribution
         3       35.0 -0.007830 -0.000269 0.000982 -0.007561 -7.701470                   58.184573
         0        5.0 -0.016634 -0.000358 0.002115 -0.016277 -7.695668                   58.096932
         2       25.0  0.006246 -0.000269 0.001117  0.006515  5.831792                   33.362967
         7       75.0  0.005011 -0.000169 0.001038  0.005180  4.989845                   24.425006
         4       45.0  0.004263 -0.000205 0.000920  0.004469  4.856952                   23.141328
         1       15.0  0.005445 -0.000206 0.001393  0.005651  4.058004                   16.154199
         5       55.0 -0.003928 -0.000247 0.000951 -0.003681 -3.871301                   14.701931
        16      165.0 -0.016239 -0.000167 0.004400 -0.016072 -3.652482                   13.086900
        17      175.0  0.026727  0.000063 0.008113  0.026664  3.286432                   10.595220
         9       95.0 -0.002930 -0.000221 0.001190 -0.002709 -2.275748                    5.080530
         8       85.0 -0.001958 -0.000280 0.001120 -0.001678 -1.498177                    2.201845
        10      105.0 -0.002229 -0.000175 0.001445 -0.002054 -1.421894                    1.983330
```

## Chi-square SVD Mode Audit

```text
 mode_index   eigenvalue  relative_eigenvalue  kept_by_svd_cut  delta_projection  raw_chi2_contribution  svd_chi2_contribution  fractional_svd_chi2_contribution  dominant_theta_deg  dominant_loading
          5 4.509496e-06             0.068849             True          0.017063              63.337167              63.337167                          0.250972                 5.0         -0.963731
         15 1.008924e-06             0.015404             True         -0.007183              50.173058              50.173058                          0.198810                55.0          0.673511
         12 1.321941e-06             0.020183             True          0.007306              39.615128              39.615128                          0.156974                75.0          0.723276
         11 1.354360e-06             0.020678             True         -0.006629              31.831288              31.831288                          0.126131                25.0         -0.826006
         13 1.121408e-06             0.017121             True         -0.003988              13.913321              13.913321                          0.055131                35.0          0.628775
         16 9.243477e-07             0.014113             True         -0.003592              13.693927              13.693927                          0.054262                45.0         -0.691070
          1 1.967501e-05             0.300391             True         -0.015680              12.258323              12.258323                          0.048573               165.0          0.982024
          0 6.549793e-05             1.000000             True          0.026404              10.441570              10.441570                          0.041375               175.0          0.999198
         10 1.579663e-06             0.024118             True          0.002869               5.113339               5.113339                          0.020262                95.0         -0.685456
         14 1.077766e-06             0.016455             True          0.002294               4.788691               4.788691                          0.018975                65.0          0.641081
          8 2.066475e-06             0.031550             True          0.002637               3.301733               3.301733                          0.013083               105.0         -0.621553
          4 4.895751e-06             0.074747             True         -0.002695               1.455433               1.455433                          0.005767               135.0          0.870537
```

## Selection Function Validation

```text
                                               run_tag       survey   weight    peak_ra   peak_dec  active_coverage_pct  sky_fraction  entropy_bits
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3        CHIME 0.912801 347.727273  48.922795            56.062826      0.560628     13.927658
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3       FRBCAT 0.029438  82.968750  33.510056            30.421956      0.304220     12.621748
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3        CRAFT 0.022971  93.515625   4.780192            19.193522      0.191935     10.788984
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3       PARKES 0.018064  15.000000 -74.603436            21.883138      0.218831     12.651411
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3        ALERT 0.005129 332.578125  15.713861             7.828776      0.078288     10.820560
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3     MeerTRAP 0.002676 258.962264 -50.480044             7.651774      0.076518     10.925947
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3         FAST 0.002453 296.718750  22.669610             6.172689      0.061727     10.587450
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3   Pan-STARRS 0.000892   2.812500 -36.423574             3.479004      0.034790      9.517876
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3        ATLAS 0.000892 283.359375  32.797168             3.481038      0.034810      9.517024
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3 VLA-realfast 0.000892  10.546875  41.810315             2.652995      0.026530      9.011445
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3   GaiaAlerts 0.000892 186.982759 -46.571847             3.503418      0.035034      9.516638
weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3       UTMOST 0.000892 104.741379 -46.571847             3.049723      0.030497      9.352149
```

## Intersurvey Correlations

```text
    survey_1   survey_2    value
      ALeRCE      ATLAS 0.499939
       ATLAS GaiaAlerts 0.499837
VLA-realfast     FRBCAT 0.318571
  Pan-STARRS GaiaAlerts 0.249756
  Pan-STARRS      ATLAS 0.249756
      FRBCAT      CHIME 0.169257
VLA-realfast      CHIME 0.151774
       ALERT      CHIME 0.046617
       CRAFT     FRBCAT 0.034131
    MeerTRAP     PARKES 0.028603
       CRAFT      CHIME 0.016208
      FRBCAT       FAST 0.009437
```

## Intersurvey Overlap

```text
    survey_1 survey_2  value
         ZTF    CHIME  100.0
      ALeRCE    ATLAS  100.0
      ALeRCE    CHIME  100.0
VLA-realfast    CHIME  100.0
     Tianlai    CHIME  100.0
       ALERT    CHIME  100.0
      TRAPUM    CHIME  100.0
       PALFA    CHIME  100.0
  VLITE-Fast    CHIME  100.0
       GBNCC    CHIME  100.0
      MASTER    CHIME  100.0
     ASAS-SN    CHIME  100.0
```

## Saved Files

```text
sf_validation: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_sf_validation.csv
intersurvey_corr: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_intersurvey_corr.csv
intersurvey_overlap: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_intersurvey_overlap.csv
stats_summary: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_stats_summary.csv
jackknife_w_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_jackknife_w_errors.csv
jackknife_abs_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_jackknife_abs_errors.csv
covariance_matrix: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_covariance_matrix.csv
chi2_bin_diagnostics: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_chi2_bin_diagnostics.csv
svd_mode_contributions: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/tables/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_svd_mode_contributions.csv
report: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3/report/weightedisotropy_nomask_sf_gal20_bin10_nside64_smooth3_summary.md
```
