# FRB Isotropy Analysis Report

Consolidated output of the isotropy pipeline.

## Run Configuration

```text
run_tag                : pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3
catalog_path           : /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/SkyPosition.csv
catalog_size           : 2732
use_gal_mask           : True
use_selection_function : False
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
chi2                     : 1.61554e+06
chi2_red                 : 95031.9
p_chi2_analytic          : 0
p_chi2_empirical         : 0.000999001
p_chi2_empirical_floor   : 0.000999001
sigma_equiv_empirical    : 3.09053
hartlap_factor           : 0.980981
rms_normalized_deviation : 295.235
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd               : 1.61552e+06
chi2_svd_red           : 95030.5
p_chi2_svd_analytic    : 0
p_chi2_svd_empirical   : 0.000999001
sigma_svd_empirical    : 3.09053
svd_modes_kept         : 17/18
svd_retained_condition : 8.83658
```

## Covariance Diagnostics

```text
mocks                : 1000
bins                 : 18
covariance_rank      : 18/18
covariance_condition : 190.023
effective_modes      : 9.02025
h0_w_shape           : (1000, 18)
h0_abs_shape         : (1000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.555556
w_ks_pvalue        : 0.00666885
w_ks_empirical_p   : 0.36963
w_ad_stat          : 5.10584
w_ad_pvalue        : 0.00317897
w_ad_empirical_p   : 0.213786
abs_ks_stat        : 1
abs_ks_pvalue      : 4.11353e-05
abs_ks_empirical_p : 0.000999001
abs_ad_stat        : 8.83538
abs_ad_pvalue      : 0.001
abs_ad_empirical_p : 0.000999001
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 12.9655
empirical_pvalue                : 0.000999001
```

## Jackknife Summary

```text
n_regions        : 131
median_sigma_w   : 0.142637
min_sigma_w      : 0.102899
max_sigma_w      : 0.994604
median_sigma_abs : 0.28226
min_sigma_abs    : 0.204016
max_sigma_abs    : 1.51166
```

## Chi-square Bin Audit

```text
 bin_index  theta_deg     w_obs   h0_mean   h0_std     delta        pull  diagonal_chi2_contribution
         1       15.0  1.529770 -0.000294 0.003081  1.530064  496.674110               241993.462031
         2       25.0  1.142918 -0.000387 0.002455  1.143306  465.741690               212789.805664
         0        5.0  2.443800 -0.000774 0.005378  2.444573  454.557891               202693.111664
         3       35.0  0.920150 -0.000302 0.002339  0.920451  393.543503               151930.890285
         4       45.0  0.702761 -0.000430 0.002110  0.703191  333.202047               108912.044020
        11      115.0 -0.546169 -0.000324 0.001914 -0.545845 -285.206065                79795.445231
        14      145.0 -0.608638 -0.000369 0.002170 -0.608268 -280.311567                77080.163386
        13      135.0 -0.567461 -0.000446 0.002041 -0.567016 -277.838274                75725.950390
        12      125.0 -0.564718 -0.000341 0.002035 -0.564376 -277.332996                75450.769573
        15      155.0 -0.659868 -0.000438 0.002483 -0.659430 -265.569381                69185.740059
        10      105.0 -0.484295 -0.000374 0.001915 -0.483921 -252.667773                62626.810390
        16      165.0 -0.702369 -0.000365 0.002995 -0.702004 -234.359482                53879.759437
```

## Chi-square SVD Mode Audit

```text
 mode_index  eigenvalue  relative_eigenvalue  kept_by_svd_cut  delta_projection  raw_chi2_contribution  svd_chi2_contribution  fractional_svd_chi2_contribution  dominant_theta_deg  dominant_loading
         13    0.000004             0.130387             True         -1.049595          274821.197693          274821.197693                          0.170113                45.0         -0.581516
          2    0.000010             0.318704             True          1.609277          264310.387728          264310.387728                          0.163607                15.0          0.957363
          4    0.000007             0.228153             True         -1.174810          196765.761492          196765.761492                          0.121797                25.0         -0.690761
          0    0.000030             1.000000             True         -2.432639          192484.506120          192484.506120                          0.119147                 5.0         -0.824204
         11    0.000004             0.139176             True          0.865565          175096.248820          175096.248820                          0.108384               135.0         -0.529478
          8    0.000005             0.161180             True         -0.857664          148444.564574          148444.564574                          0.091887               135.0          0.505991
          7    0.000005             0.169145             True          0.849270          138698.790164          138698.790164                          0.085854               145.0         -0.873291
         16    0.000003             0.113166             True          0.468310           63036.383815           63036.383815                          0.039019                85.0         -0.524445
         14    0.000004             0.127933             True         -0.427704           46509.786362           46509.786362                          0.028789                75.0         -0.622726
         10    0.000004             0.144921             True          0.376251           31773.530398           31773.530398                          0.019668                65.0          0.698140
         12    0.000004             0.132949             True         -0.341657           28558.463434           28558.463434                          0.017678               105.0          0.763535
          1    0.000025             0.835274             True          0.609473           14465.059200           14465.059200                          0.008954               175.0          0.822626
```

## Saved Files

```text
sf_validation: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_sf_validation.csv
intersurvey_corr: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_intersurvey_corr.csv
intersurvey_overlap: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_intersurvey_overlap.csv
stats_summary: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_stats_summary.csv
jackknife_w_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_jackknife_w_errors.csv
jackknife_abs_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_jackknife_abs_errors.csv
covariance_matrix: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_covariance_matrix.csv
chi2_bin_diagnostics: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_chi2_bin_diagnostics.csv
svd_mode_contributions: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_svd_mode_contributions.csv
report: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3/report/pureisotropy_mask_nosf_gal20_bin10_nside64_smooth3_summary.md
```
