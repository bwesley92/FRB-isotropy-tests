# FRB Isotropy Analysis Report

Consolidated output of the isotropy pipeline.

## Run Configuration

```text
run_tag                : pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3
catalog_path           : /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/SkyPosition.csv
catalog_size           : 4480
use_gal_mask           : False
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
chi2                     : 4.40401e+06
chi2_red                 : 259060
p_chi2_analytic          : 0
p_chi2_empirical         : 0.000999001
p_chi2_empirical_floor   : 0.000999001
sigma_equiv_empirical    : 3.09053
hartlap_factor           : 0.980981
rms_normalized_deviation : 512.861
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd               : 4.40401e+06
chi2_svd_red           : 259060
p_chi2_svd_analytic    : 0
p_chi2_svd_empirical   : 0.000999001
sigma_svd_empirical    : 3.09053
svd_modes_kept         : 17/18
svd_retained_condition : 12.346
```

## Covariance Diagnostics

```text
mocks                : 1000
bins                 : 18
covariance_rank      : 18/18
covariance_condition : 282.143
effective_modes      : 7.65269
h0_w_shape           : (1000, 18)
h0_abs_shape         : (1000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.555556
w_ks_pvalue        : 0.00666885
w_ks_empirical_p   : 0.286713
w_ad_stat          : 5.10584
w_ad_pvalue        : 0.00317897
w_ad_empirical_p   : 0.0829171
abs_ks_stat        : 1
abs_ks_pvalue      : 4.11353e-05
abs_ks_empirical_p : 0.000999001
abs_ad_stat        : 8.83538
abs_ad_pvalue      : 0.001
abs_ad_empirical_p : 0.000999001
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 14.6453
empirical_pvalue                : 0.000999001
```

## Jackknife Summary

```text
n_regions        : 178
median_sigma_w   : 0.0993879
min_sigma_w      : 0.0832866
max_sigma_w      : 0.838121
median_sigma_abs : 0.205853
min_sigma_abs    : 0.1414
max_sigma_abs    : 1.07015
```

## Chi-square Bin Audit

```text
 bin_index  theta_deg     w_obs   h0_mean   h0_std     delta        pull  diagonal_chi2_contribution
         3       35.0  1.129771 -0.000293 0.001416  1.130064  797.942336               624602.336110
         2       25.0  1.349546 -0.000248 0.001708  1.349793  790.175501               612502.278751
         0        5.0  2.923088 -0.000755 0.003850  2.923843  759.488483               565852.153625
         4       45.0  0.930274 -0.000266 0.001326  0.930539  701.910929               483308.682988
         1       15.0  1.431312 -0.000222 0.002189  1.431535  654.040238               419632.892938
        13      135.0 -0.709588 -0.000222 0.001288 -0.709366 -550.664169               297463.870824
        12      125.0 -0.641517 -0.000220 0.001192 -0.641296 -537.879064               283811.421647
        14      145.0 -0.745289 -0.000275 0.001495 -0.745014 -498.255098               243536.516473
        11      115.0 -0.564028 -0.000233 0.001168 -0.563795 -482.781261               228644.836640
         5       55.0  0.548206 -0.000171 0.001208  0.548377  453.939777               202142.237388
        15      155.0 -0.771507 -0.000267 0.001703 -0.771241 -452.958415               201269.168688
        10      105.0 -0.462538 -0.000216 0.001117 -0.462322 -414.007113               168141.993735
```

## Chi-square SVD Mode Audit

```text
 mode_index  eigenvalue  relative_eigenvalue  kept_by_svd_cut  delta_projection  raw_chi2_contribution  svd_chi2_contribution  fractional_svd_chi2_contribution  dominant_theta_deg  dominant_loading
          8    0.000002             0.124108             True          1.543468           1.282733e+06           1.282733e+06                          0.291265                45.0          0.677158
          3    0.000005             0.322060             True         -1.775078           6.537929e+05           6.537929e+05                          0.148454                15.0         -0.700187
          0    0.000015             1.000000             True         -2.852991           5.439294e+05           5.439294e+05                          0.123508                 5.0         -0.998144
          4    0.000003             0.212229             True          1.125018           3.985260e+05           3.985260e+05                          0.090492               155.0         -0.779193
          7    0.000002             0.150837             True          0.938873           3.905224e+05           3.905224e+05                          0.088674                35.0          0.864919
          6    0.000002             0.157591             True         -0.827098           2.900840e+05           2.900840e+05                          0.065868               145.0          0.810287
          5    0.000003             0.198351             True         -0.846295           2.412968e+05           2.412968e+05                          0.054790                25.0         -0.719938
         10    0.000002             0.107278             True         -0.513362           1.641643e+05           1.641643e+05                          0.037276                55.0         -0.598143
         12    0.000001             0.100858             True         -0.457453           1.386518e+05           1.386518e+05                          0.031483                65.0         -0.652038
         14    0.000001             0.091356             True          0.353762           9.154355e+04           9.154355e+04                          0.020786                95.0         -0.738838
          1    0.000014             0.952285             True          0.879339           5.426097e+04           5.426097e+04                          0.012321               175.0         -0.997115
         11    0.000002             0.105152             True         -0.277956           4.909918e+04           4.909918e+04                          0.011149               125.0          0.694021
```

## Saved Files

```text
sf_validation: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_sf_validation.csv
intersurvey_corr: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_intersurvey_corr.csv
intersurvey_overlap: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_intersurvey_overlap.csv
stats_summary: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_stats_summary.csv
jackknife_w_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_jackknife_w_errors.csv
jackknife_abs_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_jackknife_abs_errors.csv
covariance_matrix: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_covariance_matrix.csv
chi2_bin_diagnostics: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_chi2_bin_diagnostics.csv
svd_mode_contributions: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/tables/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_svd_mode_contributions.csv
report: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3/report/pureisotropy_nomask_nosf_gal20_bin10_nside64_smooth3_summary.md
```
