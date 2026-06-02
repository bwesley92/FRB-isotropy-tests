# FRB Isotropy Analysis Report

Consolidated output of the isotropy pipeline.

## Run Configuration

```text
run_tag                : nomask_nosf_nosens_bin10
catalog_path           : /home/brunowesley/projetos/FRB-isotropy-tests/FRB_catalogs/Random_SkyPosition.csv
catalog_size           : 8446
use_gal_mask           : 0
use_selection_function : 0
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
chi2                     : 238696
chi2_red                 : 14918.5
p_chi2_analytic          : 0
p_chi2_empirical         : 0.000999001
p_chi2_empirical_floor   : 0.000999001
sigma_equiv_empirical    : 3.09053
hartlap_factor           : 0.981982
rms_normalized_deviation : 108.908
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd               : 206873
chi2_svd_red           : 12929.5
p_chi2_svd_analytic    : 0
p_chi2_svd_empirical   : 0.000999001
sigma_svd_empirical    : 3.09053
svd_modes_kept         : 16/17
svd_retained_condition : 7.38054
```

## Covariance Diagnostics

```text
mocks                : 1000
bins                 : 17
covariance_rank      : 17/17
covariance_condition : 160.526
effective_modes      : 9.02881
h0_w_shape           : (1000, 17)
h0_abs_shape         : (1000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.588235
w_ks_pvalue        : 0.00461056
w_ks_empirical_p   : 0.144855
w_ad_stat          : 4.64442
w_ad_pvalue        : 0.00466412
w_ad_empirical_p   : 0.160839
abs_ks_stat        : 1
abs_ks_pvalue      : 4.11353e-05
abs_ks_empirical_p : 0.000999001
abs_ad_stat        : 8.83538
abs_ad_pvalue      : 0.001
abs_ad_empirical_p : 0.000999001
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 0.115811
empirical_pvalue                : 0.000999001
```

## Jackknife Summary

```text
n_regions        : 192
median_sigma_w   : 0.00881193
min_sigma_w      : 0.00597758
max_sigma_w      : 0.159367
median_sigma_abs : 0.00421049
min_sigma_abs    : 0.00351298
max_sigma_abs    : 0.0392479
```

## Chi-square Bin Audit

```text
 bin_index  theta_deg     w_obs   h0_mean   h0_std     delta        pull  diagonal_chi2_contribution
         0   6.205871  0.476613 -0.000326 0.001621  0.476940  294.147343                84963.692584
         2  26.878958  0.137425 -0.000148 0.000790  0.137573  174.203819                29800.178501
         1  16.451833  0.152790 -0.000157 0.000975  0.152946  156.830402                24152.607986
         3  37.380913  0.090879 -0.000073 0.000677  0.090952  134.360700                17727.522061
         6  68.920659 -0.068608 -0.000110 0.000599 -0.068498 -114.261447                12820.440870
         7  79.472580 -0.067104 -0.000139 0.000601 -0.066964 -111.330136                12171.076758
         8  90.099838 -0.056425 -0.000102 0.000582 -0.056323  -96.816278                 9204.501869
         5  58.365070 -0.041177 -0.000116 0.000631 -0.041061  -65.105694                 4162.377560
        14 153.228226  0.020570 -0.000090 0.000777  0.020660   26.599793                  694.800343
         9 100.597015 -0.016217 -0.000135 0.000613 -0.016082  -26.232058                  675.722278
        12 132.264248  0.016063 -0.000145 0.000681  0.016208   23.816111                  556.987209
        13 142.728522  0.012665 -0.000106 0.000714  0.012771   17.894571                  314.446014
```

## Chi-square SVD Mode Audit

```text
 mode_index   eigenvalue  relative_eigenvalue  kept_by_svd_cut  delta_projection  raw_chi2_contribution  svd_chi2_contribution  fractional_svd_chi2_contribution  dominant_theta_deg  dominant_loading
          1 2.597312e-06             0.985446             True         -0.451461           77058.500015           77058.500015                          0.372492            6.205871         -0.968260
         16 1.641897e-08             0.006230            False         -0.023067           31822.946668               0.000000                          0.000000           90.099838         -0.321986
         14 3.683361e-07             0.139750             True          0.095856           24496.238757           24496.238757                          0.118412           79.472580         -0.678636
          4 6.695135e-07             0.254020             True          0.111704           18301.438024           18301.438024                          0.088467           26.878958          0.839914
          5 6.235952e-07             0.236598             True          0.102194           16445.611788           16445.611788                          0.079496          153.228226          0.816566
          3 9.352510e-07             0.354844             True          0.123538           16024.308974           16024.308974                          0.077460           16.451833          0.691970
          2 9.890618e-07             0.375260             True          0.107404           11453.020323           11453.020323                          0.055363          163.555117         -0.699118
         10 4.433336e-07             0.168205             True         -0.068940           10527.259900           10527.259900                          0.050888           79.472580         -0.507425
          6 5.500864e-07             0.208708             True         -0.064890            7516.732812            7516.732812                          0.036335          142.728522          0.856986
          9 4.513468e-07             0.171245             True         -0.055152            6617.804636            6617.804636                          0.031990          132.264248         -0.559149
         13 3.902996e-07             0.148084             True          0.049113            6068.851968            6068.851968                          0.029336           68.920659         -0.622168
          0 2.635671e-06             1.000000             True         -0.109283            4449.579700            4449.579700                          0.021509          173.013686         -0.966029
```

## Intersurvey Correlations

```text
Empty DataFrame
Columns: [survey_1, survey_2, value]
Index: []
```

## Intersurvey Overlap

```text
Empty DataFrame
Columns: [survey_1, survey_2, value]
Index: []
```

## Saved Files

```text
sf_validation: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/sf_validation.csv
intersurvey_corr: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/intersurvey_corr.csv
intersurvey_overlap: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/intersurvey_overlap.csv
stats_summary: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/stats_summary.csv
jackknife_w_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/jackknife_w_errors.csv
jackknife_abs_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/jackknife_abs_errors.csv
covariance_matrix: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/covariance_matrix.csv
chi2_bin_diagnostics: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/chi2_bin_diagnostics.csv
svd_mode_contributions: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/tables/svd_mode_contributions.csv
report: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Secondary_analysis/outputs/nomask_nosf_nosens_bin10/report/summary.md
```
