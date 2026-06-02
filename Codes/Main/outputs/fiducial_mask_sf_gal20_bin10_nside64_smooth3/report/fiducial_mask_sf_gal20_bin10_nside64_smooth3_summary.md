# FRB Isotropy Analysis Report

Consolidated output of the isotropy pipeline.

## Run Configuration

```text
run_tag                : fiducial_mask_sf_gal20_bin10_nside64_smooth3
catalog_path           : /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/SkyPosition.csv
catalog_size           : 2732
use_gal_mask           : True
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
chi2                     : 24.1226
chi2_red                 : 1.41897
p_chi2_analytic          : 0.116143
p_chi2_empirical         : 0.131868
p_chi2_empirical_floor   : 0.000999001
sigma_equiv_empirical    : 1.1176
hartlap_factor           : 0.980981
rms_normalized_deviation : 1.204
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd               : 24.0573
chi2_svd_red           : 1.41513
p_chi2_svd_analytic    : 0.117887
p_chi2_svd_empirical   : 0.132867
sigma_svd_empirical    : 1.11294
svd_modes_kept         : 17/18
svd_retained_condition : 40.5621
```

## Covariance Diagnostics

```text
mocks                : 1000
bins                 : 18
covariance_rank      : 18/18
covariance_condition : 890.574
effective_modes      : 4.7538
h0_w_shape           : (1000, 18)
h0_abs_shape         : (1000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.666667
w_ks_pvalue        : 0.000429259
w_ks_empirical_p   : 0.00999001
w_ad_stat          : 6.1787
w_ad_pvalue        : 0.00134339
w_ad_empirical_p   : 0.00899101
abs_ks_stat        : 0.333333
abs_ks_pvalue      : 0.730111
abs_ks_empirical_p : 0.793207
abs_ad_stat        : -0.568113
abs_ad_pvalue      : 0.25
abs_ad_empirical_p : 0.779221
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 0.04134
empirical_pvalue                : 0.537463
```

## Jackknife Summary

```text
n_regions        : 131
median_sigma_w   : 0.0134152
min_sigma_w      : 0.0103095
max_sigma_w      : 0.0922355
median_sigma_abs : 0.0182829
min_sigma_abs    : 0.01314
max_sigma_abs    : 0.061502
```

## Chi-square Bin Audit

```text
 bin_index  theta_deg     w_obs   h0_mean   h0_std     delta      pull  diagonal_chi2_contribution
         0        5.0 -0.010865 -0.000615 0.002852 -0.010250 -3.594498                   12.674684
        10      105.0 -0.004907 -0.000389 0.002370 -0.004518 -1.906205                    3.564512
         9       95.0 -0.003306 -0.000505 0.001990 -0.002801 -1.407595                    1.943642
         4       45.0  0.001555 -0.000362 0.001618  0.001917  1.184642                    1.376685
        16      165.0  0.006076 -0.000393 0.006027  0.006468  1.073200                    1.129852
         5       55.0 -0.002111 -0.000450 0.001689 -0.001661 -0.983668                    0.949201
         1       15.0  0.001416 -0.000304 0.001922  0.001719  0.894807                    0.785451
        11      115.0  0.001800 -0.000516 0.002681  0.002316  0.863703                    0.731795
         8       85.0  0.001233 -0.000365 0.001899  0.001599  0.841805                    0.695158
         6       65.0  0.000848 -0.000330 0.001680  0.001178  0.701408                    0.482617
         7       75.0  0.000779 -0.000408 0.001710  0.001186  0.693626                    0.471967
        12      125.0  0.001570 -0.000316 0.003119  0.001886  0.604718                    0.358729
```

## Chi-square SVD Mode Audit

```text
 mode_index  eigenvalue  relative_eigenvalue  kept_by_svd_cut  delta_projection  raw_chi2_contribution  svd_chi2_contribution  fractional_svd_chi2_contribution  dominant_theta_deg  dominant_loading
          6    0.000008             0.081930             True          0.010437              12.956018              12.956018                          0.538549                 5.0         -0.974069
          7    0.000007             0.069276             True          0.004246               2.536571               2.536571                          0.105439               115.0          0.714111
         13    0.000003             0.031475             True          0.002347               1.705057               1.705057                          0.070875                65.0          0.734367
         14    0.000003             0.029906             True          0.002175               1.541729               1.541729                          0.064086                55.0         -0.562416
          9    0.000004             0.043399             True          0.002429               1.324859               1.324859                          0.055071                95.0         -0.717623
          1    0.000037             0.362893             True         -0.006778               1.233658               1.233658                          0.051280               165.0         -0.986919
         11    0.000004             0.038747             True          0.002083               1.091358               1.091358                          0.045365                85.0          0.590273
         12    0.000004             0.035718             True         -0.001142               0.355546               0.355546                          0.014779                25.0         -0.568468
         10    0.000004             0.039434             True         -0.001158               0.331468               0.331468                          0.013778                15.0         -0.789809
          8    0.000005             0.053491             True          0.001153               0.242052               0.242052                          0.010061               105.0         -0.710109
          2    0.000023             0.229845             True          0.002355               0.235160               0.235160                          0.009775               155.0         -0.927734
         15    0.000003             0.027471             True          0.000739               0.193853               0.193853                          0.008058                45.0          0.658301
```

## Selection Function Validation

```text
                                     run_tag       survey   weight    peak_ra   peak_dec  active_coverage_pct  sky_fraction  entropy_bits
fiducial_mask_sf_gal20_bin10_nside64_smooth3        CHIME 0.934186 315.000000  89.269029            38.545736      0.385457     13.541732
fiducial_mask_sf_gal20_bin10_nside64_smooth3       FRBCAT 0.028154 326.953125 -40.228185            21.006266      0.210063     12.551346
fiducial_mask_sf_gal20_bin10_nside64_smooth3        CRAFT 0.012066 351.562500 -32.089951            15.055339      0.150553     11.974846
fiducial_mask_sf_gal20_bin10_nside64_smooth3       PARKES 0.008044  15.000000 -74.603436             6.868490      0.068685     10.619011
fiducial_mask_sf_gal20_bin10_nside64_smooth3        ALERT 0.005484 332.578125  15.713861             5.708822      0.057088     10.230918
fiducial_mask_sf_gal20_bin10_nside64_smooth3     MeerTRAP 0.002925  53.823529 -52.029727             5.741374      0.057414     10.465048
fiducial_mask_sf_gal20_bin10_nside64_smooth3         FAST 0.001828   9.843750  18.839405             3.470866      0.034709      9.593056
fiducial_mask_sf_gal20_bin10_nside64_smooth3   Pan-STARRS 0.001463   2.812500 -36.423574             3.479004      0.034790      9.517593
fiducial_mask_sf_gal20_bin10_nside64_smooth3        ATLAS 0.001097 308.671875 -22.024313             2.862549      0.028625      9.098121
fiducial_mask_sf_gal20_bin10_nside64_smooth3 VLA-realfast 0.000731  10.546875  41.810315             2.720133      0.027201      6.751249
fiducial_mask_sf_gal20_bin10_nside64_smooth3   GaiaAlerts 0.000731 308.671875 -22.024313             2.288818      0.022888      8.515225
fiducial_mask_sf_gal20_bin10_nside64_smooth3       UTMOST 0.000731 334.396552 -46.571847             2.162679      0.021627      8.323216
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
VLA-realfast     FRBCAT 100.000000
VLA-realfast      CHIME 100.000000
     Tianlai      CHIME 100.000000
       ALERT      CHIME 100.000000
      TRAPUM      CHIME 100.000000
  VLITE-Fast      CHIME 100.000000
       GBNCC      CHIME 100.000000
      MASTER      CHIME 100.000000
     ASAS-SN      CHIME 100.000000
       ATLAS GaiaAlerts  66.666667
  Pan-STARRS      CHIME  50.000000
```

## Saved Files

```text
sf_validation: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_sf_validation.csv
intersurvey_corr: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_intersurvey_corr.csv
intersurvey_overlap: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_intersurvey_overlap.csv
stats_summary: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_stats_summary.csv
jackknife_w_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_jackknife_w_errors.csv
jackknife_abs_errors: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_jackknife_abs_errors.csv
covariance_matrix: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_covariance_matrix.csv
chi2_bin_diagnostics: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_chi2_bin_diagnostics.csv
svd_mode_contributions: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/tables/fiducial_mask_sf_gal20_bin10_nside64_smooth3_svd_mode_contributions.csv
report: /home/brunowesley/projetos/FRB-isotropy-tests/Codes/Main/outputs/fiducial_mask_sf_gal20_bin10_nside64_smooth3/report/fiducial_mask_sf_gal20_bin10_nside64_smooth3_summary.md
```
