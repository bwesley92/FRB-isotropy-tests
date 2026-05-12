# Level 4 FRB Isotropy Report

This report consolidates the main text outputs from the Level 4 run.

## Run Configuration

```text
catalog_path         : /home/brunowesley/projetos/FRB-isotropy-tests/FRB_catalogs/SkyPosition.csv
masked_catalog_size  : 2732
bin_type             : Linear
n_bins               : 17
bin_size_deg         : 10
n_rand_factor        : 20
n_mocks              : 2000
n_ensemble           : 20
n_mocks_per_ensemble : 100
nside_sf             : 64
smooth_sigma_deg     : 5
perturbation_scale   : 1
nside_jackknife      : 8
svd_eigenvalue_cut   : 0.01
```

## Full Covariance Diagnostic

```text
chi2                     : 32.2614
chi2_red                 : 2.01634
p_chi2_analytic          : 0.00924448
p_chi2_empirical         : 0.0264868
p_chi2_empirical_floor   : 0.00049975
p_chi2_at_mc_floor       : False
sigma_equiv_empirical    : 1.93514
hartlap_factor           : 0.990995
rms_normalized_deviation : 1.39488
```

## Primary Isotropy Statistic (SVD-Regularized)

```text
chi2_svd                   : 29.8118
chi2_svd_red               : 1.86324
p_chi2_svd_analytic        : 0.0190026
p_chi2_svd_empirical       : 0.0374813
p_chi2_svd_empirical_floor : 0.00049975
p_chi2_svd_at_mc_floor     : False
sigma_svd_empirical        : 1.78069
svd_modes_kept             : 16/17
svd_retained_condition     : 11.6392
```

## Covariance Diagnostics

```text
mocks                : 2000
bins                 : 17
covariance_rank      : 17/17
covariance_condition : 549.828
effective_modes      : 7.34596
h0_w_shape           : (2000, 17)
h0_abs_shape         : (2000, 9)
```

## Non-parametric Profile Tests

```text
w_ks_stat          : 0.529412
w_ks_pvalue        : 0.0155606
w_ks_empirical_p   : 0.533733
w_ad_stat          : 4.43487
w_ad_pvalue        : 0.00556529
w_ad_empirical_p   : 0.377811
abs_ks_stat        : 0.666667
abs_ks_pvalue      : 0.0335664
abs_ks_empirical_p : 0.155922
abs_ad_stat        : 2.00159
abs_ad_pvalue      : 0.0486591
abs_ad_empirical_p : 0.166917
```

## Absolute Anisotropy Amplitude

```text
observed_rms_absolute_amplitude : 0.00234466
empirical_pvalue                : 0.25987
```

## Jackknife Summary

```text
n_regions        : 366
median_sigma_w   : 0.0233091
min_sigma_w      : 0.0189188
max_sigma_w      : 0.0710457
median_sigma_abs : 0.0137226
min_sigma_abs    : 0.00784629
max_sigma_abs    : 0.028485
```

## Selection Function Validation

Top surveys by weight:

```text
      survey   weight    peak_ra   peak_dec  active_coverage_pct  entropy_bits
       CHIME 0.934186 225.000000  89.269029            42.844645     13.657882
      FRBCAT 0.028154 326.953125 -40.228185            39.599609     13.338460
       CRAFT 0.012066 351.562500 -32.089951            34.250895     13.054569
      PARKES 0.008044  76.500000 -67.934929            13.120524     11.619174
       ALERT 0.005484 332.578125  15.713861            13.720703     11.505026
    MeerTRAP 0.002925 327.000000 -79.018714            17.360433     11.880547
        FAST 0.001828   9.843750  15.094787             9.049479     10.871968
  Pan-STARRS 0.001463   2.812500 -36.423574            10.056559     10.973066
       ATLAS 0.001097 308.671875 -22.024313             7.023112     10.509451
      UTMOST 0.000731 334.396552 -46.571847             4.193115      9.699636
VLA-realfast 0.000731  10.546875  41.810315             1.383464      8.098118
  GaiaAlerts 0.000731 308.671875 -22.024313             4.909261      9.959093
```

## Intersurvey Correlations

Largest absolute off-diagonal Pearson correlations:

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

Largest off-diagonal overlap fractions, in percent:

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
summary: Level4_summary.csv
diagnostics: Level4_diagnostics.xlsx
```
