# Level 4 Modular Notebook - Usage Guide

## 🎯 Overview

The original `frb_isotropy_analysis.ipynb` executed the entire Level 4 pipeline in one massive cell:

```python
results = main_level4(run_sensitivity=True, save_tables=True, n_jobs=-1)
```

This meant:
- ❌ Every time you changed one parameter, you had to re-run **everything** (~45-60 minutes)
- ❌ If the code crashed halfway, you lost all progress
- ❌ Difficult to debug individual steps
- ❌ Hard to inspect intermediate results
- ❌ Could not cache computational results

---

## ✨ New Modular Approach: `Level_4_Modular.ipynb`

The new notebook breaks the pipeline into **10 independent, chainable steps**:

```
STEP 0: Imports (1 min)
   ↓
STEP 1: Load & Clean Data (< 1 min)
   ↓
STEP 2: Build Selection Functions (2-3 min)
   ↓
STEP 3: Intersurvey Analysis (1 min)
   ↓
STEP 4: Compute Observed 2pACF (2 min)
   ↓
STEP 5: Generate H0 Ensemble (15-20 min) ⚠️ EXPENSIVE
   ↓
STEP 6: Jackknife Error Estimation (10-15 min) ⚠️ EXPENSIVE
   ↓
STEP 7: Statistical Tests (< 1 min)
   ↓
STEP 8: Visualization (5 min)
   ↓
STEP 9: Sensitivity Analysis (60 min) ⚠️ OPTIONAL & EXPENSIVE
   ↓
STEP 10: Save Results (< 1 min)
```

---

## 🚀 Usage Scenarios

### Scenario 1: Quick Testing (5 minutes)

You want to test a new parameter without waiting.

**Run only:**
1. STEP 0 (imports)
2. STEP 1 (load data)
3. STEP 4 (compute 2pACF with your data)
4. Skip STEP 5 & 6 (expensive), or load cached results

**Result:** See w(θ) immediately without spending 25 minutes on mocks!

### Scenario 2: Full Analysis (45 minutes)

You want the complete Level 4 pipeline.

**Run all steps in order:**
- STEP 0 → STEP 1 → ... → STEP 10
- Skip STEP 9 (sensitivity) unless investigating parameter sensitivity

**Result:** Full statistical tests + jackknife errors in ~40-45 minutes

### Scenario 3: Inspection & Debugging

You ran STEP 5 (mocks) and want to inspect results.

```python
# In a new cell after STEP 5:
print("Ensemble statistics:")
print(f"Mean w: {np.mean(all_w_h0, axis=0)[:5]}...")
print(f"Shape: {all_w_h0.shape}")

# Plot histograms
plt.hist(all_w_h0[:, 10], bins=20)  # Distribution at bin 10
plt.show()
```

No need to re-run anything. Just add exploratory cells!

### Scenario 4: Parameter Sensitivity

You want to test different values of `SMOOTH_SIGMA`.

**Workflow:**
1. Run STEP 0-4 once (caches data/SF)
2. Modify STEP 2 to rebuild SFs with new smoothing:
   ```python
   sf_dict, ... = build_survey_selection_functions_improved(
       df_data,
       nside=NSIDE_SF,
       smooth_sigma=8.0,  # Changed from 5.0
   )
   ```
3. Re-run STEP 4 (fast, recomputes 2pACF)
4. Reuse STEP 5 & 6 results (cached in `all_w_h0`, `jackknife`)
5. Re-run STEP 7 (statistical tests in <1 sec)

**Advantage:** Test 5 different smoothing values in ~10 minutes instead of 5× 45 = 225 minutes!

---

## 📊 Computational Cost Breakdown

| Step | Time | Cost | Can Skip? | Cache? |
|------|------|------|-----------|--------|
| 0. Imports | <1 min | Trivial | No | N/A |
| 1. Load & Clean | <1 min | Trivial | No | Yes |
| 2. Selection Functions | 2-3 min | Low | No | Yes |
| 3. Intersurvey | 1 min | Low | Yes | Yes |
| 4. 2pACF | 2 min | Low | No | Yes |
| 5. H0 Ensemble | 15-20 min | **HIGH** | Conditional | Yes |
| 6. Jackknife | 10-15 min | **HIGH** | Conditional | Yes |
| 7. Statistics | <1 min | Trivial | No | N/A |
| 8. Plotting | 5 min | Low | No | N/A |
| 9. Sensitivity | 60 min | **VERY HIGH** | Yes | Yes |
| 10. Save | <1 min | Trivial | No | N/A |

**Total minimum:** ~30-40 minutes (excluding Sensitivity)
**Full analysis:** ~45-60 minutes

---

## 🔧 How to Cache & Reuse Results

### Saving results (after STEP 6):

```python
# Save to files
import pickle

# Option 1: Save to pickle (fast loading)
with open('h0_ensemble.pkl', 'wb') as f:
    pickle.dump(all_w_h0, f)
    
with open('jackknife.pkl', 'wb') as f:
    pickle.dump(jackknife, f)
```

### Loading results (SKIP STEP 5 & 6):

```python
# In a new notebook session, after STEP 4:

with open('h0_ensemble.pkl', 'rb') as f:
    all_w_h0 = pickle.load(f)
    
with open('jackknife.pkl', 'rb') as f:
    jackknife = pickle.load(f)

# Compute absolute from w
all_abs_h0 = np.array([get_absolute_sum(theta, w) for w in all_w_h0])

# Skip directly to STEP 7 (statistics)!
stats = compute_statistics(w_obs, abs_obs, all_w_h0, all_abs_h0)
```

---

## 📝 Best Practices

### ✅ DO:
- Run steps in order the first time
- Add exploratory cells between steps to inspect results
- Cache expensive steps (5 & 6) to disk
- Use separate cells for parameter tweaks
- Document which parameters produced each result

### ❌ DON'T:
- Jump to STEP 7 without running STEP 5 & 6 (missing H0 ensemble)
- Modify `df_data` or `df_raw` later (mucks up jackknife regions)
- Run STEP 9 (sensitivity) unless really needed—it takes 1 hour!
- Assume results carry over if you restart the kernel (they don't—rerun STEP 0)

---

## 🔍 Inspection Examples

### Example 1: Look at w(θ) in detail

```python
# After STEP 4, before STEP 5:

h0_mean_prelim = np.mean(all_w_h0, axis=0)  # Not yet computed
# This will error because all_w_h0 doesn't exist—expected!

# Instead, look at observed only:
for i in [0, 10, 20, 34]:
    print(f"Bin {i}: θ={theta[i]:.1f}°, w_obs={w_obs[i]:.6f}, abs_w={abs_obs[i]:.6f}")
```

### Example 2: Check survey weights

```python
# After STEP 2:

print("Survey dominance:")
for name, w in sorted(survey_weights.items(), key=lambda x: -x[1])[:3]:
    print(f"  {name}: {w*100:.1f}%")
```

### Example 3: Jackknife sample sizes

```python
# After STEP 6:

print(f"Min FRBs in jackknife sample: {jackknife.w_samples.shape}")
# This shows which regions contribute most to the error bars
```

---

## ⚡ Time-Saving Tips

### Tip 1: Parallel Speedup

Set `n_jobs` to match your CPU core count:
```python
# In STEP 5:
all_w_h0, all_abs_h0 = run_ensemble_mocks(
    df_data, ...,
    n_jobs=8,  # Use 8 cores (faster than -1 if you have 8)
)

# On 8-core machine: ~15-20 min
# On 16-core machine: ~8-10 min
# On 4-core machine: ~30 min
```

### Tip 2: Skip Sensitivity (Usually Not Needed)

STEP 9 sensitivity analysis takes 1 hour and rarely changes conclusions. Only run if:
- Results are borderline (σ ~ 2-3)
- You need to publish (requires robustness check)
- Investigating parameter dependence

```python
# Just comment out STEP 9, leave blank cell
# No harm to results
```

### Tip 3: Use Notebooks Effectively

Create a **master notebook** with annotations:
```markdown
# Analysis Session: May 15, 2026

## First Run: Check isotropy signal
- Modified SMOOTH_SIGMA=5.0
- Results: χ²/dof=2.1, σ=2.3
- **Conclusion:** Weak anisotropy signal

## Second Run: Vary SMOOTH_SIGMA
- SMOOTH_SIGMA=3.0 → χ²/dof=2.05, σ=2.2
- SMOOTH_SIGMA=8.0 → χ²/dof=2.15, σ=2.4
- **Conclusion:** Signal is robust
```

---

## 🆘 Troubleshooting

### Problem: "NameError: name 'df_data' not defined"
**Cause:** Restarted kernel; need to re-run STEP 1
**Solution:** Run STEP 1 again

### Problem: "STEP 5 is taking forever (>30 min)"
**Cause:** `n_jobs=-1` on slow machine
**Solution:** Reduce n_mocks:
```python
# In code4_completo.py, change:
N_MOCKS_PER_ENSEMBLE = 10  # was 20
# Then re-import and re-run STEP 5
```

### Problem: "all_w_h0 arrays don't match dimensions"
**Cause:** STEP 5 rerun with different parameters, old data still in memory
**Solution:** Clear variables and re-run from STEP 4:
```python
del all_w_h0, all_abs_h0  # Free old data
# Re-run STEP 5
```

### Problem: "Jackknife errors are huge"
**Cause:** Signal is unstable; depends on specific sky regions
**Solution:** 
1. Check `jackknife.w_samples` for outliers
2. Visualize which regions drive errors:
   ```python
   plt.plot(jackknife.w_err)
   plt.xlabel('Angular bin')
   plt.ylabel('Error bar size')
   ```
3. Investigate if signal is robust or artifact

---

## 📚 Reference: Step-by-Step Outputs

| Step | Produces | Variable Name | Shape/Type |
|------|----------|---------------|-----------|
| 0 | Constants imported | `BIN_TYPE`, `NSIDE_SF`, etc | Scalars |
| 1 | Cleaned catalog | `df_data` | DataFrame (2732, 3) |
| 2 | Selection functions | `sf_dict`, `survey_weights` | Dict of arrays |
| 3 | Survey overlaps | `corr_df`, `overlap_df` | DataFrames |
| 4 | 2pACF observed | `theta`, `w_obs`, `abs_obs` | (35,), (35,), (7,) |
| 5 | H0 ensemble | `all_w_h0`, `all_abs_h0` | (200, 35), (200, 7) |
| 6 | Jackknife errors | `jackknife` | JackknifeResult object |
| 7 | Statistical tests | `stats` | TestStatistics object |
| 8 | PNG plots | Files on disk | PNG images |
| 9 | Sensitivity table | `sensitivity_results` | DataFrame (12, 8) |
| 10 | CSV files | Files on disk | CSV tables |

---

## 🎓 Learning Progression

**Beginner:** Run all steps in order, inspect each output
**Intermediate:** Modify parameters, reuse cached results
**Advanced:** Write custom analysis cells, combine with external data

---

**Happy analyzing! 🚀**

*Questions? Check `CODE4_DETAILED_EXPLANATION.md` for scientific details.*
