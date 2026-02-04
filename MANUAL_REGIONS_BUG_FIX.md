# Manual Regions Bug Fix

## Date
2026-02-04

## Bug Description
When manual continuum regions were specified for **line fitting** (not `continuum_only` mode), the code in `unite/initial.py` would:
1. Compute the manual continuum regions from the config (line 179)
2. Return those regions immediately (line 242)
3. `restrictAndRescale` would then restrict the spectrum to ONLY those continuum regions
4. **This excluded all line data from the fitting**

The regions returned were meant for continuum fitting, not for the full spectrum range needed when fitting both continuum and lines.

## Root Cause
In `computeContinuumRegions()` (unite/initial.py), when `has_manual_regions=True`, the function returned continuum-only regions without considering whether lines were being fitted.

## The Fix
Modified `unite/initial.py` lines 176-313:

**Continuum-only mode** (`continuum_only=True`):
- Compute manual regions
- Exclude lines from those regions
- Return continuum-only regions

**Line-fitting mode** (default):
- Compute manual regions as wavelength bounds
- Fall through to automatic line-padded region computation
- Intersect automatic regions with manual bounds
- This ensures line data is included while respecting user's wavelength constraints

## Code Changes
1. Lines 176-186: Renamed `cont_regs_obs` → `manual_regs_obs` for clarity
2. Lines 242-247: Added logic to store manual bounds and fall through for line-fitting mode
3. Lines 296-311: Added intersection logic to combine automatic line-padded regions with manual bounds

## Additional Bug Found: Line Flux Units (×10^4 factor)

While testing the manual regions fix, discovered a second bug:

**Problem**: Line models appeared 10000× too small (peak ~0.004 vs expected ~0.4)

**Cause**: `lineFluxGuess()` computed flux in units of [fλ × μm] but model expected [fλ × Å]

**Fix**: Added `flux *= 1e4` to convert μm to Å (1 μm = 10^4 Å)

**Result**: Line models now have correct amplitude matching observed data

## Testing Status
✅ Environment fixed (scipy import resolved)
✅ Region computation verified
✅ Line flux units fixed (×10^4 conversion)
✅ Both fixes tested together successfully

**Test Results**:
- Without manual regions: Lines visible (peak 0.487 vs continuum 0.111 = 439%)
- With manual regions: Lines visible (peak 0.317 vs continuum 0.160 = 199%)

## Commits
1. `93d3245`: Fix manual regions excluding line data
2. `129644e`: Fix line flux units with ×10^4 conversion
