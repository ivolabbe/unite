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

## Testing Status
- Environment fixed (scipy import issue resolved with fresh install)
- Region computation verified: Lines ARE in computed regions
- **Outstanding issue**: Line models still appear tiny in fits (peak ~0.004 vs continuum ~0.6)
  - This affects both manual and automatic region modes
  - Fitted integrated fluxes are reasonable (~500-1500)
  - May be separate issue from manual regions bug

## Next Steps
User should test with their data to verify manual regions now work correctly for line fitting.
