# Continuum Region Control

## Overview

Fine-grained control over which parts of the spectrum to use for continuum fitting.

## Key Changes

1. **Not validation**: Use the normal `NIRSpecFit` pipeline for real spectra (not `ValidationSuite`)
2. **Region specification**: Control include/exclude regions via config
3. **Flexible wavelength frames**: Specify regions in rest-frame or observed-frame
4. **Logical combination**: Manual regions combine with automatic line masking

## Config Format

```json
{
  "Unit": "Angstrom",
  "Name": "continuum_fit",
  "continuum_only": true,  // Required for continuum-only mode
  "continuum": {
    "type": "blackbody",
    "temperature": (4000, 10000),
    "regions": {
      "include": (3500, 7000),  // Single tuple or list of tuples
      "exclude": (4400, 4650),  // Single tuple or list of tuples
      "rest_frame": true        // true = rest frame, false = observed frame
    }
  }
}
```

## Region Logic

1. **Start** with `include` regions (or full spectrum if not specified)
2. **Subtract** automatic line exclusion zones (from `Groups`)
3. **Subtract** manual `exclude` regions

## Examples

### Example 1: Fit redward of Balmer break, exclude iron feature

```json
{
  "Unit": "Angstrom",
  "Name": "balmer_break",
  "continuum_only": true,
  "continuum": {
    "type": "blackbody",
    "regions": {
      "include": (3500, 7000),  // 3500-7000 Å rest frame
      "exclude": (4400, 4650),  // Exclude iron feature ~4570 Å
      "rest_frame": true
    }
  }
}
```

**Use case**: Fit PRISM spectra at z~8, focus on continuum redward of Balmer break

### Example 2: Multiple clean windows

```json
{
  "continuum": {
    "type": "blackbody",
    "regions": {
      "include": [(3500, 4500), (5500, 6500)],  // Two separate windows
      "exclude": [],
      "rest_frame": true
    }
  }
}
```

**Use case**: Avoid regions with complex line blending or telluric contamination

### Example 3: Observed-frame exclusion (tellurics)

```json
{
  "continuum": {
    "type": "blackbody",
    "regions": {
      "include": (3500, 7000),
      "exclude": [(13000, 14500), (18000, 20000)],  // Observed-frame in Å
      "rest_frame": false  // Wavelengths in observed frame
    }
  }
}
```

**Use case**: Exclude telluric absorption bands in observed frame

### Example 4: Continuum + emission lines

```json
{
  "Unit": "Angstrom",
  "Groups": {
    "narrow1": {
      "TieRedshift": true,
      "Species": [
        {"Name": "Ha", "Lines": [{"Wavelength": 6564.61, "LineType": "narrow"}]}
      ]
    }
  },
  "continuum": {
    "type": "blackbody",
    "regions": {
      "include": (3500, 7000),
      "exclude": (4400, 4650),
      "rest_frame": true
    }
  }
}
```

**Use case**: Fit emission lines AND continuum, with manual region control

## Implementation Details

### Unit Conversion

- `include`/`exclude` wavelengths are in `config['Unit']` (same as emission lines)
- Automatically converted to spectrum units (e.g., microns for PRISM)
- Redshift applied if `rest_frame=true`

### Region Processing

Implemented in `unite/initial.py::computeContinuumRegions()`:

```python
# Pseudocode
include_regions = manual_include or full_spectrum
line_exclusions = compute_from_emission_lines(config)
manual_exclusions = manual_exclude

final_regions = include_regions - line_exclusions - manual_exclusions
```

### Helper Functions

- `_normalize_region_input()`: Convert tuple or list to list of tuples
- `_merge_overlapping_regions()`: Merge adjacent/overlapping regions
- `_subtract_regions()`: Logical subtraction of regions

## Usage Example

See `examples/continuum_only_fitting.py` for a complete working example.

```python
import json
from astropy.table import Table
from unite.fitting import NIRSpecFit

# Setup
rows = Table({
    'root': ['source'],
    'srcid': [1],
    'file': ['spectrum.fits'],
    'spectra_directory': ['./spectra'],
    'grade': [1],
    'grating': ['PRISM'],
    'z': [7.7644],
    'zfit': [7.7644],
})

# Config with region control
config = {
    'Unit': 'Angstrom',
    'Name': 'continuum_fit',
    'continuum_only': True,  # Required for continuum-only mode
    'continuum': {
        'type': 'blackbody',
        'regions': {
            'include': (3500, 7000),
            'exclude': (4400, 4650),
            'rest_frame': True
        }
    }
}

# Save and run
with open('config.json', 'w') as f:
    json.dump(config, f)

fit_results = NIRSpecFit(
    rows=rows,
    config='config.json',
    output_dir='output',
    N=500,
    model_version='v2'
)
```

## Common Use Cases

### PRISM at z~8

```json
"regions": {
  "include": (3500, 7000),  // Redward of Balmer break
  "exclude": (4400, 4650),  // Iron feature
  "rest_frame": true
}
```

### G235M/G395M with telluric avoidance

```json
"regions": {
  "include": (3000, 8000),
  "exclude": [(13000, 14500), (18000, 20000)],  // Tellurics (obs frame)
  "rest_frame": false
}
```

### Multiple clean windows

```json
"regions": {
  "include": [(3500, 4300), (4700, 5400), (5800, 6500)],
  "exclude": [],
  "rest_frame": true
}
```

## Testing

Regions are computed before fitting. To inspect:

```python
from unite.spectra import NIRSpecSpectra
from unite.initial import computeContinuumRegions

spectra = NIRSpecSpectra(rows)
cont_regs, cont_guesses = computeContinuumRegions(config, spectra)

print(f"Number of regions: {len(cont_regs)}")
for i, reg in enumerate(cont_regs):
    print(f"Region {i+1}: {reg[0]:.3f} - {reg[1]:.3f} {spectra.λ_unit}")
```

## Backward Compatibility

- If `regions` not specified: uses full spectrum with automatic line masking (default behavior)
- Existing configs without `regions` continue to work unchanged
- `ValidationSuite` remains separate and unchanged
