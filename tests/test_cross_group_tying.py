"""Tests for cross-group tying of TieRedshift / TieDispersion."""

import copy

import pytest

from unite.parameters import configToMatrices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line(wavelength: float, rel_strength=None):
    """Minimal line dict for configToMatrices."""
    return {'Wavelength': wavelength, 'RelStrength': rel_strength}


def _species(name: str, linetype: str, wavelengths: list[float]):
    """Single-species with untied lines."""
    return {
        'Name': name,
        'LineType': linetype,
        'Lines': [_line(w) for w in wavelengths],
    }


def _group(tie_z=True, tie_σ=True, species=None):
    """Minimal group dict."""
    return {
        'TieRedshift': tie_z,
        'TieDispersion': tie_σ,
        'Species': species or [],
    }


def _unique_z_indices(config):
    """Return sorted unique z indices assigned to all lines."""
    # Run configToMatrices and inspect the z coupling matrix (orig[1])
    (orig, _add, _oa), _lt = configToMatrices(copy.deepcopy(config))
    z_mat = orig[1]  # shape (n_unique_z, n_total_lines)
    # For each line (column), find which unique-z row is nonzero
    import jax.numpy as jnp
    dense = jnp.array(z_mat.todense())
    line_to_z = {}
    for col in range(dense.shape[1]):
        rows = jnp.where(dense[:, col])[0]
        assert len(rows) == 1, f"Line {col} maps to {len(rows)} z params"
        line_to_z[col] = int(rows[0])
    return line_to_z, dense.shape[0]


def _unique_σ_indices(config):
    """Return sorted unique σ indices assigned to all lines."""
    import jax.numpy as jnp
    (orig, _add, _oa), _lt = configToMatrices(copy.deepcopy(config))
    σ_mat = orig[2]  # shape (n_unique_σ, n_total_lines)
    dense = jnp.array(σ_mat.todense())
    line_to_σ = {}
    for col in range(dense.shape[1]):
        rows = jnp.where(dense[:, col])[0]
        assert len(rows) == 1, f"Line {col} maps to {len(rows)} σ params"
        line_to_σ[col] = int(rows[0])
    return line_to_σ, dense.shape[0]


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------

def _two_group_config(tie_z_A=True, tie_σ_A=True, tie_z_B=True, tie_σ_B=True):
    """Two groups (A, B) each with one species and one line."""
    return {
        'Groups': {
            'A': _group(tie_z=tie_z_A, tie_σ=tie_σ_A,
                        species=[_species('Ha', 'narrow', [6563.0])]),
            'B': _group(tie_z=tie_z_B, tie_σ=tie_σ_B,
                        species=[_species('Hb', 'narrow', [4861.0])]),
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineBoolTying:
    """Bool-only tying should work identically to the original code."""

    def test_tied_true(self):
        """TieRedshift=True, TieDispersion=True — each group gets 1 z and 1 σ."""
        config = _two_group_config()
        z_map, n_z = _unique_z_indices(config)
        σ_map, n_σ = _unique_σ_indices(config)
        # Two groups → 2 unique z and 2 unique σ
        assert n_z == 2
        assert n_σ == 2
        # Lines in different groups get different indices
        assert z_map[0] != z_map[1]
        assert σ_map[0] != σ_map[1]

    def test_untied_false_two_species(self):
        """TieRedshift=False with 2 species → 2 z per group."""
        config = {
            'Groups': {
                'A': _group(tie_z=False, tie_σ=True, species=[
                    _species('Ha', 'narrow', [6563.0]),
                    _species('Hb', 'narrow', [4861.0]),
                ]),
            }
        }
        z_map, n_z = _unique_z_indices(config)
        # Each species gets its own z → 2 unique
        assert n_z == 2
        assert z_map[0] != z_map[1]


class TestCrossGroupRedshift:
    """Cross-group tying for redshift via string reference."""

    def test_basic(self):
        """B ties redshift to A → both share one z."""
        config = _two_group_config(tie_z_B='A')
        z_map, n_z = _unique_z_indices(config)
        assert n_z == 1
        assert z_map[0] == z_map[1]

    def test_dispersion_independent(self):
        """When only redshift is cross-tied, dispersion stays independent."""
        config = _two_group_config(tie_z_B='A')
        σ_map, n_σ = _unique_σ_indices(config)
        assert n_σ == 2
        assert σ_map[0] != σ_map[1]


class TestCrossGroupDispersion:
    """Cross-group tying for dispersion via string reference."""

    def test_basic(self):
        """B ties dispersion to A → both share one σ."""
        config = _two_group_config(tie_σ_B='A')
        σ_map, n_σ = _unique_σ_indices(config)
        assert n_σ == 1
        assert σ_map[0] == σ_map[1]

    def test_redshift_independent(self):
        """When only dispersion is cross-tied, redshift stays independent."""
        config = _two_group_config(tie_σ_B='A')
        z_map, n_z = _unique_z_indices(config)
        assert n_z == 2
        assert z_map[0] != z_map[1]


class TestBothCrossTied:
    """Both redshift and dispersion cross-tied."""

    def test_both(self):
        config = _two_group_config(tie_z_B='A', tie_σ_B='A')
        z_map, n_z = _unique_z_indices(config)
        σ_map, n_σ = _unique_σ_indices(config)
        assert n_z == 1
        assert n_σ == 1
        assert z_map[0] == z_map[1]
        assert σ_map[0] == σ_map[1]


class TestMixed:
    """Mixed: cross-tie redshift, bool-tie dispersion."""

    def test_mixed(self):
        config = _two_group_config(tie_z_B='A', tie_σ_B=True)
        z_map, n_z = _unique_z_indices(config)
        σ_map, n_σ = _unique_σ_indices(config)
        # z shared
        assert n_z == 1
        assert z_map[0] == z_map[1]
        # σ independent
        assert n_σ == 2
        assert σ_map[0] != σ_map[1]


class TestTransitive:
    """Transitive tying: C→B→A should all share one z."""

    def test_chain(self):
        config = {
            'Groups': {
                'A': _group(tie_z=True, tie_σ=True,
                            species=[_species('Ha', 'narrow', [6563.0])]),
                'B': _group(tie_z='A', tie_σ=True,
                            species=[_species('Hb', 'narrow', [4861.0])]),
                'C': _group(tie_z='B', tie_σ=True,
                            species=[_species('Hg', 'narrow', [4341.0])]),
            }
        }
        z_map, n_z = _unique_z_indices(config)
        assert n_z == 1
        assert z_map[0] == z_map[1] == z_map[2]


class TestInvalidReference:
    """Referencing a non-existent or forward-declared group raises ValueError."""

    def test_nonexistent(self):
        config = _two_group_config(tie_z_B='nonexistent')
        with pytest.raises(ValueError, match="nonexistent"):
            configToMatrices(config)

    def test_forward_reference(self):
        """B references C which comes after B — should fail."""
        config = {
            'Groups': {
                'A': _group(tie_z=True, tie_σ=True,
                            species=[_species('Ha', 'narrow', [6563.0])]),
                'B': _group(tie_z='C', tie_σ=True,
                            species=[_species('Hb', 'narrow', [4861.0])]),
                'C': _group(tie_z=True, tie_σ=True,
                            species=[_species('Hg', 'narrow', [4341.0])]),
            }
        }
        with pytest.raises(ValueError, match="'C'"):
            configToMatrices(config)


class TestMultipleGroupsPartialTying:
    """Multiple groups where only some are cross-tied."""

    def test_three_groups_one_tied(self):
        """A, B, C — only C ties to A. Should have 2 unique z (A+C share, B separate)."""
        config = {
            'Groups': {
                'A': _group(tie_z=True, tie_σ=True,
                            species=[_species('Ha', 'narrow', [6563.0])]),
                'B': _group(tie_z=True, tie_σ=True,
                            species=[_species('Hb', 'narrow', [4861.0])]),
                'C': _group(tie_z='A', tie_σ=True,
                            species=[_species('Hg', 'narrow', [4341.0])]),
            }
        }
        z_map, n_z = _unique_z_indices(config)
        assert n_z == 2
        # A and C share
        assert z_map[0] == z_map[2]
        # B is different
        assert z_map[1] != z_map[0]

    def test_multi_species_cross_tied(self):
        """Group with multiple species, cross-tied to another group."""
        config = {
            'Groups': {
                'A': _group(tie_z=True, tie_σ=True,
                            species=[_species('Ha', 'narrow', [6563.0]),
                                     _species('Hb', 'narrow', [4861.0])]),
                'B': _group(tie_z='A', tie_σ=True,
                            species=[_species('Hg', 'narrow', [4341.0])]),
            }
        }
        z_map, n_z = _unique_z_indices(config)
        # A has 2 lines (tied within → 1 unique z), B ties to A → same z
        assert n_z == 1
        assert z_map[0] == z_map[1] == z_map[2]

    def test_untied_species_cross_tied(self):
        """Group A has TieRedshift=False (2 species), B ties to A.
        B should share A's *first* z index."""
        config = {
            'Groups': {
                'A': _group(tie_z=False, tie_σ=True,
                            species=[_species('Ha', 'narrow', [6563.0]),
                                     _species('Hb', 'narrow', [4861.0])]),
                'B': _group(tie_z='A', tie_σ=True,
                            species=[_species('Hg', 'narrow', [4341.0])]),
            }
        }
        z_map, n_z = _unique_z_indices(config)
        # A untied → 2 unique z within A; B ties to A → reuses A's starting z
        # A-line0 and B-line0 share, A-line1 is separate
        assert z_map[0] == z_map[2]  # Ha and Hg share
        assert z_map[0] != z_map[1]  # Ha and Hb differ
