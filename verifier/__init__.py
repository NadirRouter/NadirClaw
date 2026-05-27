"""Reproducibility tooling for NadirClaw / Nadir verifier work.

This package ships standalone utilities used to make verifier and
classifier training reproducible — most importantly the
:mod:`contamination_audit` module, which lets anyone re-run the
benchmark-contamination check that backs the "no held-out leakage"
claims for Nadir's RouterBench / RouterArena results.

The utilities here import nothing from the rest of the NadirClaw
package; you can vendor them into another project verbatim.
"""

__all__ = ["contamination_audit"]
