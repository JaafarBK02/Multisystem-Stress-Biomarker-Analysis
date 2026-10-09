"""Preprocessing for latent profile analysis of multisystem stress biomarkers (MIDUS 2).

Modules:
    config      variable mappings and per-column missing-data rules
    cleaning    extraction, missing-data handling, derived scores
    transforms  Box-Cox / log transforms, z-scoring, outlier diagnostics
    pipeline    end-to-end run + command-line entry point
    synthetic   fake MIDUS-format data for demos and tests
"""

__version__ = "0.1.0"
