# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-07-19

### Added
- `BaseReducer` — sklearn-compatible base class (`BaseEstimator` + `TransformerMixin`) with unified fit/transform contract, 2-D/3-D input normalisation, shape validation, and `fit_time_`/`transform_time_` attributes
- `PAA` — Piecewise Aggregate Approximation (stateless)
- `PCA` — Principal Component Analysis (per-channel)
- `AE` — Dense Autoencoder with `verbose` and `tqdm` progress support
