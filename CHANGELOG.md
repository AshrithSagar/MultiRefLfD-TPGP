# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.1.1] - 2025-05-03

### Added

- Jupyter notebooks used have now been made visible in the `notebooks` folder.

### Fixed

- `LocalPolicyGP` and `FrameRelevanceGP` classes now correctly address the pipeline implementation.

## [v0.1.0] - 2025-04-27

### Added

- Load frames for LASA dataset

### Changed

- Demonstration utils shifted to `demonstrations.py`

### Fixed

- Alignment parts now correctly computes `D0_star`
- Cleanup

## [v0.0.3] - 2025-04-25

### Added

- ManimCE animations

### Changed

- Migrated from `pip` to `uv` for package management.
- Replaced `black` with `ruff` for code formatting and linting.
- Switched from `setuptools` to `hatchling` for packaging.

## [v0.0.2] - 2025-04-19

### Added

- Dependencies: `pyro-ppl`, `torch`
- `set_seed` function for reproducibility.
- `prepare_data` function to preprocess demonstration data for GP training.
- `LocalPolicyGP`, `FrameRelevanceGP` classes

### Changed

- `resample` function handles progress interpolation and alignment.
- `load_data` function now with an optional `show_plot` parameter

### Fixed

- Fixed keypoint plotting issue by correcting index calculation for progress values.

## [v0.0.1] - 2025-04-16

### Added

- Initial release.

[unreleased]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.1.1...HEAD
[v0.1.1]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.0.3...v0.1.0
[v0.0.3]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.0.2...v0.0.3
[v0.0.2]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/releases/tag/v0.0.1
