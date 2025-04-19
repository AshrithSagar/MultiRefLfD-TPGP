# Changelog

## [Unreleased]

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

[unreleased]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.0.2...HEAD
[v0.0.2]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/AshrithSagar/MultiRefLfD-TPGP/releases/tag/v0.0.1
