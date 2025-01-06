# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial package scaffold (`pyproject.toml`, src layout, hatchling build).
- Shared exception hierarchy (`wavrail.errors`).
- Shared dataclasses and type aliases (`wavrail.types`).
- Bootstrap commit with the parent-provided pre-commit hook installed.

## [0.1.0] - 2026-09-25

### Added

- First tagged release. Establishes the public module layout and exposes
  the version constant `wavrail.__version__`.

[Unreleased]: https://github.com/StarWorkshop/wavrail/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/StarWorkshop/wavrail/releases/tag/v0.1.0