"""Command-line interface for wavrail.

The CLI is intentionally thin: each subcommand constructs the
relevant library objects and dispatches to a small
``run_<subcommand>`` function. Errors are mapped to exit codes
so that shell pipelines can react to failures.
"""

from wavrail.cli.main import (
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE_ERROR,
    build_parser,
    main,
)

__all__ = [
    "EXIT_OK",
    "EXIT_RUNTIME_ERROR",
    "EXIT_USAGE_ERROR",
    "build_parser",
    "main",
]
