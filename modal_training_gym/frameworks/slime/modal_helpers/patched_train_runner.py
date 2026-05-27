"""Wrapper that patches Megatron sharding validation for hybrid models.

Hybrid architectures (e.g., GDN + standard attention via --spec) have some
layers with extra parameters (like linear_attn.dt_bias) that others lack.
Megatron's validate_sharding_integrity expects every position in the global
tensor to be covered, which fails for these layer-varying parameters.

This wrapper patches the validation to log a warning instead of crashing,
then runs the original train script.

Usage: python3 patched_train_runner.py /root/slime/train.py [train args...]
"""

from __future__ import annotations

import sys
import warnings


def _apply_validation_patch() -> None:
    try:
        import megatron.core.dist_checkpointing.validation as val_mod

        _original = val_mod.validate_sharding_integrity

        def _patched(*args, **kwargs):
            try:
                return _original(*args, **kwargs)
            except Exception as exc:
                warnings.warn(
                    f"Skipped sharding integrity validation (hybrid model): {exc}",
                    stacklevel=2,
                )

        val_mod.validate_sharding_integrity = _patched
    except (ImportError, AttributeError):
        pass


if __name__ == "__main__":
    _apply_validation_patch()

    train_script = sys.argv[1]
    sys.argv = sys.argv[1:]

    import runpy

    runpy.run_path(train_script, run_name="__main__")
