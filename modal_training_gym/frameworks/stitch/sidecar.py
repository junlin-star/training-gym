"""SGLang weight-sync sidecar entry point (slime decoder).

Thin adapter over :mod:`modal_training_gym.frameworks.stitch.sidecar_spine`: the
shared spine owns every knob; this only names slime's host-side delta decoder.
:func:`sidecar_process.start_sglang_sidecar` launches it via
``python3 -m modal_training_gym.frameworks.stitch.sidecar``.
"""

from __future__ import annotations

from modal_training_gym.frameworks.stitch.sidecar_spine import run_sidecar

# slime's decoder is the engine's lazy default, so it need not be injected.
DISK_DELTA_MODULE = "slime.utils.disk_delta"


def main() -> None:
    run_sidecar(disk_delta_module=DISK_DELTA_MODULE, inject_apply_deltas=False)


if __name__ == "__main__":
    main()
