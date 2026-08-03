"""Default Gemma4-VL's ``_pg_collection`` so the training forward can run under slime.

``Gemma4VLModel.forward`` passes ``self.config._pg_collection`` into
``slice_batch_for_context_parallel``, which dereferences ``pg_collection.cp``.
slime's megatron model_provider builds bridge models without ever setting one, so
the first training forward dies with ``AttributeError: 'NoneType' object has no
attribute 'cp'``.

``ProcessGroupCollection.use_mpu_process_groups()`` is what the sibling VL models fall
back to when handed a None collection, and it is the default Qwen3-ASR's error message
names (``patch_qwen3_asr_pg_collection`` fixes the same gap for a model that raises
rather than dereferences). By the time a forward runs, slime has initialized
model-parallel state, so the MPU groups exist. Defaulting to them rather than a no-op
keeps CP slicing correct at ``context_parallel_size > 1``. Report upstream.

Idempotent. Run at image build:  python patch_gemma4_vl_pg_collection.py
"""

import pathlib

MARKER = "PATCHED_GEMMA4_VL_PG_COLLECTION"

IMPORT_ANCHOR = "from megatron.core.transformer.module import MegatronModule"
IMPORT_INJECT = (
    "\nfrom megatron.core.process_groups_config import "
    f"ProcessGroupCollection  # {MARKER}"
)

OLD_CALL = "            pg_collection=self.config._pg_collection,"
NEW_CALL = f"""            pg_collection=(
                getattr(self.config, "_pg_collection", None)
                or ProcessGroupCollection.use_mpu_process_groups()
            ),  # {MARKER}"""


def main() -> None:
    targets = list(
        pathlib.Path("/usr/local/lib").glob(
            "python3.*/dist-packages/megatron/bridge/models/gemma_vl/modeling_gemma4_vl.py"
        )
    )
    if not targets:
        print("WARNING: modeling_gemma4_vl.py not found; skipping pg_collection patch")
        return

    for path in targets:
        src = path.read_text()
        if MARKER in src:
            print("compat: gemma4_vl pg_collection already defaulted:", path)
            continue
        if OLD_CALL not in src or IMPORT_ANCHOR not in src:
            print(
                "WARNING: could not find pg_collection call site or import anchor in",
                path,
                "(upstream may have changed; pg_collection patch NOT applied)",
            )
            continue
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_INJECT, 1)
        src = src.replace(OLD_CALL, NEW_CALL, 1)
        path.write_text(src)
        print("compat: defaulted gemma4_vl pg_collection:", path)


if __name__ == "__main__":
    main()
