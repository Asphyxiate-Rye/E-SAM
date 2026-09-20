# Stub: build_sam.py imports `LoRA_Sam` at module level (train.py also
# imports `LoRA_Sam_prompt`), but the MMWHS/Sam_my training path never
# instantiates either (only sam_model_registry['vit_b'] -> Sam_my is used,
# see _build_sam_my in build_sam.py). Never present in this repo's git
# history (checked). Stub only so `import train`/`import build_sam`
# doesn't crash on the dead import.
class LoRA_Sam:
    pass


class LoRA_Sam_prompt:
    pass
