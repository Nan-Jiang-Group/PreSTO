"""Backend-specific model wrappers and samplers.

Import the subpackage for the serving stack you have installed (``backends.hf``, ``backends.vllm``,
``backends.sglang``); this namespace stays import-free so one backend never drags in the others' dependencies.
"""
