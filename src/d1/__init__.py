"""D1 alignment-audit pipeline: LLM activations -> ridge readout -> verdict.

Inference + ridge regression only. Nothing is trained, nothing touches a GPU.
Entry point: ``python scripts/run_d1.py --smoke`` (see README).
"""
