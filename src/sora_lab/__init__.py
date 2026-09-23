"""Persistent orchestration primitives for the Sora improvement lab.

The lab package deliberately does not run arbitrary LLM-generated commands.  It
stores proposals and measurements, then lets a separately reviewed runner
execute an allow-listed experiment.
"""

__all__ = ["store", "policy", "agents", "advisor", "attack_registry", "attack_archive",
           "attack_evolution", "calibration", "worlds", "resources", "scheduler", "coevolution"]
