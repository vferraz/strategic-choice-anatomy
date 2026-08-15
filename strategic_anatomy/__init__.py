"""Shared runtime for *strategic-choice-anatomy*.

Submodules are imported explicitly, never eagerly from here. In particular ``runtime``,
``router_capture``, ``action_direction``, ``steering_hooks`` and ``steering_utils`` import
``torch`` at module level: importing any of them from this ``__init__`` would make the
whole package unimportable in an analysis-only environment, which would break the Tier-1
figure rebuild. The torch-free surface is ``games``, ``reference_games``, ``prompting``,
``paper_style``, ``eq_engine`` and ``config``.
"""

__version__ = "0.1.0"
