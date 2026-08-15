"""Path resolution for released data and committed results.

This module is the single indirection point that replaces every hardcoded data-root
literal in the original research repo (``output/oneshot_akata_main_dl``,
``oneshot_akata_gptoss_recap``, ``output/oneshot_akata_layerc*``,
``output/oneshot_akata_steer*``, ``analysis/block_b/tables/causal_oneshot/...``).

Two kinds of data:

* **Heavy, downloaded** — the Zenodo deposit, unpacked into ``data_heavy/`` (gitignored)
  by ``scripts/download_data.py``. Override the location with ``SCA_DATA_ROOT``, e.g. to
  point at an existing substrate or a scratch disk::

      export SCA_DATA_ROOT=/mnt/big/sca_data

  The deposit layout the helpers below assume::

      $SCA_DATA_ROOT/
        substrate/{qwen,qwen_instruct,llama31_instruct,gptoss}/{game}/
        gptoss_recap/{game}/
        layerc/{model}/{game}/
        layerc_bridge_residuals/{model}/{game}/
        steering/{smalldose,perm,directions,saturated}/

* **Small, git-tracked** — everything under ``data/`` in this repository: the game
  metadata, the derived human references, the run manifests, and every committed
  analysis table a Tier-1 figure rebuild reads.

Helpers return :class:`pathlib.Path` (composes with ``/``, ``os.path.join`` and f-strings
alike) and never touch the filesystem: they resolve a location, they do not create or
validate it. Use :func:`require` when a missing path should fail with an actionable
message instead of an obscure downstream error.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "data_root",
    "results_root",
    "substrate_root",
    "gptoss_recap_root",
    "layerc_root",
    "layerc_bridge_root",
    "steering_root",
    "games_root",
    "game_features_csv",
    "taxonomy_dir",
    "human_refs_root",
    "manifests_root",
    "traits_path",
    "require",
]

#: Environment variable overriding the heavy-data location.
DATA_ROOT_ENV = "SCA_DATA_ROOT"

#: Environment variable overriding repository-root detection (rarely needed; useful when
#: the package is installed non-editably and the git-tracked ``data/`` tree lives elsewhere).
REPO_ROOT_ENV = "SCA_REPO_ROOT"

_PACKAGE_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Repository root — the directory containing ``strategic_anatomy/`` and ``data/``.

    Honours ``SCA_REPO_ROOT``. Otherwise walks up from this file looking for a
    ``pyproject.toml``, which keeps the answer correct for an editable install and makes
    the failure obvious rather than silent for a non-editable one.
    """
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (_PACKAGE_DIR.parent, *_PACKAGE_DIR.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return _PACKAGE_DIR.parent


# --------------------------------------------------------------------------- heavy data


def data_root() -> Path:
    """Root of the downloaded data deposit (``$SCA_DATA_ROOT``, default ``<repo>/data_heavy``)."""
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data_heavy"


def substrate_root() -> Path:
    """All-layer residual capture substrate: ``substrate/{model}/{game}/``.

    Per game: ``results.parquet``, ``acts.npz``, ``config.json``, ``_DONE``; ``gptoss``
    additionally has ``router.npz``.
    """
    return data_root() / "substrate"


def gptoss_recap_root() -> Path:
    """GPT-OSS uniform-site recapture: ``gptoss_recap/{game}/`` (adds ``genids.npz``)."""
    return data_root() / "gptoss_recap"


def layerc_root() -> Path:
    """Layer C token-lens scores: ``layerc/{model}/{game}/tokens.parquet``."""
    return data_root() / "layerc"


def layerc_bridge_root() -> Path:
    """Layer B/C bridge residuals: ``layerc_bridge_residuals/{model}/{game}/resid.npy``."""
    return data_root() / "layerc_bridge_residuals"


def steering_root() -> Path:
    """Causal steering outputs: ``steering/{smalldose,perm,directions,saturated}/``."""
    return data_root() / "steering"


# --------------------------------------------------------------- small, git-tracked data


def results_root() -> Path:
    """Committed analysis tables — what a Tier-1 figure rebuild reads."""
    return repo_root() / "data" / "results"


def games_root() -> Path:
    """Game metadata: ``game_features.csv`` and the ``taxonomy/`` equivalence tables."""
    return repo_root() / "data" / "games"


def game_features_csv() -> Path:
    """The 144-row canonical game feature table — the canonical action axis lives here."""
    return games_root() / "game_features.csv"


def taxonomy_dir() -> Path:
    """Equivalence tables and human-reference crosswalks derived from the game features."""
    return games_root() / "taxonomy"


def human_refs_root() -> Path:
    """Derived human-reference aggregates (per-game, never per-participant)."""
    return repo_root() / "data" / "human_refs"


def manifests_root() -> Path:
    """Run manifests: ``oneshot_config.json``, the game universe, the 54-game steer sample."""
    return repo_root() / "data" / "manifests"


# -------------------------------------------------------------------------------- traits


def traits_path(spec: str | os.PathLike | None = None) -> Path:
    """Resolve the trait-cue definition file.

    ``spec`` is whatever the run manifest carried. ``data/manifests/oneshot_config.json``
    records ``"src/traits_oneshot.json"`` — the location in the private repo at collection
    time. That string is a **provenance record of the published run** and is deliberately
    left byte-identical, so this resolver accepts it: the file now ships as package data
    inside ``strategic_anatomy/``, and step 3 below recognises it by basename.

    Order: ``spec`` as given → ``<repo>/spec`` → the packaged file of the same name.
    """
    tried: list[Path] = []
    name = "traits_oneshot.json"
    if spec:
        candidate = Path(spec).expanduser()
        tried.append(candidate)
        if candidate.exists():
            return candidate
        rooted = repo_root() / candidate
        tried.append(rooted)
        if rooted.exists():
            return rooted
        name = candidate.name

    packaged = _PACKAGE_DIR / name
    tried.append(packaged)
    if packaged.exists():
        return packaged

    raise FileNotFoundError(
        "Trait definition file not found. Tried:\n  " + "\n  ".join(str(t) for t in tried)
    )


# ----------------------------------------------------------------------------- utilities


def require(path: Path, what: str = "data") -> Path:
    """Return ``path``, or raise :class:`FileNotFoundError` with an actionable message."""
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Required {what} not found at {path}.\n"
        f"Heavy data is not tracked in git. Fetch the deposit with\n"
        f"    python scripts/download_data.py\n"
        f"or point {DATA_ROOT_ENV} at an existing copy:\n"
        f"    export {DATA_ROOT_ENV}=/path/to/deposit"
    )
