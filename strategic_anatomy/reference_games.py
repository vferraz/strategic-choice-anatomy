"""Reference games loader for Nagel and Griffiths datasets."""
import json
import warnings

from strategic_anatomy.config import games_root


def load_reference_games() -> dict:
    """Load reference games from processed JSONs. Returns empty dict if files missing."""
    games = {}
    processed_dir = games_root()
    for fname in ["nagel_games.json", "griffiths_games.json"]:
        fpath = processed_dir / fname
        if fpath.exists():
            raw = json.loads(fpath.read_text())
            games.update(raw)
        else:
            warnings.warn(
                f"Reference games file not found: {fpath}. "
                "Run scripts/prepare_reference_datasets.py first."
            )
    return games


# Lazy singleton
_cache = None


def get_reference_games() -> dict:
    global _cache
    if _cache is None:
        _cache = load_reference_games()
    return _cache
