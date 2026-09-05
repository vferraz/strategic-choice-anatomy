PY ?= uv run python
SH ?= bash

.DEFAULT_GOAL := help
.PHONY: help test figures figures-tier1 clean-figures \
        tables tables-layer-a tables-layer-b tables-layer-c tables-steering \
        verify verify-data manifest download-data check gate hygiene figures-diff

# ── help ────────────────────────────────────────────────────────────────────────────────
help:  ## show this help
	@echo "strategic-choice-anatomy — make targets"
	@echo
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Tier 1 (clone only)      : make test figures-tier1"
	@echo "Tier 2 (needs deposit)   : make download-data && make tables verify  (coverage: REPRODUCING.md)"
	@echo "Tier 3 (needs GPU)       : see REPRODUCING.md and scripts/launchers/"

# ── Tier 1 ──────────────────────────────────────────────────────────────────────────────
test:  ## run the test suite (no GPU, no deposit)
	$(PY) -m pytest -q -m "not tier2 and not gpu"

figures-tier1:  ## rebuild the 6 figures that need only the committed tables
	TIER1_ONLY=1 PY="$(PY)" $(SH) scripts/dev/tier1_figures.sh

figures:  ## rebuild all 11 paper figures (5 of them need the deposit)
	PY="$(PY)" $(SH) scripts/dev/tier1_figures.sh

figures-diff:  ## pixel-compare rebuilt figures against the committed reference renders
	$(PY) scripts/dev/compare_figures.py

clean-figures:  ## delete rebuilt figures (the committed reference PDFs are untouched)
	@find analysis -type d -name figures -prune -exec sh -c \
	  'rm -f "$$1"/*.pdf "$$1"/*.png' _ {} \;
	@rm -rf analysis/layer_a/figures/_attrib_panel_data
	@echo "removed rebuilt figures; data/results/figures_reference/ is untouched"

# ── Tier 2 — regenerate tables from the deposit ─────────────────────────────────────────
tables: tables-layer-a tables-layer-b tables-layer-c tables-steering  ## regenerate the four table families (see REPRODUCING.md for what is and is not covered)

tables-layer-a:  ## Layer A: behaviour, model selection, traits, attribution
	$(PY) analysis/layer_a/build_data_layer.py
	$(PY) analysis/layer_a/build_regime_rationality.py
	$(PY) analysis/layer_a/build_trait_steering_oneshot.py
	$(PY) analysis/layer_a/validate_foundation.py
	$(PY) analysis/layer_a/build_attribution.py

tables-layer-b:  ## Layer B: decodability, crystallization, fusion, recruitment
	$(PY) analysis/layer_b/build_residual_cache.py
	$(PY) analysis/layer_b/build_decodability.py
	$(PY) analysis/layer_b/build_crystallization.py
	$(PY) analysis/layer_b/validate_foundation.py
	$(PY) analysis/layer_b/fusion_associative/recap_cache.py
	$(PY) analysis/layer_b/fusion_associative/oss_router_fusion.py
	$(PY) analysis/layer_b/fusion_associative/build_fusion_figures.py
	$(PY) analysis/layer_b/recruitment/build_all.py
# rebuild_bridge_variants LAST: it is a post-processing patch, not an independent builder.
# It replaces the GPT-OSS rows of within_model_bridge.csv, within_model_bridge_points.csv and
# final_geometry_bridge.csv -- all three of which recruitment/build_all.py writes wholesale --
# so running it earlier means build_all silently discards its output.
	$(PY) analysis/layer_b/rebuild/rebuild_bridge_variants.py

tables-layer-c:  ## Layer C: token-lens scores and statistics
	$(PY) analysis/layer_c/compute_layerc.py
	$(PY) analysis/layer_c/stats_layerc.py

tables-steering:  ## Steering: small-dose analysis and summaries
	$(PY) analysis/steering/analyze_smalldose.py
	$(PY) analysis/steering/smalldose_final_summary.py

verify:  ## Tier 2: regenerate tables and hash-compare against the committed copies
	$(PY) -m pytest -q -m tier2

# ── data ────────────────────────────────────────────────────────────────────────────────
download-data:  ## fetch the released deposit into data_heavy/ (~21 GB)
	$(PY) scripts/download_data.py

verify-data:  ## check the committed data/ tree against data/MANIFEST.json
	$(PY) scripts/download_data.py --verify-committed

manifest:  ## regenerate data/MANIFEST.json
	$(PY) scripts/dev/build_manifest.py

# ── development gates ───────────────────────────────────────────────────────────────────
gate:  ## collection/steering surface gate — NEEDS the [gpu] extra (those modules import torch)
	PY="$(PY)" $(SH) scripts/dev/check_phase2.sh

hygiene:  ## no machine paths, secrets, symlinks or heavy data in anything git would stage
	$(SH) scripts/dev/check_hygiene.sh

check: test verify-data hygiene  ## everything a clone can check without the deposit or a GPU
	@$(PY) scripts/dev/build_manifest.py --check
	@echo
	@echo "all clone-only checks passed"
	@echo "(\`make gate\` additionally checks the collection/steering surface; it needs the [gpu] extra)"
