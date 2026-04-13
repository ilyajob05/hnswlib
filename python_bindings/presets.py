from dataclasses import dataclass
from typing import Literal


@dataclass
class TurboQuantPreset:
    name: str
    bits_per_coord: int
    M: int
    ef_construction: int
    ef_search: int
    description: str
    expected_recall_sift: str
    expected_recall_glove: str
    speed_vs_l2: str


# ==================== Presets ====================

PRESETS = {
    "fast": TurboQuantPreset(
        name="fast",
        bits_per_coord=8,
        M=16,
        ef_construction=200,
        ef_search=128,
        description="Maximum search speed. Adaptability; high throughput if required; recall of ~0.95+ is acceptable",
        expected_recall_sift="~0.955–0.965 @10",
        expected_recall_glove="~0.987–0.993 @10",
        speed_vs_l2="~2.0–2.2× faster than L2",
    ),
    "balanced": TurboQuantPreset(
        name="balanced",
        bits_per_coord=8,
        M=32,
        ef_construction=400,
        ef_search=256,
        description="Optimal balance between speed and quality (recommended as default)",
        expected_recall_sift="~0.967–0.968 @10",
        expected_recall_glove="~0.994 @10",
        speed_vs_l2="~1.6–1.8× faster than L2",
    ),
    "high_quality": TurboQuantPreset(
        name="high_quality",
        bits_per_coord=8,
        M=32,
        ef_construction=400,
        ef_search=512,
        description="Maximum quality without reranking. The best recall among quantized variants",
        expected_recall_sift="~0.968 @10",
        expected_recall_glove="~0.994–0.995 @10",
        speed_vs_l2="~1.4× faster than L2",
    ),
    "aggressive": TurboQuantPreset(
        name="aggressive",
        bits_per_coord=4,
        M=32,
        ef_construction=400,
        ef_search=512,
        description="Maximum compression (4-bit). Use only with reranking (L2-graph + TQ-search + rerank).",
        expected_recall_sift="~0.68–0.69 @10 (without rerank)",
        expected_recall_glove="~0.94–0.96 @10 (without rerank)",
        speed_vs_l2="~2.5–3.0× faster than L2",
    ),
}


def get_preset(
    name: Literal["fast", "balanced", "high_quality", "aggressive"],
) -> TurboQuantPreset:
    """Returns a preset by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


# Удобная функция для быстрого создания Space + Index
def create_index_with_preset(
    dim: int,
    preset_name: Literal["fast", "balanced", "high_quality", "aggressive"] = "balanced",
    **hnsw_kwargs,
):
    """Creates a TurboQuantSpace + hnswlib.Index with the selected preset."""
    import hnswlib
    from turboquant import TurboQuantSpace

    preset = get_preset(preset_name)

    space = TurboQuantSpace(dim=dim, bits_per_coord=preset.bits_per_coord)

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(
        max_elements=...,
        M=preset.M,
        ef_construction=preset.ef_construction,
        **hnsw_kwargs,
    )

    print(f"✅ Created index with preset '{preset.name}'")
    print(f"   {preset.description}")
    print(
        f"   bits={preset.bits_per_coord}, M={preset.M}, ef_construction={preset.ef_construction}"
    )
    return space, index, preset
