"""TD-30f replay harness (synthetic corpora, zero inference)."""
from scripts.analysis.guard_length_replay import replay, trigram_unique_ratio


def test_replay_bins_only_full_length_prefixes_and_reports_min_ratio():
    prose = " ".join(f"w{i}" for i in range(300))
    loop = "the cat sat on the mat " * 60
    flag = lambda t: len(set(zip(*[t.split()[i:] for i in range(3)]))) < 0.5 * max(1, len(t.split()) - 2)
    out = replay({"prose": prose, "loop": loop}, {"g": flag}, lengths=(50, 200, 400))
    g = out["guards"]["g"]
    assert g["50"]["docs"] == 2 and g["50"]["flagged"] == 1 and g["50"]["flagged_docs"] == ["loop"]
    assert g["200"]["docs"] == 2          # loop has 360 words, prose 300
    assert g["400"]["docs"] == 0 and g["400"]["false_positive_rate"] is None
    assert g["50"]["min_trigram_unique_ratio"] < 0.2


def test_trigram_ratio_edge():
    assert trigram_unique_ratio(["a", "b"]) is None
    assert trigram_unique_ratio(["a", "b", "c", "d"]) == 1.0
