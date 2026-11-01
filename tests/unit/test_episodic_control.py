"""AP-29 validation refuses malformed inputs before opening an experiment."""
import pytest

from orchestration.repl_memory.episodic_control import run_paired_arms


@pytest.mark.parametrize("episodes,deleted", [
    ([{"id": "a"}, {"id": "a"}], set()),
    ([{"id": "a"}], {"missing"}),
    ([{"id": "a", "entry_type": "pattern"}], set()),
])
def test_invalid_pairs_fail_before_candidate(episodes, deleted):
    def forbidden(_):
        pytest.fail("candidate must not run on invalid inputs")
    with pytest.raises(ValueError):
        run_paired_arms(episodes, delete_ids=deleted, embedder=object(),
                        embedding_dim=32, query_text="alpha", abstract=forbidden)
