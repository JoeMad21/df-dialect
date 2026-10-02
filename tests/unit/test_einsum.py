import pytest

from df_dialect.df.einsum import EinsumError, parse_einsum


def test_matmul_shape():
    assert parse_einsum("mk,kn->mn").result_shape((64, 128), (128, 32)) == (64, 32)


def test_batched_attention_scores():
    spec = parse_einsum("bqd,bkd->bqk")
    assert spec.result_shape((8, 16, 64), (8, 32, 64)) == (8, 16, 32)


@pytest.mark.parametrize(
    "spec, message",
    [
        ("mk,kn", "exactly one '->'"),
        ("mK,kn->mn", "lowercase"),
        ("mm,kn->mn", "repeats an index"),
        ("mk,kn->mz", "does not appear"),
    ],
)
def test_malformed_specs(spec, message):
    with pytest.raises(EinsumError, match=message):
        parse_einsum(spec)


def test_inconsistent_sizes():
    with pytest.raises(EinsumError, match="index 'k'"):
        parse_einsum("mk,kn->mn").result_shape((4, 8), (4, 8))


def test_rank_mismatch():
    with pytest.raises(EinsumError, match="rank"):
        parse_einsum("mk,kn->mn").result_shape((4, 8, 2), (8, 4))
