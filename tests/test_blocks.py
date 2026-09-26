"""Block construction in ``blocks.build_unit_blocks``."""

from collections import Counter

from npnf.metrics.blocks import UnitMember, UnitScope, build_unit_blocks


def _blocks(n_members: int, draws_per_member: int, n_blocks: int, block_size: int):
    members = [
        UnitMember(genotype_id=f"g{index}", yearsite_uid="ys", condition_index=index)
        for index in range(n_members)
    ]
    return build_unit_blocks(
        members,
        draws_per_member=draws_per_member,
        unit_scope=UnitScope.ENVIRONMENT,
        unit_id="ys",
        n_blocks=n_blocks,
        block_size=block_size,
        seed=0,
    )


def _keys(block) -> list[tuple[int, int]]:
    return [(selection.condition_index, selection.draw_index) for selection in block]


def test_enough_draws_gives_disjoint_full_blocks():
    blocks = _blocks(3, draws_per_member=8, n_blocks=3, block_size=4)
    keys = [key for block in blocks for key in _keys(block)]
    assert [len(block) for block in blocks] == [4, 4, 4]
    assert len(keys) == len(set(keys))


def test_too_few_draws_keeps_full_blocks_without_duplicates():
    blocks = _blocks(3, draws_per_member=4, n_blocks=7, block_size=5)
    assert [len(block) for block in blocks] == [5] * 7
    for block in blocks:
        keys = _keys(block)
        assert len(keys) == len(set(keys))
        assert all(draw < 4 for _, draw in keys)
    # 12 draws give 2 balanced blocks per pass, so 7 blocks take 4 passes.
    usage = Counter(key for block in blocks for key in _keys(block))
    assert max(usage.values()) <= 4


def test_unit_smaller_than_one_block_uses_all_draws_in_every_block():
    blocks = _blocks(2, draws_per_member=2, n_blocks=3, block_size=5)
    all_draws = [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert [sorted(_keys(block)) for block in blocks] == [all_draws] * 3
