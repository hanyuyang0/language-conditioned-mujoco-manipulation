from vla_sim.data import create_episode_split, validate_episode_split


def test_stratified_episode_split_is_disjoint_and_balanced():
    records = [
        {"episode": i, "task": "push" if i < 100 else "pick_place"}
        for i in range(200)
    ]
    split = create_episode_split(records, seed=42)
    assert len(split["train"]) == 160
    assert len(split["val"]) == 20
    assert len(split["test"]) == 20
    assert split["task_counts"]["push"] == {"train": 80, "val": 10, "test": 10}
    assert split["task_counts"]["pick_place"] == {"train": 80, "val": 10, "test": 10}
    validate_episode_split(split, list(range(200)))
    assert split == create_episode_split(records, seed=42)


def test_overlap_is_rejected():
    bad = {"train": [0, 1], "val": [1, 2], "test": [3]}
    try:
        validate_episode_split(bad)
    except ValueError as error:
        assert "leakage" in str(error).lower()
    else:
        raise AssertionError("overlap should have raised ValueError")
