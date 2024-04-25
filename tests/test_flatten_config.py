from data_utils.functional import flatten_dict


def test_flatten_dict():
    assert flatten_dict({"a": [1, 2], "b": [3, 4]}) == [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
