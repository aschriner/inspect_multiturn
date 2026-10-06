from importlib.metadata import entry_points, version

import inspect_multiturn


def test_version_matches_metadata():
    assert inspect_multiturn.__version__ == version("inspect-multiturn")


def test_inspect_entry_point_is_registered():
    eps = entry_points(group="inspect_ai")
    ep = next(ep for ep in eps if ep.name == "inspect_multiturn")
    assert ep.value == "inspect_multiturn._registry"
    ep.load()
