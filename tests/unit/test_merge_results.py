# Tests for merging results written by parallel Monte Carlo processes.

from constellai.scripts.merge_results import merge_results


def test_cells_from_all_files_are_kept():
    a = {"meta": {"git_commit": "abc"}, "cells": {"k1": {"v": 1}}}
    b = {"meta": {"git_commit": "abc"}, "cells": {"k2": {"v": 2}}}
    merged = merge_results([a, b])
    assert set(merged["cells"]) == {"k1", "k2"}
    assert merged["meta"]["git_commit"] == "abc"
    assert merged["meta"]["merged_files"] == 2


def test_a_duplicate_key_keeps_the_later_file():
    a = {"meta": {}, "cells": {"k": {"v": 1}}}
    b = {"meta": {}, "cells": {"k": {"v": 2}}}
    assert merge_results([a, b])["cells"]["k"] == {"v": 2}