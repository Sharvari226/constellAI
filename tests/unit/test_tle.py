"""Unit tests for TLE parsing."""

import pytest

from constellai.orbital_mechanics.tle import parse_tle, parse_tle_file

LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"


def test_parse_tle_extracts_correct_catalog_number():
    record = parse_tle(LINE1, LINE2)
    assert record.satellite_id == "5"


def test_parse_tle_uses_placeholder_name_when_none_given():
    record = parse_tle(LINE1, LINE2)
    assert "5" in record.name


def test_parse_tle_uses_given_name():
    record = parse_tle(LINE1, LINE2, name="Test Satellite")
    assert record.name == "Test Satellite"


def test_parse_tle_preserves_raw_lines():
    record = parse_tle(LINE1, LINE2)
    assert record.line1 == LINE1
    assert record.line2 == LINE2


def test_parse_tle_rejects_garbage_input():
    with pytest.raises(ValueError):
        parse_tle("not a real tle line", "also not real")


def test_parse_tle_file_two_line_format(tmp_path):
    tle_file = tmp_path / "two_line.tle"
    tle_file.write_text(f"{LINE1}\n{LINE2}\n")

    records = parse_tle_file(str(tle_file))

    assert len(records) == 1
    assert records[0].satellite_id == "5"


def test_parse_tle_file_three_line_format(tmp_path):
    tle_file = tmp_path / "three_line.tle"
    tle_file.write_text(f"TEST SAT\n{LINE1}\n{LINE2}\n")

    records = parse_tle_file(str(tle_file))

    assert len(records) == 1
    assert records[0].name == "TEST SAT"


def test_parse_tle_file_multiple_satellites(tmp_path):
    tle_file = tmp_path / "multi.tle"
    tle_file.write_text(f"{LINE1}\n{LINE2}\n{LINE1}\n{LINE2}\n")

    records = parse_tle_file(str(tle_file))

    assert len(records) == 2


def test_parse_tle_file_rejects_malformed_input(tmp_path):
    tle_file = tmp_path / "bad.tle"
    tle_file.write_text(f"{LINE1}\n")

    with pytest.raises(ValueError):
        parse_tle_file(str(tle_file))