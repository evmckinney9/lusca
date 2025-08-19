"""Tests for src.lusca.mpl_freeze."""

from src.lusca.mpl_freeze import _parse_line


def test_parse_line():
    """Test with a simple command."""
    line = "demo_plot var1 var2 --outdir custom_dir"
    name, vars, outdir = _parse_line(line)

    assert name == "demo_plot"
    assert vars == ["var1", "var2"]
    assert outdir == "custom_dir"

    # Test with default output directory
    line = "demo_plot var1 var2"
    name, vars, outdir = _parse_line(line)

    assert name == "demo_plot"
    assert vars == ["var1", "var2"]
    assert outdir == "docs/figs"

    # Test with no variables
    line = "demo_plot"
    name, vars, outdir = _parse_line(line)

    assert name == "demo_plot"
    assert vars == []
    assert outdir == "docs/figs"
