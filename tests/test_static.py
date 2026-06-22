import ast
from pathlib import Path


def test_python_sources_parse():
    for path in Path("src").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_package_imports_without_real_adios2():
    import nanoplasma_analysis

    assert hasattr(nanoplasma_analysis, "NanoPlasmaRun")
    assert hasattr(nanoplasma_analysis.NanoPlasmaRun, "export_reduced_h5")


def test_step_parser():
    from nanoplasma_analysis import extract_step_from_filename

    assert extract_step_from_filename("simData_000123.bp5") == 123
    assert extract_step_from_filename("/tmp/openPMD/simOutput_200000.bp5") == 200000
