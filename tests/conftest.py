from pathlib import Path

import pytest

TESTDATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def ts1_xml_path_module() -> Path:
    return TESTDATA / "TS_1.xml"


@pytest.fixture
def ts1_xml_path() -> Path:
    """Warp tilt-series XML with populated movement grids (C#-golden fixture,
    vendored from warpylib testdata)."""
    return TESTDATA / "TS_1.xml"


@pytest.fixture
def test_aln_path() -> Path:
    """Small self-consistent AreTomo3 .aln with local alignments (2 patches, 2 dark
    frames, SEC 1,3,4,5,6,8 over 8 raw sections — vendored from cryoet-alignment
    tests/data; cryoet-alignment >= 0.3 enforces AreTomo3's file invariants at parse)."""
    return TESTDATA / "test.aln"
