import re
from pathlib import Path

from offline_particles.kernels.status import Status
from offline_particles.kernels.status._status import STATUS_VALUES


def _read_status_names_from_pxd() -> set[str]:
    pxd_path = Path(__file__).resolve().parents[2] / "src" / "offline_particles" / "kernels" / "status" / "__init__.pxd"
    names: set[str] = set()
    pattern = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=")

    for line in pxd_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            names.add(match.group(1))

    return names


def test_status_members_match_cython_names() -> None:
    assert set(STATUS_VALUES) == {member.name for member in Status}


def test_status_values_match_cython_values() -> None:
    status_values_from_enum = {member.name: int(member) for member in Status}
    assert STATUS_VALUES == status_values_from_enum


def test_status_values_include_all_pxd_enum_members() -> None:
    pxd_names = _read_status_names_from_pxd()
    missing = pxd_names.difference(STATUS_VALUES)
    assert not missing, f"Missing STATUS_VALUES entries for: {sorted(missing)}"
