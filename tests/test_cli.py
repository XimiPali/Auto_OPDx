"""Tests for the ``align-plates`` tool.

These cover the parts that break when someone else runs it: file pairing across
naming conventions, argument handling, and the failure paths. The alignment
maths itself is covered by ``test_lattice.py``; here we care that the tool
behaves predictably when pointed at real folders.
"""

from __future__ import annotations

import contextlib
import json
import time

import pytest

from Auto_OPDx.alignment.cli import (
    _photo_window,
    build_parser,
    find_pairs,
    main,
)


def _touch(d, name):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"")
    return p


# --------------------------------------------------------------- pairing ----

def test_pairs_identical_stems(tmp_path):
    img, lab = tmp_path / "img", tmp_path / "lab"
    for s in ("P1", "P2"):
        _touch(img, f"{s}.jpg")
        _touch(lab, f"{s}.OPDx")
    pairs, no_photo, no_scan = find_pairs(img, lab)
    assert [p[0] for p in pairs] == ["P1", "P2"]
    assert no_photo == [] and no_scan == []


def test_pairs_mixed_photo_extensions(tmp_path):
    """Half the real dataset is .bmp and half .jpg."""
    img, lab = tmp_path / "img", tmp_path / "lab"
    _touch(img, "P1.jpg")
    _touch(img, "P2.bmp")
    _touch(lab, "P1.OPDx")
    _touch(lab, "P2.OPDx")
    pairs, _, _ = find_pairs(img, lab)
    assert {p[0] for p in pairs} == {"P1", "P2"}


def test_pairs_reports_orphans_both_ways(tmp_path):
    img, lab = tmp_path / "img", tmp_path / "lab"
    _touch(img, "P1.jpg")
    _touch(img, "stray.jpg")
    _touch(lab, "P1.OPDx")
    _touch(lab, "lonely.OPDx")
    pairs, no_photo, no_scan = find_pairs(img, lab)
    assert [p[0] for p in pairs] == ["P1"]
    assert no_photo == ["lonely"]       # scan without a photo
    assert no_scan == ["stray"]         # photo without a scan


def test_pattern_handles_differing_names(tmp_path):
    """--pattern is the escape hatch when the folders disagree on naming."""
    img, lab = tmp_path / "img", tmp_path / "lab"
    _touch(img, "P1_photo.jpg")
    _touch(lab, "P1.OPDx")
    assert find_pairs(img, lab)[0] == []            # default pattern misses it
    pairs, _, _ = find_pairs(img, lab, "{stem}_photo")
    assert [p[0] for p in pairs] == ["P1"]


def test_extension_case_is_ignored(tmp_path):
    """Real scans are '.OPDx'; nobody should have to match that casing."""
    img, lab = tmp_path / "img", tmp_path / "lab"
    _touch(img, "P1.JPG")
    _touch(lab, "P1.opdx")
    pairs, _, _ = find_pairs(img, lab)
    assert [p[0] for p in pairs] == ["P1"]


def test_non_image_files_are_ignored(tmp_path):
    img, lab = tmp_path / "img", tmp_path / "lab"
    _touch(img, "P1.jpg")
    _touch(img, "notes.txt")
    _touch(img, ".DS_Store")
    _touch(lab, "P1.OPDx")
    _touch(lab, "README.md")
    pairs, _, no_scan = find_pairs(img, lab)
    assert [p[0] for p in pairs] == ["P1"]
    assert no_scan == []


# ---------------------------------------------------------- photo window ----

class _Fit:
    def __init__(self, pitch):
        self.pitch = pitch


def test_window_sizes_off_the_tighter_pitch():
    """The two photo pitches differ (~250 px by column, ~182 by row).

    A square window must fit inside the *smaller* one or it reaches into the
    neighbouring cell.
    """
    assert _photo_window(_Fit((250.0, 182.0)), None) == 173    # 182 * 0.95
    assert _photo_window(_Fit((182.0, 250.0)), None) == 173     # order-independent


def test_window_override_wins():
    assert _photo_window(_Fit((250.0, 182.0)), 64) == 64


def test_window_survives_a_degenerate_pitch():
    assert _photo_window(_Fit((float("nan"), 0.0)), None) == 160


# -------------------------------------------------------------- the CLI -----

def test_missing_input_folder_exits_nonzero(tmp_path, capsys):
    code = main(["--images", str(tmp_path / "nope"),
                 "--labels", str(tmp_path),
                 "--out", str(tmp_path / "out")])
    assert code == 1
    assert "not a folder" in capsys.readouterr().err


def test_no_pairs_exits_nonzero_and_hints_at_pattern(tmp_path, capsys):
    img, lab = tmp_path / "img", tmp_path / "lab"
    _touch(img, "A.jpg")
    _touch(lab, "B.OPDx")
    code = main(["--images", str(img), "--labels", str(lab),
                 "--out", str(tmp_path / "out")])
    assert code == 1
    assert "--pattern" in capsys.readouterr().err


def test_dry_run_writes_nothing(tmp_path, capsys):
    img, lab, out = tmp_path / "img", tmp_path / "lab", tmp_path / "out"
    _touch(img, "P1.jpg")
    _touch(lab, "P1.OPDx")
    code = main(["--images", str(img), "--labels", str(lab),
                 "--out", str(out), "--dry-run"])
    assert code == 0
    assert not out.exists(), "a dry run must not create the output folder"
    assert "dry run" in capsys.readouterr().out


def test_unreadable_plate_fails_without_stopping_the_run(tmp_path, capsys):
    """One corrupt file must not abort a 69-plate batch."""
    img, lab, out = tmp_path / "img", tmp_path / "lab", tmp_path / "out"
    _touch(img, "P1.jpg")          # empty files -> loader raises
    _touch(lab, "P1.OPDx")
    code = main(["--images", str(img), "--labels", str(lab), "--out", str(out)])
    assert code == 1               # nothing succeeded
    printed = capsys.readouterr().out
    assert "FAILED" in printed
    # the run still produced its bookkeeping
    assert (out / "summary.csv").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["plates_failed"] == 1
    assert manifest["plates_ok"] == 0


def test_truncated_scan_is_rejected_not_hung(tmp_path):
    """Regression: DektakLoad loops forever on a short file instead of raising.

    An unattended 69-plate batch must not stall on one bad input, so the size
    guard has to fail fast. If this test ever hangs, the guard was removed.
    """
    from Auto_OPDx.alignment.plate import load_plate

    bad = tmp_path / "truncated.OPDx"
    bad.write_bytes(b"")
    with pytest.raises(ValueError, match="empty or truncated"):
        load_plate(bad)

    bad.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="empty or truncated"):
        load_plate(bad)


def test_load_plate_rejects_a_missing_file(tmp_path):
    from Auto_OPDx.alignment.plate import load_plate

    with pytest.raises(ValueError, match="not a file"):
        load_plate(tmp_path / "nope.OPDx")


def test_watchdog_interrupts_a_stuck_call():
    from Auto_OPDx.alignment.cli import _watchdog

    with pytest.raises(TimeoutError, match="--timeout"):
        with _watchdog(1):
            time.sleep(3)


def test_watchdog_is_a_noop_when_disabled():
    from Auto_OPDx.alignment.cli import _watchdog

    with _watchdog(0):
        pass            # must not raise, must not arm an alarm


def test_watchdog_restores_the_previous_handler():
    """A fired alarm must not leak into the next plate."""
    import signal

    from Auto_OPDx.alignment.cli import _watchdog

    before = signal.getsignal(signal.SIGALRM)
    with contextlib.suppress(TimeoutError):
        with _watchdog(1):
            time.sleep(3)
    assert signal.getsignal(signal.SIGALRM) is before


def test_parser_defaults_match_the_documented_ones():
    a = build_parser().parse_args(
        ["--images", "i", "--labels", "l", "--out", "o"]
    )
    assert (a.rows, a.cols) == (8, 8)
    assert a.label_um == 110.0
    assert a.photo_px == 96
    assert a.height_px == 32
    assert a.photo_window_px is None      # auto
    assert a.pattern == "{stem}"
    assert a.timeout == 180


def test_parser_requires_all_three_folders():
    for args in (["--images", "i"],
                 ["--labels", "l"],
                 ["--images", "i", "--labels", "l"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(args)


def test_skip_existing_with_everything_done_keeps_aggregates(tmp_path, capsys):
    # regression: a rerun that skipped every plate used to overwrite
    # manifest.json with plates_ok=0 while summary.csv/cells.csv stayed put
    img, lab, out = tmp_path / "img", tmp_path / "lab", tmp_path / "out"
    _touch(img, "A.jpg")
    _touch(lab, "A.OPDx")
    _touch(out / "plates", "A.npz")
    manifest = out / "manifest.json"
    manifest.write_text('{"plates_ok": 1}\n')

    code = main(["--images", str(img), "--labels", str(lab),
                 "--out", str(out), "--skip-existing"])

    assert code == 0
    assert "nothing left to process" in capsys.readouterr().out
    assert manifest.read_text() == '{"plates_ok": 1}\n'
    assert not (out / "summary.csv").exists()


def test_partial_skip_does_not_rewrite_aggregates(tmp_path, capsys):
    img, lab, out = tmp_path / "img", tmp_path / "lab", tmp_path / "out"
    _touch(img, "A.jpg")
    _touch(lab, "A.OPDx")
    _touch(img, "B.jpg")
    _touch(lab, "B.OPDx")
    _touch(out / "plates", "A.npz")          # A already done, B is new
    manifest = out / "manifest.json"
    manifest.write_text('{"plates_ok": 1}\n')
    # B's scan is an empty file, so it fails fast without a real plate
    code = main(["--images", str(img), "--labels", str(lab),
                 "--out", str(out), "--skip-existing", "--quiet"])

    assert code == 1  # B fails on the empty scan
    assert "were NOT rewritten" in capsys.readouterr().out
    assert manifest.read_text() == '{"plates_ok": 1}\n'
    assert not (out / "summary.csv").exists()
