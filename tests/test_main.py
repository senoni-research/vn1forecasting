import sys

from pytest import CaptureFixture, MonkeyPatch

from vn1forecasting.__main__ import main


def test_main_version(capsys: CaptureFixture[str], monkeypatch: MonkeyPatch) -> None:
    # Mock sys.argv to simulate passing the --version argument
    monkeypatch.setattr(sys, "argv", ["vn1forecasting", "--version"])

    main()
    captured = capsys.readouterr()
    assert "vn1forecasting v0.1.0" in captured.out


def test_main_default(capsys: CaptureFixture[str], monkeypatch: MonkeyPatch) -> None:
    # Mock sys.argv to simulate no arguments
    monkeypatch.setattr(sys, "argv", ["vn1forecasting"])

    main()
    captured = capsys.readouterr()
    assert "Welcome to vn1forecasting!" in captured.out
