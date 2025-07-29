"""Golden bytes for JSONL export; drift must be deliberate."""

from __future__ import annotations

from wavrail.ctc._common import Hypothesis
from wavrail.report.export import hypothesis_record, to_jsonl, write_jsonl

GOLDEN_LINE = (
    '{"confidence":0.75,"frame_indices":[0,4],"kind":"hypothesis",'
    '"name":"utt-1","schema_version":"1","text":"你好 ab","tokens":[1,2]}'
)


def _record() -> dict:
    hyp = Hypothesis(tokens=(1, 2), text="你好 ab", confidence=0.75, frame_indices=(0, 4))
    return hypothesis_record("utt-1", hyp)


def test_to_jsonl_matches_golden_bytes() -> None:
    assert to_jsonl([_record()]) == GOLDEN_LINE + "\n"


def test_written_file_matches_golden_bytes(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    write_jsonl([_record()], path)
    assert path.read_bytes() == (GOLDEN_LINE + "\n").encode("utf-8")


def test_multiple_records_keep_one_line_each(tmp_path) -> None:
    path = tmp_path / "golden.jsonl"
    write_jsonl([_record(), _record()], path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [GOLDEN_LINE, GOLDEN_LINE]
