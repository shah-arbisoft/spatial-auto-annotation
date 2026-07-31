"""The VLM reply parser: models return JSON, but rarely only JSON."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_vlm_pilot import parse_relations


class TestParse:
    def test_clean_json(self):
        rels, bad = parse_relations(
            '{"relations": [{"s": 0, "p": "on", "o": 3}]}', 5)
        assert rels == [[0, "on", 3]]
        assert not bad

    def test_fenced_json(self):
        reply = '```json\n{"relations": [{"s": 1, "p": "near", "o": 2}]}\n```'
        rels, bad = parse_relations(reply, 5)
        assert rels == [[1, "near", 2]]
        assert not bad

    def test_prose_around_json_is_tolerated(self):
        reply = ('Here are the relationships I found:\n'
                 '{"relations": [{"s": 0, "p": "behind", "o": 1}]}\n'
                 'Let me know if you need more.')
        rels, _ = parse_relations(reply, 3)
        assert rels == [[0, "behind", 1]]

    def test_out_of_range_index_is_dropped_not_crashed(self):
        """A hallucinated object number must not corrupt the score."""
        rels, bad = parse_relations(
            '{"relations": [{"s": 0, "p": "on", "o": 9}, '
            '{"s": 1, "p": "on", "o": 2}]}', 5)
        assert rels == [[1, "on", 2]]
        assert len(bad) == 1

    def test_self_pair_is_dropped(self):
        rels, bad = parse_relations('{"relations": [{"s": 2, "p": "near", "o": 2}]}', 5)
        assert rels == []
        assert len(bad) == 1

    def test_unknown_predicate_is_dropped(self):
        rels, bad = parse_relations(
            '{"relations": [{"s": 0, "p": "next to", "o": 1}]}', 5)
        assert rels == []
        assert "next to" in bad[0]

    def test_predicate_case_and_space_are_normalised(self):
        rels, _ = parse_relations(
            '{"relations": [{"s": 0, "p": "  In Front Of ", "o": 1}]}', 5)
        assert rels == [[0, "in front of", 1]]

    def test_string_indices_are_accepted(self):
        rels, bad = parse_relations(
            '{"relations": [{"s": "0", "p": "on", "o": "1"}]}', 5)
        assert rels == [[0, "on", 1]]
        assert not bad

    def test_malformed_record_is_isolated(self):
        rels, bad = parse_relations(
            '{"relations": [{"p": "on", "o": 1}, {"s": 0, "p": "on", "o": 1}]}', 5)
        assert rels == [[0, "on", 1]]
        assert len(bad) == 1

    def test_no_json_at_all(self):
        rels, bad = parse_relations("I cannot see the image clearly.", 5)
        assert rels == []
        assert bad

    def test_broken_json_reports_rather_than_raises(self):
        rels, bad = parse_relations('{"relations": [{"s": 0, "p": ', 5)
        assert rels == []
        assert bad

    def test_empty_relation_list_is_valid(self):
        rels, bad = parse_relations('{"relations": []}', 5)
        assert rels == []
        assert not bad
