"""Unit tests for :class:`~app.services.hazards.HazardRuleEngine`.

All tests are pure — no I/O, no inference deps.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.services.detector import Detection
from app.services.hazards import HazardRuleEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det(label: str, bbox: list[float], conf: float = 0.9, cls: int = 0) -> Detection:
    return Detection(label=label, confidence=conf, bbox=bbox, class_id=cls)


def _write_rules(path: Path, rules: list[dict]) -> Path:
    config_path = path / "hazards.yaml"
    config_path.write_text(yaml.dump({"hazards": rules}))
    return config_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_empty_when_file_missing(self, tmp_path):
        engine = HazardRuleEngine(config_path=tmp_path / "no.yaml")
        assert engine.evaluate([]) == []

    def test_loads_rules_from_yaml(self, tmp_path):
        path = _write_rules(tmp_path, [{"name": "fire", "labels": ["fire"], "severity": "critical"}])
        engine = HazardRuleEngine(config_path=path)
        assert len(engine._rules) == 1


# ---------------------------------------------------------------------------
# Label matching
# ---------------------------------------------------------------------------


class TestLabelMatching:
    def test_matches_exact_label(self, tmp_path):
        path = _write_rules(tmp_path, [{"name": "fire", "labels": ["fire"], "severity": "high"}])
        engine = HazardRuleEngine(config_path=path)
        alerts = engine.evaluate([_det("fire", [0, 0, 100, 100])])
        assert len(alerts) == 1
        assert alerts[0].name == "fire"
        assert alerts[0].severity == "high"

    def test_no_match_wrong_label(self, tmp_path):
        path = _write_rules(tmp_path, [{"name": "fire", "labels": ["fire"], "severity": "high"}])
        engine = HazardRuleEngine(config_path=path)
        assert engine.evaluate([_det("person", [0, 0, 100, 100])]) == []

    def test_multiple_detections_same_label_trigger_multiple_alerts(self, tmp_path):
        path = _write_rules(tmp_path, [{"name": "fire", "labels": ["fire"], "severity": "high"}])
        engine = HazardRuleEngine(config_path=path)
        alerts = engine.evaluate([
            _det("fire", [0, 0, 50, 50]),
            _det("fire", [100, 100, 200, 200]),
        ])
        assert len(alerts) == 2

    def test_no_detections_no_alerts(self, tmp_path):
        path = _write_rules(tmp_path, [{"name": "fire", "labels": ["fire"], "severity": "high"}])
        engine = HazardRuleEngine(config_path=path)
        assert engine.evaluate([]) == []


# ---------------------------------------------------------------------------
# Aspect-ratio heuristic
# ---------------------------------------------------------------------------


class TestAspectRatio:
    def _make_engine(self, tmp_path, ratio: float) -> HazardRuleEngine:
        path = _write_rules(tmp_path, [{
            "name": "fallen_person",
            "labels": ["person"],
            "severity": "high",
            "aspect_ratio_min": ratio,
        }])
        return HazardRuleEngine(config_path=path)

    def test_triggers_when_ratio_met(self, tmp_path):
        engine = self._make_engine(tmp_path, 1.5)
        # width=200, height=100 → ratio=2.0 ≥ 1.5 → triggers
        alerts = engine.evaluate([_det("person", [0, 0, 200, 100])])
        assert len(alerts) == 1

    def test_no_trigger_when_ratio_not_met(self, tmp_path):
        engine = self._make_engine(tmp_path, 1.5)
        # width=100, height=200 → ratio=0.5 < 1.5 → no trigger
        assert engine.evaluate([_det("person", [0, 0, 100, 200])]) == []

    def test_no_trigger_when_height_zero(self, tmp_path):
        engine = self._make_engine(tmp_path, 1.5)
        assert engine.evaluate([_det("person", [0, 0, 100, 0])]) == []


# ---------------------------------------------------------------------------
# Proximity (near_labels)
# ---------------------------------------------------------------------------


class TestProximity:
    def _make_engine(self, tmp_path, proximity_px: int = 200) -> HazardRuleEngine:
        path = _write_rules(tmp_path, [{
            "name": "stove_with_person",
            "labels": ["oven"],
            "severity": "medium",
            "near_labels": ["person"],
            "proximity_px": proximity_px,
        }])
        return HazardRuleEngine(config_path=path)

    def test_triggers_when_nearby(self, tmp_path):
        engine = self._make_engine(tmp_path, proximity_px=200)
        # oven centre (50,50), person centre (150,150) → L∞=100 ≤ 200 → triggers
        alerts = engine.evaluate([
            _det("oven", [0, 0, 100, 100]),
            _det("person", [100, 100, 200, 200]),
        ])
        assert len(alerts) == 1

    def test_no_trigger_when_far(self, tmp_path):
        engine = self._make_engine(tmp_path, proximity_px=50)
        # oven centre (50,50), person centre (500,500) → L∞=450 > 50 → no trigger
        assert engine.evaluate([
            _det("oven", [0, 0, 100, 100]),
            _det("person", [450, 450, 550, 550]),
        ]) == []

    def test_no_trigger_without_near_label_present(self, tmp_path):
        engine = self._make_engine(tmp_path, proximity_px=200)
        # oven present but no person
        assert engine.evaluate([_det("oven", [0, 0, 100, 100])]) == []


# ---------------------------------------------------------------------------
# HazardAlert.to_dict
# ---------------------------------------------------------------------------


class TestHazardAlertToDict:
    def test_serialises_all_fields(self, tmp_path):
        path = _write_rules(tmp_path, [{"name": "fire", "labels": ["fire"], "severity": "critical", "description": "Fire!"}])
        engine = HazardRuleEngine(config_path=path)
        alerts = engine.evaluate([_det("fire", [10.0, 20.0, 110.0, 120.0])])
        d = alerts[0].to_dict()
        assert d["name"] == "fire"
        assert d["severity"] == "critical"
        assert d["description"] == "Fire!"
        assert "detection" in d
        assert d["detection"]["label"] == "fire"
