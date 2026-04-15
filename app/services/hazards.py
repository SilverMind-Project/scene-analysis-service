"""Hazard detection rule engine.

Applies a YAML-configured rule set to a list of :class:`Detection` objects
produced by the object detector.  Returns a list of :class:`HazardAlert`
instances for any triggered rules.

Design
------
All logic is pure: ``HazardRuleEngine.evaluate`` takes a list of detections
and returns a list of alerts.  No I/O, no side effects — trivially testable.

Proximity check
~~~~~~~~~~~~~~~
When a rule specifies ``near_labels``, the hazard fires only when a
triggering detection's bounding box is within ``proximity_px`` pixels (L∞
distance between box centres) of a detection whose label is in the
``near_labels`` list.

Aspect-ratio heuristic
~~~~~~~~~~~~~~~~~~~~~~
``aspect_ratio_min`` on a rule enables a simple fallen-person heuristic:
the detection is only flagged if ``bbox_width / bbox_height >= aspect_ratio_min``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.services.detector import Detection

logger = logging.getLogger(__name__)


@dataclass
class HazardAlert:
    """A hazard detected in a frame.

    Attributes:
        name: Rule name.
        severity: ``low | medium | high | critical``.
        description: Human-readable description.
        detection: The :class:`Detection` that triggered the rule.
    """

    name: str
    severity: str
    description: str
    detection: Detection

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "description": self.description,
            "detection": self.detection.to_dict(),
        }


class HazardRuleEngine:
    """Applies a YAML rule set to a list of :class:`Detection` objects.

    Args:
        config_path: Path to a ``hazards.yaml`` file.  If the file does not
            exist the engine loads with zero rules (no hazards ever fire).
    """

    def __init__(self, config_path: str | Path = "config/hazards.yaml") -> None:
        self._rules: list[dict[str, Any]] = []
        self._load(config_path)

    def _load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            logger.warning("hazards_config_not_found path=%s", p)
            return
        with p.open() as fh:
            data = yaml.safe_load(fh) or {}
        self._rules = data.get("hazards", [])
        logger.info("hazards_loaded count=%d path=%s", len(self._rules), p)

    # ------------------------------------------------------------------

    def evaluate(self, detections: list[Detection]) -> list[HazardAlert]:
        """Return a :class:`HazardAlert` for each rule triggered by *detections*."""
        alerts: list[HazardAlert] = []
        for rule in self._rules:
            triggered = self._check_rule(rule, detections)
            alerts.extend(triggered)
        return alerts

    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_centre(bbox: list[float]) -> tuple[float, float]:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    @staticmethod
    def _linf_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _check_rule(
        self, rule: dict[str, Any], detections: list[Detection]
    ) -> list[HazardAlert]:
        labels: list[str] = rule.get("labels", [])
        near_labels: list[str] = rule.get("near_labels", [])
        proximity_px: float = rule.get("proximity_px", 200)
        aspect_ratio_min: float | None = rule.get("aspect_ratio_min")
        severity: str = rule.get("severity", "low")
        name: str = rule.get("name", "unknown")
        description: str = rule.get("description", "")

        # Detections that match the primary labels
        primary = [d for d in detections if d.label in labels]

        alerts: list[HazardAlert] = []
        for det in primary:
            if not self._passes_aspect_ratio(det, aspect_ratio_min):
                continue
            if near_labels and not self._is_near(det, near_labels, detections, proximity_px):
                continue
            alerts.append(
                HazardAlert(
                    name=name,
                    severity=severity,
                    description=description,
                    detection=det,
                )
            )
        return alerts

    @staticmethod
    def _passes_aspect_ratio(det: Detection, min_ratio: float | None) -> bool:
        if min_ratio is None:
            return True
        x1, y1, x2, y2 = det.bbox
        h = y2 - y1
        if h <= 0:
            return False
        return (x2 - x1) / h >= min_ratio

    def _is_near(
        self,
        det: Detection,
        near_labels: list[str],
        all_detections: list[Detection],
        proximity_px: float,
    ) -> bool:
        centre = self._bbox_centre(det.bbox)
        for other in all_detections:
            if other is det:
                continue
            if other.label not in near_labels:
                continue
            if self._linf_distance(centre, self._bbox_centre(other.bbox)) <= proximity_px:
                return True
        return False
