"""TranscriptIssue and risk scoring data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class RiskScore:
    """Multi-component risk assessment for a transcript span."""
    agreement_risk: float = 0.0      # 0.0–1.0
    confidence_risk: float = 0.0
    timing_risk: float = 0.0
    repetition_risk: float = 0.0
    acoustic_risk: float = 0.0
    semantic_risk: float = 0.0       # 0 if no LLM review

    # Configurable weights for combining
    WEIGHTS = {
        "agreement": 1.0,
        "confidence": 0.8,
        "timing": 0.6,
        "acoustic": 0.5,
        "semantic": 0.7,
    }

    @property
    def overall(self) -> float:
        """Weighted combination of non-zero component risks."""
        w = self.WEIGHTS
        components = {
            "agreement": self.agreement_risk,
            "confidence": self.confidence_risk,
            "timing": self.timing_risk,
            "acoustic": self.acoustic_risk,
            "semantic": self.semantic_risk,
        }
        # Only consider components that have a non-zero risk value
        active_weight = 0.0
        weighted_sum = 0.0
        for key, risk in components.items():
            if risk > 0:
                active_weight += w.get(key, 0.5)
                weighted_sum += risk * w.get(key, 0.5)
        if active_weight == 0:
            return 0.0
        return min(1.0, weighted_sum / active_weight)

    @property
    def severity(self) -> str:
        """Map overall risk to severity label."""
        o = self.overall
        if o >= 0.7:
            return "high"
        if o >= 0.4:
            return "medium"
        return "low"

    def components(self) -> dict[str, float]:
        """Return non-zero components with names."""
        return {
            "agreement": self.agreement_risk,
            "confidence": self.confidence_risk,
            "timing": self.timing_risk,
            "repetition": self.repetition_risk,
            "acoustic": self.acoustic_risk,
            "semantic": self.semantic_risk,
        }

    def explanation(self) -> str:
        """Human-readable explanation of the risk."""
        parts = []
        comps = self.components()
        names = {
            "agreement": "Ensemble disagreement",
            "confidence": "Low ASR confidence",
            "timing": "Timing anomaly",
            "repetition": "Repetition detected",
            "acoustic": "Acoustic mismatch",
            "semantic": "LLM flagged as inconsistent",
        }
        for key, val in comps.items():
            if val > 0.3:
                parts.append(f"• {names.get(key, key)} ({val:.0%})")
        if not parts:
            return "No significant risk factors"
        return "\n".join(parts)


@dataclass
class TranscriptIssue:
    """A flagged issue in the transcript."""
    id: str                        # "issue_001"
    category: str                  # agreement|repetition|timing|semantic|acoustic
    severity: str                  # low|medium|high
    confidence: float              # 0.0–1.0
    start: float                   # seconds
    end: float                     # seconds
    word_ids: list[str] = field(default_factory=list)
    original_text: str = ""
    suggested_text: Optional[str] = None
    explanation: str = ""
    sources: list[str] = field(default_factory=list)
    risk: Optional[RiskScore] = None
    status: str = "open"           # open|accepted|edited|dismissed

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
