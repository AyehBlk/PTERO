"""PTERO provenance tracking."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

@dataclass
class PteroStep:
    function: str; params: dict = field(default_factory=dict); timestamp: str = ""
    duration_seconds: float = 0.0
    n_proteins_before: int = 0; n_proteins_after: int = 0
    n_samples_before: int = 0; n_samples_after: int = 0; notes: str = ""
    def __post_init__(self):
        if not self.timestamp: self.timestamp = datetime.now(timezone.utc).isoformat()
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d): return cls(**{k:v for k,v in d.items() if k in cls.__dataclass_fields__})

class ProvenanceTracker:
    def __init__(self): self._steps: list[PteroStep] = []
    @property
    def steps(self): return self._steps.copy()
    def __len__(self): return len(self._steps)
    def __getitem__(self, idx): return self._steps[idx]
    def __iter__(self): return iter(self._steps)
    def record(self, function, params, **kw):
        clean = {k: v.value if hasattr(v,'value') else (str(v) if not isinstance(v,(str,int,float,bool,type(None),list)) else v) for k,v in params.items()}
        step = PteroStep(function=function, params=clean, **kw)
        self._steps.append(step); return step
    def to_script(self, var="exp"):
        lines = ["import ptero\n"]
        for i, s in enumerate(self._steps):
            ps = ", ".join(f"{k}={v!r}" for k,v in s.params.items())
            if "io.from_" in s.function or "read" in s.function:
                lines.append(f"{var} = ptero.{s.function}({ps})")
            else:
                lines.append(f"ptero.{s.function}({var}, {ps})" if ps else f"ptero.{s.function}({var})")
        return "\n".join(lines)
    def summary(self):
        lines = ["Processing History:"]
        for i,s in enumerate(self._steps):
            lines.append(f"  {i+1}. {s.function} → {s.n_proteins_after}p × {s.n_samples_after}s")
        return "\n".join(lines)
    def to_json(self): return json.dumps([s.to_dict() for s in self._steps], indent=2)
    @classmethod
    def from_json(cls, j):
        t = cls()
        for d in json.loads(j): t._steps.append(PteroStep.from_dict(d))
        return t
