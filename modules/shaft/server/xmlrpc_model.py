from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Step:
    status: int
    result: Optional[dict] = None


@dataclass
class StepsResult:
    steps: Dict[str, Step] = field(default_factory=dict)

    def set_step(self, step_name: str, status: int, result: Optional[dict] = None):
        self.steps[step_name] = Step(status=status, result=result)

    def get_step(self, step_name: str) -> Optional[Step]:
        return self.steps.get(step_name)
