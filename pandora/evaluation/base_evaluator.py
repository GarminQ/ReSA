# base_evaluator.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class EvaluationData:
    """
    Unified input for all evaluators.
    """
    query: Optional[str] = None
    response: Optional[str] = None

@dataclass
class EvaluationResult:
    """
    Unified output for all evaluators.
    """
    score: float
    reason: Optional[str] = None


class BaseEvaluator(ABC):

    def __init__(self, **kwargs):
        """Unified constructor."""
        self.config = kwargs or {}

    @abstractmethod
    def evaluate(self, data: List[EvaluationData]) -> List[EvaluationResult]:
        """Unified evaluate method."""
        pass