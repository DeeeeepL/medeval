from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..data.schema import ScoringPoint

class Judge(ABC):
    @abstractmethod
    def score_single_choice(self,
                            gt_letters: List[str],
                            pred_letters: List[str],
                            total_score: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def score_open_response(self,
                            question: str,
                            positive_points: List[ScoringPoint],
                            negative_points: List[ScoringPoint],
                            student_answer: str,
                            total_score: int) -> Dict[str, Any]:
        ...
