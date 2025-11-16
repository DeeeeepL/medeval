from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel

QuestionType = Literal["single_choice", "multi_choice", "open_response"]

class ScoringPoint(BaseModel):
    criterion: str
    points: int
    tags: List[str] = []

class Metadata(BaseModel):
    category1: Optional[str] = None
    category2: Optional[str] = None
    task: Optional[str] = None
    tags: List[str] = []
    type: QuestionType
    score: int = 1
    difficulity: Optional[str] = None
    prompt_template: Optional[str] = None
    dialogue: List[Dict[str, str]] = []
    synonyms: Dict[str, list] = {}
    positive_scoring_points: List[ScoringPoint] = []
    negative_scoring_points: List[ScoringPoint] = []
    source: Optional[str] = None

class Item(BaseModel):
    question_id: str
    question: str
    answer: Any
    options: List[str] = []
    metadata: Metadata
    multimodal_data: List[Dict[str, Any]] = []
    answer_pred: Optional[str] = None
    model: Optional[str] = None

class DatasetMetadata(BaseModel):
    duplicate: Optional[bool] = None
    duplicated: Optional[bool] = None  # 有的叫 duplicated
    allowed_keys: List[str] = []
    dataset_modal: str = "text"
    dataset_id: str
    dataset_name: str

class EvalDataset(BaseModel):
    dataset_metadata: DatasetMetadata
    dataset: List[Item]
