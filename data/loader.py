import json, random
from pathlib import Path
from .schema import EvalDataset

def load_dataset(path: str | Path, seed: int = 42,
                 max_examples: int | None = None) -> EvalDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ds = EvalDataset(**data)
    items = ds.dataset
    if max_examples is not None:
        rnd = random.Random(seed)
        rnd.shuffle(items)
        ds.dataset = items[:max_examples]
    return ds
