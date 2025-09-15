import openml
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_datasets

_OPENML_COLUMNS = [
    "did",
    "name",
    "NumberOfInstances",
    "NumberOfFeatures",
    "NumberOfClasses",
    "version",
    "status",
]

def load_openml_dataset(dataset_id):
    try:
        dataset = openml.datasets.get_dataset(int(dataset_id))
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.concat([X, y], axis=1)
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def load_huggingface_dataset(name):
    try:
        dataset = load_dataset(name)
        # Prefer 'train' split when present
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
        return df.head(2000)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def _openml_defaults():
    return pd.DataFrame([
        {"did": 61, "name": "iris", "description": "Iris • 150 rows, 5 cols, classes: 3"},
        {"did": 37, "name": "diabetes", "description": "Diabetes • 442 rows, 11 cols, regression"},
        {"did": 24, "name": "mushroom", "description": "Mushroom • 8124 rows, 23 cols, classes: 2"},
        {"did": 40945, "name": "adult", "description": "Adult Income • 48842 rows, 15 cols, classes: 2"},
    ])


def list_openml_datasets(limit=300):
    try:
        df = openml.datasets.list_datasets(output_format="dataframe")
        keep = [c for c in _OPENML_COLUMNS if c in df.columns]
        df = df[keep].copy()
        if "NumberOfInstances" in df.columns:
            df = df.sort_values("NumberOfInstances", ascending=False)
        df = df.head(limit)
        def _desc(row):
            name = row.get("name", "dataset")
            n = row.get("NumberOfInstances", "?")
            p = row.get("NumberOfFeatures", "?")
            cls = row.get("NumberOfClasses", "?")
            return f"{name} • {n} rows, {p} cols, classes: {cls}"
        df["description"] = df.apply(_desc, axis=1)
        if df.empty:
            return _openml_defaults()
        return df
    except Exception:
        return _openml_defaults()


def _hf_defaults():
    return pd.DataFrame([
        {"id": "imdb", "name": "imdb", "description": "IMDB movie reviews • sentiment classification"},
        {"id": "ag_news", "name": "ag_news", "description": "AG News • topic classification"},
        {"id": "banking77", "name": "banking77", "description": "Banking77 • intent classification"},
    ])


def list_huggingface_datasets(limit=300):
    try:
        items = list_datasets(full=True, limit=limit)
        records = []
        for it in items:
            desc = ""
            try:
                if getattr(it, "cardData", None) and isinstance(it.cardData, dict):
                    desc = it.cardData.get("short_description") or it.cardData.get("summary") or ""
            except Exception:
                pass
            desc = desc or (getattr(it, "description", None) or "")
            if desc and len(desc) > 140:
                desc = desc[:137] + "..."
            records.append({
                "id": getattr(it, "id", ""),
                "name": getattr(it, "id", ""),
                "description": desc,
            })
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return _hf_defaults()
        return df
    except Exception:
        return _hf_defaults() 
