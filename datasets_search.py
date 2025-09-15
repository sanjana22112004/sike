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
    "format",
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


def list_openml_datasets(limit=300):
    try:
        df = openml.datasets.list_datasets(output_format="dataframe")
        # Keep common fields and sort by instances desc
        df = df[_OPENML_COLUMNS + ["tag"]].copy()
        df = df.sort_values("NumberOfInstances", ascending=False).head(limit)
        # Build description
        def _desc(row):
            cls = row["NumberOfClasses"] if not pd.isna(row["NumberOfClasses"]) else "?"
            return f"{row['name']} • {row['NumberOfInstances']} rows, {row['NumberOfFeatures']} cols, classes: {cls}"
        df["description"] = df.apply(_desc, axis=1)
        return df
    except Exception:
        return pd.DataFrame(columns=["did", "name", "description"]) 


def list_huggingface_datasets(limit=300):
    try:
        items = list_datasets(full=True, limit=limit)
        records = []
        for it in items:
            # description may be long; truncate
            desc = (it.cardData.get("short_description") if getattr(it, "cardData", None) else None) or (it.description or "")
            if desc and len(desc) > 140:
                desc = desc[:137] + "..."
            records.append({
                "id": it.id,
                "name": it.id,
                "description": desc or "",
            })
        return pd.DataFrame.from_records(records)
    except Exception:
        return pd.DataFrame(columns=["id", "name", "description"]) 
