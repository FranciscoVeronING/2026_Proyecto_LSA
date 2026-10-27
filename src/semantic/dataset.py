"""Load the training pairs and the trial set."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from config import DEFAULT_DATASET, SPLIT_SEED, TEST_SIZE
from gloss import join_glosses


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_parallel_pairs(dataset_path: Path | None = None) -> list[dict[str, str]]:
    """Training pairs. Gloss tokens are kept as stored, then joined with spaces."""
    dataset_path = dataset_path or DEFAULT_DATASET
    raw = load_json(dataset_path)
    rows = raw.get("dataset", raw.get("examples", []))
    return [
        {
            "spanish": str(item["spanish"]).strip(),
            "glosses": join_glosses(item["glosses"]),
        }
        for item in rows
    ]


def split_pairs(
    pairs: list[dict[str, str]],
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[Any, Any]:
    from datasets import Dataset

    full = Dataset.from_list(pairs)
    split = full.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]


def references_by_spanish(pairs: list[dict[str, str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for item in pairs:
        gloss = item["glosses"]
        if gloss not in mapping[item["spanish"]]:
            mapping[item["spanish"]].append(gloss)
    return dict(mapping)


def load_trial_groups(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    groups: list[dict[str, Any]] = []
    for item in payload.get("test_dataset", payload.get("dataset", [])):
        spanish_list = item["spanish"]
        if isinstance(spanish_list, str):
            spanish_list = [spanish_list]
        groups.append(
            {
                "category": item.get("category", ""),
                "spanish": [str(s).strip() for s in spanish_list],
                "glosses": join_glosses(item["glosses"]),
            }
        )
    return groups


def load_trial_samples(path: Path) -> list[dict[str, Any]]:
    """One row per Spanish paraphrase."""
    samples: list[dict[str, Any]] = []
    for group in load_trial_groups(path):
        for spanish in group["spanish"]:
            samples.append(
                {
                    "category": group["category"],
                    "spanish": spanish,
                    "glosses": group["glosses"],
                }
            )
    return samples
