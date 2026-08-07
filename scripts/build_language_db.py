from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


ABSTRACT_CLASSES = {
    "noun.attribute",
    "noun.cognition",
    "noun.communication",
    "noun.event",
    "noun.feeling",
    "noun.motive",
    "noun.process",
    "noun.quantity",
    "noun.relation",
    "noun.state",
    "noun.time",
    "verb.cognition",
    "verb.communication",
    "verb.emotion",
    "verb.social",
}
CONCRETE_CLASSES = {
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.food",
    "noun.location",
    "noun.object",
    "noun.person",
    "noun.plant",
    "noun.substance",
}


SCHEMA = """
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE lexeme (
    lemma TEXT PRIMARY KEY,
    pos TEXT NOT NULL,
    sense_count INTEGER NOT NULL,
    lexfiles TEXT NOT NULL,
    definition TEXT NOT NULL,
    zipf_frequency REAL,
    abstraction REAL NOT NULL,
    derivation_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE derivation (
    base TEXT NOT NULL,
    derived TEXT NOT NULL,
    base_pos TEXT,
    derived_pos TEXT,
    affix TEXT NOT NULL,
    affix_type TEXT NOT NULL
);
CREATE INDEX derivation_derived_idx ON derivation(derived);
CREATE INDEX derivation_base_idx ON derivation(base);
CREATE TABLE inflection (
    lemma TEXT NOT NULL,
    form TEXT NOT NULL,
    features TEXT,
    segmentation TEXT
);
CREATE INDEX inflection_form_idx ON inflection(form);
"""


def abstraction_score(classes: set[str]) -> float:
    if not classes:
        return 50.0
    abstract = sum(item in ABSTRACT_CLASSES for item in classes)
    concrete = sum(item in CONCRETE_CLASSES for item in classes)
    if abstract + concrete == 0:
        return 50.0
    return round(100.0 * abstract / (abstract + concrete), 2)


def build(args: argparse.Namespace) -> None:
    try:
        import wn
        from wordfreq import zipf_frequency
    except ImportError as exc:
        raise SystemExit("Install the build-only packages `wn` and `wordfreq` first.") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    connection = sqlite3.connect(output)
    connection.executescript(SCHEMA)

    derivation_counts: dict[str, int] = defaultdict(int)
    with Path(args.derivational).open(encoding="utf-8") as handle:
        rows = []
        for line in handle:
            base, derived, base_pos, derived_pos, affix, affix_type = line.rstrip("\n").split("\t")
            rows.append((base.lower(), derived.lower(), base_pos, derived_pos, affix.lower(), affix_type))
            derivation_counts[base.lower()] += 1
            if len(rows) >= 20000:
                connection.executemany("INSERT INTO derivation VALUES (?,?,?,?,?,?)", rows)
                rows.clear()
        if rows:
            connection.executemany("INSERT INTO derivation VALUES (?,?,?,?,?,?)", rows)

    with Path(args.inflectional).open(encoding="utf-8") as handle:
        rows = []
        for line in handle:
            lemma, form, features, segmentation = line.rstrip("\n").split("\t")
            rows.append((lemma.lower(), form.lower(), features, segmentation.lower()))
            if len(rows) >= 30000:
                connection.executemany("INSERT INTO inflection VALUES (?,?,?,?)", rows)
                rows.clear()
        if rows:
            connection.executemany("INSERT INTO inflection VALUES (?,?,?,?)", rows)
    connection.commit()

    wordnet = wn.Wordnet(args.wordnet)
    aggregated: dict[str, dict] = {}
    for word in wordnet.words():
        lemma = word.lemma().lower().replace("_", " ")
        if not lemma or len(lemma) > 80:
            continue
        entry = aggregated.setdefault(
            lemma,
            {"pos": set(), "sense_count": 0, "lexfiles": set(), "definition": ""},
        )
        entry["pos"].add(word.pos)
        senses = word.senses()
        entry["sense_count"] += len(senses)
        for sense in senses:
            synset = sense.synset()
            entry["lexfiles"].add(synset.lexfile())
            if not entry["definition"]:
                entry["definition"] = synset.definition()

    rows = []
    for lemma, entry in aggregated.items():
        rows.append(
            (
                lemma,
                ",".join(sorted(entry["pos"])),
                entry["sense_count"],
                ",".join(sorted(entry["lexfiles"])),
                entry["definition"],
                round(float(zipf_frequency(lemma, "en")), 3),
                abstraction_score(entry["lexfiles"]),
                derivation_counts.get(lemma, 0),
            )
        )
        if len(rows) >= 10000:
            connection.executemany("INSERT INTO lexeme VALUES (?,?,?,?,?,?,?,?)", rows)
            rows.clear()
    if rows:
        connection.executemany("INSERT INTO lexeme VALUES (?,?,?,?,?,?,?,?)", rows)

    metadata = {
        "schema_version": "1",
        "built_at": args.built_at,
        "wordnet": "Open English WordNet 2025 (CC BY 4.0)",
        "morphynet": "MorphyNet 1.0",
        "wordfreq": "wordfreq 3.1 (frequency data CC BY-SA 4.0)",
        "lexeme_count": str(len(aggregated)),
        "derivation_count": str(sum(1 for _ in Path(args.derivational).open(encoding="utf-8"))),
        "inflection_count": str(sum(1 for _ in Path(args.inflectional).open(encoding="utf-8"))),
    }
    connection.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
    connection.commit()
    connection.execute("ANALYZE")
    connection.execute("VACUUM")
    connection.close()
    print(json.dumps(metadata, indent=2))
    print(f"Created {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivational", required=True)
    parser.add_argument("--inflectional", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wordnet", default="oewn:2025")
    parser.add_argument("--built-at", default="2026-07-19")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
