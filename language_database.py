from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any


class LanguageDatabase:
    def __init__(self, path: Path):
        self.path = path

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def metadata(self) -> dict[str, str]:
        if not self.available:
            return {}
        with self._connect() as connection:
            return {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM metadata")}

    def _lexeme(self, connection: sqlite3.Connection, lemma: str) -> dict[str, Any] | None:
        row = connection.execute("SELECT * FROM lexeme WHERE lemma = ?", (lemma,)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["parts_of_speech"] = payload.pop("pos").split(",") if payload.get("pos") else []
        payload["lexical_classes"] = payload.pop("lexfiles").split(",") if payload.get("lexfiles") else []
        return payload

    def analyze(self, raw_word: str) -> dict[str, Any] | None:
        word = raw_word.lower()
        if not self.available:
            return None
        with self._connect() as connection:
            inflection = connection.execute(
                "SELECT lemma, features, segmentation FROM inflection WHERE form = ? "
                "ORDER BY CASE WHEN features LIKE 'N|%' THEN 0 ELSE 1 END LIMIT 1",
                (word,),
            ).fetchone()
            lemma = inflection["lemma"] if inflection else word
            derivation = connection.execute(
                "SELECT base, affix, affix_type, base_pos, derived_pos FROM derivation "
                "WHERE derived = ? ORDER BY length(affix) DESC LIMIT 1",
                (lemma,),
            ).fetchone()

            root = derivation["base"] if derivation else lemma
            lexeme = self._lexeme(connection, lemma) or self._lexeme(connection, root)
            root_lexeme = self._lexeme(connection, root)
            morphology = self._morphology_payload(word, lemma, inflection, derivation)
            if lexeme is None and not morphology:
                return None

            relation_count = int((root_lexeme or lexeme or {}).get("derivation_count", 0))
            productivity = min(100.0, 100.0 * math.log1p(relation_count) / math.log(41))
            metrics = {
                "zipf_frequency": (lexeme or {}).get("zipf_frequency"),
                "sense_count": int((lexeme or {}).get("sense_count", 0)),
                "abstraction": float((lexeme or {}).get("abstraction", 50.0)),
                "productivity": round(productivity, 2),
                "derivation_count": relation_count,
                "parts_of_speech": (lexeme or {}).get("parts_of_speech", []),
                "lexical_classes": (lexeme or {}).get("lexical_classes", []),
                "definition": (lexeme or {}).get("definition") or "",
            }
            return {
                "recognized": lexeme is not None,
                "lemma": lemma,
                "root": root,
                "morphology": morphology,
                "metrics": metrics,
                "sources": ["Open English WordNet 2025", "MorphyNet 1.0", "wordfreq 3.1"],
            }

    @staticmethod
    def _morphology_payload(
        word: str,
        lemma: str,
        inflection: sqlite3.Row | None,
        derivation: sqlite3.Row | None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if derivation:
            affix = derivation["affix"]
            base = derivation["base"]
            if derivation["affix_type"] == "prefix":
                items.extend(
                    [
                        {"text": affix, "kind": "prefix", "gloss": "MorphyNet derivational affix"},
                        {"text": base, "kind": "root", "gloss": "MorphyNet derivational base"},
                    ]
                )
            else:
                items.extend(
                    [
                        {"text": base, "kind": "root", "gloss": "MorphyNet derivational base"},
                        {"text": affix, "kind": "suffix", "gloss": "MorphyNet derivational affix"},
                    ]
                )
        else:
            items.append({"text": lemma, "kind": "root", "gloss": "dictionary lemma"})

        if inflection and word != lemma:
            segmentation = inflection["segmentation"] or ""
            ending = segmentation.rsplit("|", 1)[-1] if "|" in segmentation else word[len(lemma) :]
            if ending:
                items.append(
                    {
                        "text": ending,
                        "kind": "suffix",
                        "gloss": f"inflection: {inflection['features']}",
                    }
                )
        return items

    def status(self) -> dict[str, Any]:
        metadata = self.metadata()
        return {
            "available": self.available,
            "path": str(self.path),
            "metadata": metadata,
        }
