from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from language_database import LanguageDatabase
from leaf_codec import encode_text


LETTER_FREQUENCY = {
    "A": 8.167,
    "B": 1.492,
    "C": 2.782,
    "D": 4.253,
    "E": 12.702,
    "F": 2.228,
    "G": 2.015,
    "H": 6.094,
    "I": 6.966,
    "J": 0.153,
    "K": 0.772,
    "L": 4.025,
    "M": 2.406,
    "N": 6.749,
    "O": 7.507,
    "P": 1.929,
    "Q": 0.095,
    "R": 5.987,
    "S": 6.327,
    "T": 9.056,
    "U": 2.758,
    "V": 0.978,
    "W": 2.360,
    "X": 0.150,
    "Y": 1.974,
    "Z": 0.074,
}


# These features describe uppercase Latin letterforms in a deliberately simplified,
# editable model. They are design parameters, not claims about a universal typeface.
LETTER_FORM = {
    "A": (3, 0, 2, 1, 1),
    "B": (1, 2, 2, 2, 0),
    "C": (0, 1, 0, 0, 1),
    "D": (1, 1, 2, 1, 0),
    "E": (4, 0, 3, 0, 1),
    "F": (3, 0, 2, 0, 0),
    "G": (1, 1, 1, 0, 0),
    "H": (3, 0, 2, 0, 2),
    "I": (1, 0, 0, 0, 2),
    "J": (1, 1, 0, 0, 0),
    "K": (3, 0, 1, 0, 0),
    "L": (2, 0, 1, 0, 0),
    "M": (4, 0, 3, 0, 1),
    "N": (3, 0, 2, 0, 0),
    "O": (0, 1, 0, 1, 2),
    "P": (1, 1, 1, 1, 0),
    "Q": (1, 1, 1, 1, 0),
    "R": (2, 1, 2, 1, 0),
    "S": (0, 2, 0, 0, 0),
    "T": (2, 0, 1, 0, 1),
    "U": (2, 1, 0, 0, 1),
    "V": (2, 0, 1, 0, 1),
    "W": (4, 0, 3, 0, 1),
    "X": (2, 0, 1, 0, 2),
    "Y": (3, 0, 1, 0, 1),
    "Z": (3, 0, 2, 0, 0),
}


PREFIXES = {
    "counter": (72, "opposition"),
    "under": (45, "lower/below"),
    "inter": (64, "between"),
    "trans": (68, "across/change"),
    "super": (58, "above/excess"),
    "anti": (66, "against"),
    "post": (46, "after"),
    "pre": (38, "before"),
    "sub": (44, "under"),
    "over": (52, "excess/above"),
    "non": (48, "absence"),
    "dis": (57, "apart/negation"),
    "mis": (54, "wrongly"),
    "un": (52, "negation/reversal"),
    "re": (40, "again/back"),
    "co": (36, "together"),
    "de": (48, "down/reversal"),
}


SUFFIXES = {
    "ization": (76, "process of making"),
    "ation": (64, "process/result"),
    "ition": (64, "process/result"),
    "ology": (74, "field of study"),
    "escence": (72, "process/state"),
    "fulness": (67, "state/quality"),
    "lessness": (70, "absence/state"),
    "ability": (68, "capacity"),
    "ibility": (68, "capacity"),
    "ment": (54, "result/process"),
    "ness": (56, "state/quality"),
    "able": (48, "capacity"),
    "ible": (48, "capacity"),
    "ance": (55, "state/action"),
    "ence": (55, "state/action"),
    "tion": (61, "process/result"),
    "sion": (61, "process/result"),
    "ity": (58, "state/quality"),
    "ive": (46, "tendency/quality"),
    "ous": (45, "quality"),
    "al": (39, "related to"),
    "ing": (40, "ongoing action"),
    "ed": (34, "completed action"),
    "er": (35, "agent/comparison"),
    "est": (42, "superlative"),
    "ly": (32, "manner"),
    "ion": (57, "process/result"),
    "s": (20, "plural/third person"),
}


@dataclass(frozen=True)
class Morpheme:
    text: str
    kind: str
    semantic_score: float | None
    gloss: str
    source: str


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _load_lexicon(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _letter_profiles() -> dict[str, dict[str, Any]]:
    information = {letter: -math.log2(freq / 100.0) for letter, freq in LETTER_FREQUENCY.items()}
    minimum, maximum = min(information.values()), max(information.values())
    profiles: dict[str, dict[str, Any]] = {}

    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        strokes, curves, junctions, enclosures, symmetry = LETTER_FORM[letter]
        information_score = 100.0 * (information[letter] - minimum) / (maximum - minimum)
        form_score = _clamp(
            strokes * 12.0
            + curves * 13.0
            + junctions * 11.0
            + enclosures * 14.0
            - symmetry * 4.0
        )
        combined = 0.65 * information_score + 0.35 * form_score
        profiles[letter] = {
            "letter": letter,
            "frequency_percent": LETTER_FREQUENCY[letter],
            "information_bits": round(information[letter], 4),
            "information_complexity": round(information_score, 2),
            "form_complexity": round(form_score, 2),
            "letter_complexity": round(combined, 2),
            "form_features": {
                "strokes": strokes,
                "curves": curves,
                "junctions": junctions,
                "enclosures": enclosures,
                "symmetry_axes": symmetry,
            },
            "topology": topology_from_score(combined, enclosures, curves),
        }
    return profiles


def topology_from_score(score: float, enclosures: int = 0, curves: int = 0) -> dict[str, Any]:
    return {
        "target_leaf_quantile": round(score / 100.0, 3),
        "branch_order": 1 + min(3, int(score // 25)),
        "side_branch_count": 2 + min(6, int(round(score / 16.0))),
        "loop_tendency": min(3, enclosures + int(score >= 82)),
        "curvature": round(_clamp(curves * 0.30 + score / 220.0, 0.08, 0.95), 3),
        "node_density": round(0.15 + score * 0.008, 3),
    }


LETTER_PROFILES = _letter_profiles()


def _prefix_options(word: str) -> list[tuple[list[Morpheme], str]]:
    options: list[tuple[list[Morpheme], str]] = [([], word)]
    frontier = [([], word)]
    for _ in range(2):
        next_frontier: list[tuple[list[Morpheme], str]] = []
        for found, remaining in frontier:
            for prefix in sorted(PREFIXES, key=len, reverse=True):
                if remaining.startswith(prefix) and len(remaining) - len(prefix) >= 3:
                    score, gloss = PREFIXES[prefix]
                    candidate = (
                        found + [Morpheme(prefix, "prefix", score, gloss, "affix_rules")],
                        remaining[len(prefix) :],
                    )
                    options.append(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return options


def _suffix_options(word: str) -> list[tuple[str, list[Morpheme]]]:
    options: list[tuple[str, list[Morpheme]]] = [(word, [])]
    frontier = [(word, [])]
    for _ in range(2):
        next_frontier: list[tuple[str, list[Morpheme]]] = []
        for remaining, found in frontier:
            for suffix in sorted(SUFFIXES, key=len, reverse=True):
                if remaining.endswith(suffix) and len(remaining) - len(suffix) >= 3:
                    score, gloss = SUFFIXES[suffix]
                    candidate = (
                        remaining[: -len(suffix)],
                        [Morpheme(suffix, "suffix", score, gloss, "affix_rules")] + found,
                    )
                    options.append(candidate)
                    next_frontier.append(candidate)
        frontier = next_frontier
    return options


def segment_word(word: str, lexicon: dict[str, dict[str, Any]]) -> list[Morpheme]:
    candidates: list[tuple[float, list[Morpheme], str, list[Morpheme]]] = []
    for prefixes, remaining in _prefix_options(word):
        for root, suffixes in _suffix_options(remaining):
            if len(root) < 3:
                continue
            affix_count = len(prefixes) + len(suffixes)
            known_bonus = 1000.0 if root in lexicon else 0.0
            paired_affix_bonus = 30.0 if prefixes and suffixes else affix_count * 8.0
            conservative_penalty = 0.0 if root in lexicon else affix_count * 2.0
            score = known_bonus + len(root) * 2.0 + paired_affix_bonus - conservative_penalty
            candidates.append((score, prefixes, root, suffixes))

    if not candidates:
        candidates = [(0.0, [], word, [])]
    _, prefixes, root, suffixes = max(candidates, key=lambda item: item[0])

    root_entry = lexicon.get(root)
    if root_entry is None:
        semantic_score = None
        gloss = "unknown root; structural fallback"
        source = "estimated"
    else:
        semantic_score = float(root_entry["score"])
        gloss = str(root_entry.get("gloss", "curated root"))
        source = "root_lexicon"

    return prefixes + [Morpheme(root, "root", semantic_score, gloss, source)] + suffixes


def _external_semantic_score(metrics: dict[str, Any]) -> float:
    zipf = metrics.get("zipf_frequency")
    rarity = _clamp((7.2 - float(zipf)) * 18.0) if zipf is not None else 50.0
    polysemy = _clamp(100.0 * math.log1p(int(metrics.get("sense_count", 0))) / math.log(13))
    abstraction = float(metrics.get("abstraction", 50.0))
    productivity = float(metrics.get("productivity", 0.0))
    return _clamp(0.35 * rarity + 0.30 * abstraction + 0.20 * polysemy + 0.15 * productivity)


def _database_morphemes(
    word: str,
    lexicon: dict[str, dict[str, Any]],
    analysis: dict[str, Any] | None,
) -> list[Morpheme] | None:
    if not analysis or not analysis.get("morphology"):
        return None
    root_name = str(analysis.get("root", word))
    override = lexicon.get(root_name)
    if override:
        root_score = float(override["score"])
        root_gloss = str(override.get("gloss", "curated root"))
        root_source = "artistic override"
    else:
        root_score = _external_semantic_score(analysis["metrics"])
        root_gloss = str(analysis["metrics"].get("definition") or "dictionary lemma")
        root_source = "Open English WordNet + MorphyNet"

    result = []
    for item in analysis["morphology"]:
        kind = str(item["kind"])
        text = re.sub(r"[^a-z]", "", str(item["text"]).lower())
        if not text:
            continue
        if kind == "root":
            result.append(Morpheme(text, kind, root_score, root_gloss, root_source))
        else:
            rules = PREFIXES if kind == "prefix" else SUFFIXES
            score, gloss = rules.get(text, (None, str(item.get("gloss", "MorphyNet affix"))))
            result.append(Morpheme(text, kind, score, gloss, "MorphyNet"))
    return result or None


def _morphological_score(word: str, morphemes: list[Morpheme]) -> float:
    morpheme_load = min(1.0, max(0, len(morphemes) - 1) / 3.0)
    length_load = min(1.0, max(0, len(word) - 3) / 12.0)
    return 100.0 * (0.68 * morpheme_load + 0.32 * length_load)


def _semantic_score(morphemes: list[Morpheme], letter_mean: float) -> tuple[float, float]:
    scored = [m.semantic_score for m in morphemes if m.semantic_score is not None]
    root_known = any(m.kind == "root" and m.semantic_score is not None for m in morphemes)
    if not scored:
        return 0.55 * letter_mean + 22.5, 0.35
    weights = [1.4 if m.kind == "root" else 0.7 for m in morphemes if m.semantic_score is not None]
    weighted = sum(score * weight for score, weight in zip(scored, weights)) / sum(weights)
    return weighted, 0.90 if root_known else 0.62


def _spine_layout(morphemes: list[Morpheme]) -> list[dict[str, Any]]:
    weights = [max(1.0, len(m.text)) * (1.45 if m.kind == "root" else 0.85) for m in morphemes]
    total = sum(weights)
    cursor = 0.06
    usable = 0.88
    output = []
    for morpheme, weight in zip(morphemes, weights):
        span = usable * weight / total
        behavior = {
            "prefix": "basal_modifier",
            "root": "primary_crown",
            "suffix": "apical_resolution",
        }[morpheme.kind]
        output.append(
            {
                "text": morpheme.text,
                "kind": morpheme.kind,
                "start": round(cursor, 4),
                "end": round(cursor + span, 4),
                "anchor": round(cursor + span / 2.0, 4),
                "growth_behavior": behavior,
            }
        )
        cursor += span
    return output


def analyze_word(raw_word: str, lexicon_path: Path, database_path: Path | None = None) -> dict[str, Any]:
    word = re.sub(r"[^a-z]", "", raw_word.lower())
    if not word:
        raise ValueError("Enter at least one English letter.")
    if len(word) > 40:
        raise ValueError("This version supports words of up to 40 letters.")

    lexicon = _load_lexicon(lexicon_path)
    database = LanguageDatabase(database_path) if database_path else None
    database_analysis = database.analyze(word) if database and database.available else None
    morphemes = _database_morphemes(word, lexicon, database_analysis) or segment_word(word, lexicon)
    letters = [LETTER_PROFILES[letter] for letter in word.upper()]
    information_score = sum(item["information_complexity"] for item in letters) / len(letters)
    form_score = sum(item["form_complexity"] for item in letters) / len(letters)
    letter_score = sum(item["letter_complexity"] for item in letters) / len(letters)
    morphology_score = _morphological_score(word, morphemes)
    semantic_score, confidence = _semantic_score(morphemes, letter_score)

    composite = (
        0.25 * information_score
        + 0.20 * form_score
        + 0.25 * morphology_score
        + 0.30 * semantic_score
    )

    morpheme_payload = []
    for morpheme in morphemes:
        local_letters = [LETTER_PROFILES[letter] for letter in morpheme.text.upper()]
        local_letter_score = sum(item["letter_complexity"] for item in local_letters) / len(local_letters)
        resolved_semantic = (
            morpheme.semantic_score
            if morpheme.semantic_score is not None
            else 0.55 * local_letter_score + 22.5
        )
        local_score = 0.55 * local_letter_score + 0.45 * resolved_semantic
        morpheme_payload.append(
            {
                "text": morpheme.text,
                "kind": morpheme.kind,
                "gloss": morpheme.gloss,
                "semantic_score": (
                    round(morpheme.semantic_score, 2)
                    if morpheme.semantic_score is not None
                    else None
                ),
                "semantic_source": morpheme.source,
                "structural_score": round(local_letter_score, 2),
                "target_complexity": round(local_score, 2),
                "topology": topology_from_score(local_score),
            }
        )

    return {
        "input": raw_word,
        "normalized_word": word,
        "model_version": "language-mapping-mvp-0.3",
        "scores": {
            "information_complexity": round(information_score, 2),
            "form_complexity": round(form_score, 2),
            "morphological_complexity": round(morphology_score, 2),
            "semantic_complexity": round(semantic_score, 2),
            "composite_complexity": round(composite, 2),
            "semantic_confidence": round(confidence, 2),
        },
        "morphemes": morpheme_payload,
        "letters": letters,
        "spine_layout": _spine_layout(morphemes),
        "lexical_profile": database_analysis
        or {
            "recognized": False,
            "lemma": word,
            "root": next((item.text for item in morphemes if item.kind == "root"), word),
            "metrics": {},
            "sources": ["artistic root overrides", "structural fallback"],
        },
        "codec": encode_text(raw_word),
        "generation_target": {
            "main_vein_role": "reading_order_and_morphological_spine",
            "word_target_leaf_quantile": round(composite / 100.0, 3),
            "branch_order": 1 + min(4, int(composite // 22)),
            "global_node_density": round(0.16 + composite * 0.0085, 3),
            "reading_direction": "petiole_to_tip",
        },
        "warnings": [
            "Semantic scores are artistic hypotheses, not objective linguistic facts.",
            "The reversible code is independent from morphology, so unknown words remain decodable.",
        ],
    }
