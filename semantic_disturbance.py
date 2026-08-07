from __future__ import annotations

import hashlib
import random
import re
import zlib
from typing import Any


WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
VOWELS = "aeiou"
NEIGHBOURS = {
    "a": "se", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg",
    "g": "fh", "h": "gj", "i": "uo", "j": "hk", "k": "jl", "l": "ko",
    "m": "nj", "n": "bm", "o": "ip", "p": "ol", "q": "wa", "r": "et",
    "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "xa",
}


def _bounded(value: Any, default: float, maximum: float = 1.0) -> float:
    try:
        return max(0.0, min(maximum, float(value)))
    except (TypeError, ValueError):
        return default


def _audio_signature(frames: list[dict[str, Any]]) -> str:
    if not frames:
        return "silent"
    step = max(1, len(frames) // 96)
    parts = []
    for frame in frames[::step][:96]:
        parts.append(
            ":".join(
                str(int(round(float(frame.get(key, 0)))))
                for key in ("energy", "peak", "brightness", "noise", "impact")
            )
        )
    return "|".join(parts)


def _weighted_word_indices(
    words: list[re.Match[str]],
    frames: list[dict[str, Any]],
    target: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if not words or target <= 0:
        return []
    impacts = [
        (index, float(frame.get("impact", 0)) + float(frame.get("peak", 0)) * 0.25)
        for index, frame in enumerate(frames)
        if float(frame.get("impact", 0)) > 0
    ]
    impacts.sort(key=lambda item: item[1], reverse=True)
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for frame_index, _ in impacts:
        word_index = min(
            len(words) - 1,
            int(round(frame_index / max(1, len(frames) - 1) * (len(words) - 1))),
        )
        if word_index in used:
            continue
        used.add(word_index)
        selected.append((word_index, frame_index))
        if len(selected) >= target:
            return selected

    remaining = [index for index in range(len(words)) if index not in used]
    rng.shuffle(remaining)
    for word_index in remaining[: max(0, target - len(selected))]:
        frame_index = int(round(word_index / max(1, len(words) - 1) * max(0, len(frames) - 1)))
        selected.append((word_index, frame_index))
    return selected


def _mutation_operation(frame: dict[str, Any], mode: str, rng: random.Random) -> str:
    brightness = float(frame.get("brightness", 0))
    noise = float(frame.get("noise", 0))
    peak = float(frame.get("peak", 0))
    ordered = [
        (brightness, "transpose"),
        (noise, "substitute"),
        (peak, "repeat"),
    ]
    operation = max(ordered, key=lambda item: item[0])[1]
    if mode == "breakdown" and rng.random() < 0.34:
        return "delete"
    return operation


def _mutate_word(word: str, operation: str, rng: random.Random) -> tuple[str, str]:
    if len(word) < 3:
        operation = "substitute"
    lower = word.lower()
    internal = list(range(1, len(word) - 1)) if len(word) > 3 else list(range(len(word)))
    index = rng.choice(internal)
    mutated = list(word)

    if operation == "transpose" and len(word) > 3:
        next_index = min(len(word) - 2, index + 1)
        if next_index == index:
            next_index = max(1, index - 1)
        mutated[index], mutated[next_index] = mutated[next_index], mutated[index]
    elif operation == "delete" and len(word) > 4:
        del mutated[index]
    elif operation == "repeat":
        mutated.insert(index, mutated[index])
    else:
        source = lower[index]
        choices = NEIGHBOURS.get(source, VOWELS if source in VOWELS else "rstln")
        replacement = rng.choice(choices)
        if word[index].isupper():
            replacement = replacement.upper()
        mutated[index] = replacement
        operation = "substitute"

    replacement = "".join(mutated)
    if replacement == word:
        replacement = word[:index] + ("e" if lower[index] != "e" else "a") + word[index + 1 :]
        operation = "substitute"
    return replacement, operation


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def disturb_text(
    text: str,
    frames: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    mode = str(config.get("semantic_mode", "trace")).strip().lower()
    if mode not in {"preserve", "trace", "breakdown"}:
        mode = "trace"
    default_loss = {"preserve": 0.0, "trace": 0.06, "breakdown": 0.60}[mode]
    requested_loss = _bounded(config.get("semantic_loss"), default_loss, 0.80)
    if mode == "preserve":
        requested_loss = 0.0

    words = list(WORD_PATTERN.finditer(text))
    first_line_end = text.find("\n")
    first_line = text[:first_line_end] if first_line_end >= 0 else ""
    protect_first_line = bool(config.get("protect_first_line", False))
    protected_until = (
        first_line_end + 1
        if first_line_end >= 0
        and (
            protect_first_line
            or " – " in first_line
            or " — " in first_line
            or re.search(r"\bby\b", first_line, re.I)
        )
        else 0
    )
    candidates = [
        match
        for match in words
        if len(match.group(0)) >= 3 and match.start() >= protected_until
    ]
    target = 0 if requested_loss <= 0 or not candidates else max(1, round(len(candidates) * requested_loss))
    seed_material = f"{text}|{mode}|{requested_loss:.4f}|{_audio_signature(frames)}|SLD-0.1"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    candidate_lookup = {match.start(): index for index, match in enumerate(words) if match in candidates}
    candidate_words = [match for match in words if match in candidates]
    selected = _weighted_word_indices(candidate_words, frames, target, rng)
    replacements: dict[int, tuple[str, str, int]] = {}
    for candidate_index, frame_index in selected:
        match = candidate_words[candidate_index]
        frame = frames[min(len(frames) - 1, frame_index)] if frames else {}
        operation = _mutation_operation(frame, mode, rng)
        replacement, applied = _mutate_word(match.group(0), operation, rng)
        operations = [applied]
        if mode == "breakdown":
            severity = max(
                float(frame.get("energy", 0)),
                float(frame.get("peak", 0)),
                float(frame.get("noise", 0)),
                float(frame.get("impact", 0)),
            )
            extra_passes = 1 + int(severity >= 8) + int(severity >= 13)
            for _ in range(extra_passes):
                next_operation = rng.choice(("substitute", "transpose", "repeat", "delete"))
                replacement, applied = _mutate_word(replacement, next_operation, rng)
                operations.append(applied)
        replacements[match.start()] = (replacement, "+".join(operations), frame_index)

    pieces: list[str] = []
    cursor = 0
    mutations = []
    for match in words:
        pieces.append(text[cursor : match.start()])
        replacement_record = replacements.get(match.start())
        if replacement_record is None:
            pieces.append(match.group(0))
        else:
            replacement, operation, frame_index = replacement_record
            pieces.append(replacement)
            mutations.append(
                {
                    "word_index": candidate_lookup.get(match.start(), -1),
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(0),
                    "replacement": replacement,
                    "operation": operation,
                    "audio_frame": frame_index,
                }
            )
        cursor = match.end()
    pieces.append(text[cursor:])
    disturbed = "".join(pieces)
    distance = _edit_distance(text, disturbed)
    fidelity = 100.0 * (1.0 - distance / max(1, len(text)))
    return {
        "version": "SLD-0.1",
        "mode": mode,
        "requested_loss": round(requested_loss * 100.0, 2),
        "source_text": text,
        "disturbed_text": disturbed,
        "source_crc32": zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF,
        "disturbed_crc32": zlib.crc32(disturbed.encode("utf-8")) & 0xFFFFFFFF,
        "source_altered": disturbed != text,
        "fidelity": round(max(0.0, fidelity), 2),
        "affected_words": len(mutations),
        "protected_prefix_characters": protected_until,
        "edit_distance": distance,
        "mutations": mutations,
        "seed": seed,
    }
