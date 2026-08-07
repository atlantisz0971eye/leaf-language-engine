from __future__ import annotations

import hashlib
import json
import re
from statistics import median
from pathlib import Path
from typing import Any

from language_engine import analyze_word
from branch_codec import encode_branch_protocol, encode_lexical_payload
from leaf_codec import (
    _crc10,
    decode_packet,
    encode_text,
    encode_token,
)


TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE
)
WORD_PATTERN = re.compile(r"^[A-Za-z]+(?:['-][A-Za-z]+)*$")
NUMBER_PATTERN = re.compile(r"^\d+(?:[.,]\d+)*$")
TERMINAL_PATTERN = re.compile(r"[.!?]+$")
CLAUSE_PUNCTUATION = {",": "minor", ";": "major", ":": "major", "—": "minor", "–": "minor"}
PUNCTUATION_MARKERS = {
    ".": {"code": "DOT", "topology": "one closed ring"},
    ",": {"code": "COMMA", "topology": "one closed ring with one trailing spur"},
    ":": {"code": "COLON", "topology": "two closed rings"},
    ";": {"code": "SEMICOLON", "topology": "two closed rings with one trailing spur"},
    "?": {"code": "QUESTION", "topology": "one open hook above one closed ring"},
    "!": {"code": "EXCLAMATION", "topology": "one transverse bar above one closed ring"},
    "—": {"code": "EM_DASH", "topology": "two parallel transverse bars"},
    "–": {"code": "EN_DASH", "topology": "two parallel transverse bars"},
}
ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "e.g.", "i.e.", "etc.", "vs.", "fig.", "no.",
)


def _canonical_paragraph(raw: str) -> tuple[str, list[str]]:
    tokens = TOKEN_PATTERN.findall(raw)
    if not tokens:
        return "", []
    output = ""
    no_space_before = set(".,!?;:%)]}”’")
    no_space_after = set("([{“‘")
    for token in tokens:
        if not output:
            output = token
        elif token in no_space_before or output[-1] in no_space_after:
            output += token
        else:
            output += " " + token
    return output, tokens


def _split_sentences(raw: str) -> list[str]:
    protected = raw
    sentinel = "\uf000"
    for abbreviation in ABBREVIATIONS:
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group(0).replace(".", sentinel),
            protected,
            flags=re.IGNORECASE,
        )
    protected = re.sub(r"(?<=\d)\.(?=\d)", sentinel, protected)
    chunks = re.findall(r".*?(?:[.!?]+(?=\s|$)|$)", protected, flags=re.DOTALL)
    return [chunk.replace(sentinel, ".").strip() for chunk in chunks if chunk.strip()]


def _style_profile(metrics: dict[str, Any] | None, filename: str | None) -> dict[str, Any]:
    if not metrics:
        return {
            "source": "generic",
            "sample": None,
            "complexity": 0.52,
            "junction_density": 0.48,
            "mesh_density": 0.42,
            "endpoint_density": 0.52,
            "orientation_diversity": 0.72,
            "aspect_ratio": 0.58,
        }

    def bounded(value: float, ceiling: float) -> float:
        return round(max(0.0, min(1.0, float(value) / ceiling)), 4)

    width = max(1.0, float(metrics.get("original_width", 1)))
    height = max(1.0, float(metrics.get("original_height", 1)))
    return {
        "source": "analyzed specimen",
        "sample": filename,
        "complexity": bounded(metrics.get("experimental_complexity", 50), 100),
        "junction_density": bounded(metrics.get("junction_density_per_100k", 90), 180),
        "mesh_density": bounded(metrics.get("mesh_density_per_100k", 35), 70),
        "endpoint_density": bounded(metrics.get("endpoint_density_per_100k", 110), 220),
        "orientation_diversity": bounded(metrics.get("orientation_entropy", .7), 1),
        "aspect_ratio": round(min(1.2, width / height), 4),
        "raw_metrics": metrics,
    }


def _graph_capacity_profile(
    graph: dict[str, Any] | None, header_fraction: float = 1 / 3
) -> dict[str, Any]:
    if not graph or not graph.get("edges") or not graph.get("main_path", {}).get("node_ids"):
        return {
            "header_length": 4200.0,
            "payload_length": 8400.0,
            "median_width": 2.0,
            "edge_count": 0,
            "tree_edge_count": 0,
            "island_edge_count": 0,
            "fingerprint": hashlib.sha256(b"generic-leaf").hexdigest()[:16],
        }

    nodes = {int(node["id"]): node for node in graph["nodes"]}
    edges = {int(edge["id"]): edge for edge in graph["edges"]}
    incident: dict[int, list[dict[str, Any]]] = {node_id: [] for node_id in nodes}
    for edge in edges.values():
        incident[int(edge["source"])].append(edge)
        incident[int(edge["target"])].append(edge)
    main_node_ids = [int(item) for item in graph["main_path"]["node_ids"]]
    main_edge_ids = [int(item) for item in graph["main_path"]["edge_ids"]]
    main_nodes = set(main_node_ids)
    main_edges = set(main_edge_ids)
    visited: set[int] = set()
    lengths = {"header": 0.0, "payload": 0.0}
    tree_edge_count = 0

    def zone(position: float) -> str:
        return "header" if position <= header_fraction else "payload"

    def walk_subtree(node_id: int, position: float) -> None:
        nonlocal tree_edge_count
        choices = sorted(
            (
                edge
                for edge in incident.get(node_id, [])
                if int(edge["id"]) not in visited and int(edge["id"]) not in main_edges
            ),
            key=lambda edge: (
                int(edge.get("order", 6)),
                -float(edge.get("mean_width_px", 1)),
                int(edge["id"]),
            ),
        )
        for edge in choices:
            edge_id = int(edge["id"])
            visited.add(edge_id)
            tree_edge_count += 1
            lengths[zone(position)] += float(edge.get("length_px", 0)) * 2.0
            next_node = (
                int(edge["target"]) if int(edge["source"]) == node_id else int(edge["source"])
            )
            if next_node not in main_nodes:
                walk_subtree(next_node, position)

    denominator = max(1, len(main_node_ids) - 1)
    for index, node_id in enumerate(main_node_ids):
        position = index / denominator
        walk_subtree(node_id, position)
        if index < len(main_edge_ids):
            edge_id = main_edge_ids[index]
            edge = edges.get(edge_id)
            if edge and edge_id not in visited:
                visited.add(edge_id)
                tree_edge_count += 1
                lengths[zone((index + 0.5) / denominator)] += float(
                    edge.get("length_px", 0)
                )

    start = nodes[main_node_ids[0]]
    end = nodes[main_node_ids[-1]]
    axis_x = float(end["x"]) - float(start["x"])
    axis_y = float(end["y"]) - float(start["y"])
    axis_length_squared = max(1.0, axis_x * axis_x + axis_y * axis_y)
    island_edge_count = 0
    for edge_id, edge in edges.items():
        if edge_id in visited:
            continue
        island_edge_count += 1
        source = nodes[int(edge["source"])]
        target = nodes[int(edge["target"])]
        midpoint_x = (float(source["x"]) + float(target["x"])) / 2.0
        midpoint_y = (float(source["y"]) + float(target["y"])) / 2.0
        projection = (
            (midpoint_x - float(start["x"])) * axis_x
            + (midpoint_y - float(start["y"])) * axis_y
        ) / axis_length_squared
        lengths[zone(max(0.0, min(1.0, projection)))] += float(
            edge.get("length_px", 0)
        )

    fingerprint_data = {
        "main": graph["main_path"],
        "edges": [
            [
                int(edge["source"]),
                int(edge["target"]),
                int(edge.get("order", 6)),
                round(float(edge.get("length_px", 0)), 1),
            ]
            for edge in graph["edges"]
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    widths = [float(edge.get("mean_width_px", 1)) for edge in graph["edges"]]
    return {
        "header_length": round(lengths["header"], 2),
        "payload_length": round(lengths["payload"], 2),
        "median_width": round(float(median(widths)) if widths else 1.0, 3),
        "edge_count": len(edges),
        "tree_edge_count": tree_edge_count,
        "island_edge_count": island_edge_count,
        "fingerprint": fingerprint,
    }


def _event_tables(canonical_text: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    punctuation = [
        [index, character]
        for index, character in enumerate(canonical_text)
        if not character.isalnum() and character not in {" ", "\n", "'", "-"}
    ]
    segment_events = []
    cursor = 0
    previous_paragraph = None
    for segment in segments:
        paragraph_index = int(segment["paragraph_index"])
        if previous_paragraph is not None:
            cursor += 2 if paragraph_index != previous_paragraph else 1
        start = cursor
        cursor += len(segment["text"])
        segment_events.append([start, cursor, segment.get("terminator", "")])
        previous_paragraph = paragraph_index
    paragraphs = [index for index in range(len(canonical_text) - 1) if canonical_text[index : index + 2] == "\n\n"]
    return {
        "punctuation": punctuation,
        "segments": segment_events,
        "paragraph_breaks": paragraphs,
    }


def _safe_carrier_plan(
    graph: dict[str, Any] | None,
    target_resolution: int,
    module_pitch_px: float = 14.0,
    header_fraction: float = 1 / 3,
) -> dict[str, Any]:
    if not graph or not graph.get("edges") or not graph.get("main_path", {}).get("node_ids"):
        return {
            "target_resolution": target_resolution,
            "module_pitch_px": module_pitch_px,
            "minimum_stroke_px": 3.0,
            "minimum_edge_px": module_pitch_px * 4,
            "header_capacity": max(96, target_resolution // 10),
            "payload_capacity": max(192, target_resolution // 4),
            "carrier_edge_count": 0,
            "carrier_edges": [],
        }
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    main_ids = [int(item) for item in graph["main_path"]["node_ids"]]
    start = nodes[main_ids[0]]
    end = nodes[main_ids[-1]]
    axis_x = float(end["x"]) - float(start["x"])
    axis_y = float(end["y"]) - float(start["y"])
    axis_length_squared = max(1.0, axis_x * axis_x + axis_y * axis_y)
    scale = target_resolution / max(1.0, float(max(graph["width"], graph["height"])))
    minimum_stroke = 3.0
    minimum_edge = module_pitch_px * 4.0
    guard = module_pitch_px
    carriers = []
    capacities = {"header": 0, "payload": 0}
    for edge in graph["edges"]:
        source = nodes[int(edge["source"])]
        target = nodes[int(edge["target"])]
        target_length = float(edge.get("length_px", 0)) * scale
        target_width = float(edge.get("mean_width_px", 1)) * scale
        if target_length < minimum_edge or target_width < minimum_stroke:
            continue
        usable = max(0.0, target_length - guard * 2.0)
        capacity = int(usable // module_pitch_px)
        if capacity <= 0:
            continue
        midpoint_x = (float(source["x"]) + float(target["x"])) / 2.0
        midpoint_y = (float(source["y"]) + float(target["y"])) / 2.0
        projection = (
            (midpoint_x - float(start["x"])) * axis_x
            + (midpoint_y - float(start["y"])) * axis_y
        ) / axis_length_squared
        projection = max(0.0, min(1.0, projection))
        zone = "header" if projection <= header_fraction else "payload"
        capacities[zone] += capacity
        carriers.append(
            {
                "edge_id": int(edge["id"]),
                "zone": zone,
                "capacity": capacity,
                "projection": round(projection, 6),
                "target_length_px": round(target_length, 2),
                "target_width_px": round(target_width, 2),
                "order": int(edge.get("order", 6)),
            }
        )
    carriers.sort(key=lambda item: (item["projection"], item["order"], item["edge_id"]))
    return {
        "target_resolution": target_resolution,
        "module_pitch_px": module_pitch_px,
        "minimum_stroke_px": minimum_stroke,
        "minimum_edge_px": minimum_edge,
        "header_capacity": capacities["header"],
        "payload_capacity": capacities["payload"],
        "carrier_edge_count": len(carriers),
        "carrier_edges": carriers,
    }


def _branch_carrier_plan(
    graph: dict[str, Any] | None,
    target_resolution: int,
    module_pitch_px: float = 14.0,
) -> dict[str, Any]:
    if not graph or not graph.get("edges") or not graph.get("main_path", {}).get("node_ids"):
        branch_count = 12
        payload_capacity = max(384, target_resolution // 2)
        base = payload_capacity // branch_count
        return {
            "target_resolution": target_resolution,
            "module_pitch_px": module_pitch_px,
            "minimum_stroke_px": 1.2,
            "minimum_edge_px": 8.0,
            "main_capacity": max(160, target_resolution // 18),
            "payload_capacity": payload_capacity,
            "carrier_edge_count": 0,
            "main_carriers": [],
            "branches": [
                {
                    "branch_id": index,
                    "anchor_projection": round((index + 1) / (branch_count + 1), 6),
                    "side": -1 if index % 2 == 0 else 1,
                    "capacity": base + (1 if index < payload_capacity % branch_count else 0),
                    "carrier_edges": [],
                }
                for index in range(branch_count)
            ],
        }

    nodes = {int(node["id"]): node for node in graph["nodes"]}
    edges = {int(edge["id"]): edge for edge in graph["edges"]}
    main_node_ids = [int(item) for item in graph["main_path"]["node_ids"]]
    main_edge_ids = {int(item) for item in graph["main_path"]["edge_ids"]}
    main_nodes = set(main_node_ids)
    incident: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    for edge_id, edge in edges.items():
        incident[int(edge["source"])].append(edge_id)
        incident[int(edge["target"])].append(edge_id)

    start = nodes[main_node_ids[0]]
    end = nodes[main_node_ids[-1]]
    axis_x = float(end["x"]) - float(start["x"])
    axis_y = float(end["y"]) - float(start["y"])
    axis_length_squared = max(1.0, axis_x * axis_x + axis_y * axis_y)
    scale = target_resolution / max(1.0, float(max(graph["width"], graph["height"])))
    minimum_stroke = 1.2
    minimum_edge = 8.0
    guard = module_pitch_px * 0.75

    def carrier(edge: dict[str, Any], zone: str, branch_id: int | None = None) -> dict[str, Any] | None:
        target_length = float(edge.get("length_px", 0)) * scale
        target_width = float(edge.get("mean_width_px", 1)) * scale
        if zone == "protocol":
            usable = max(0.0, target_length - guard * 2.0)
            capacity = int(usable // module_pitch_px)
            required_length = module_pitch_px * 2.0
            required_width = 2.0
        else:
            capacity = 1
            required_length = minimum_edge
            required_width = minimum_stroke
        if capacity <= 0 or target_length < required_length or target_width < required_width:
            return None
        payload = {
            "edge_id": int(edge["id"]),
            "zone": zone,
            "capacity": capacity,
            "target_length_px": round(target_length, 2),
            "target_width_px": round(target_width, 2),
            "order": int(edge.get("order", 6)),
        }
        if branch_id is not None:
            payload["branch_id"] = branch_id
        return payload

    main_carriers = []
    for edge_id in graph["main_path"]["edge_ids"]:
        item = carrier(edges[int(edge_id)], "protocol")
        if item:
            main_carriers.append(item)

    visited = set(main_edge_ids)
    branches = []
    denominator = max(1, len(main_node_ids) - 1)
    for main_index, anchor_node in enumerate(main_node_ids):
        root_edges = sorted(
            (
                edge_id
                for edge_id in incident.get(anchor_node, [])
                if edge_id not in visited and edge_id not in main_edge_ids
            ),
            key=lambda edge_id: (
                int(edges[edge_id].get("order", 6)),
                -float(edges[edge_id].get("mean_width_px", 1)),
                edge_id,
            ),
        )
        for root_edge_id in root_edges:
            if root_edge_id in visited:
                continue
            branch_id = len(branches)
            stack = [(root_edge_id, anchor_node)]
            branch_edge_ids = []
            branch_carriers = []
            root_other = anchor_node
            while stack:
                edge_id, from_node = stack.pop()
                if edge_id in visited or edge_id in main_edge_ids:
                    continue
                visited.add(edge_id)
                edge = edges[edge_id]
                branch_edge_ids.append(edge_id)
                item = carrier(edge, "payload", branch_id)
                if item:
                    branch_carriers.append(item)
                next_node = (
                    int(edge["target"])
                    if int(edge["source"]) == from_node
                    else int(edge["source"])
                )
                if edge_id == root_edge_id:
                    root_other = next_node
                if next_node in main_nodes and next_node != anchor_node:
                    continue
                choices = sorted(
                    (
                        candidate
                        for candidate in incident.get(next_node, [])
                        if candidate not in visited and candidate not in main_edge_ids
                    ),
                    reverse=True,
                )
                stack.extend((candidate, next_node) for candidate in choices)
            if not branch_edge_ids:
                continue
            other = nodes[root_other]
            cross = axis_x * (float(other["y"]) - float(start["y"])) - axis_y * (
                float(other["x"]) - float(start["x"])
            )
            projection = main_index / denominator
            branch_carriers.sort(key=lambda item: (item["order"], item["edge_id"]))
            branches.append(
                {
                    "branch_id": branch_id,
                    "anchor_node": anchor_node,
                    "root_edge_id": root_edge_id,
                    "anchor_projection": round(projection, 6),
                    "side": -1 if cross < 0 else 1,
                    "edge_ids": branch_edge_ids,
                    "capacity": sum(item["capacity"] for item in branch_carriers),
                    "carrier_edges": branch_carriers,
                }
            )

    # Skeleton extraction often leaves small reticulate components detached
    # from the main tree. Treat each component as an addressable higher-order
    # branch lane, ordered by projection along the primary vein.
    for seed_edge_id in sorted(edges):
        if seed_edge_id in visited:
            continue
        branch_id = len(branches)
        component_edge_ids = []
        component_carriers = []
        stack = [seed_edge_id]
        component_nodes: set[int] = set()
        while stack:
            edge_id = stack.pop()
            if edge_id in visited:
                continue
            visited.add(edge_id)
            edge = edges[edge_id]
            component_edge_ids.append(edge_id)
            component_nodes.update((int(edge["source"]), int(edge["target"])))
            item = carrier(edge, "payload", branch_id)
            if item:
                component_carriers.append(item)
            for node_id in (int(edge["source"]), int(edge["target"])):
                stack.extend(
                    candidate
                    for candidate in incident.get(node_id, [])
                    if candidate not in visited and candidate not in main_edge_ids
                )
        if not component_carriers:
            continue
        center_x = sum(float(nodes[node_id]["x"]) for node_id in component_nodes) / max(
            1, len(component_nodes)
        )
        center_y = sum(float(nodes[node_id]["y"]) for node_id in component_nodes) / max(
            1, len(component_nodes)
        )
        projection = (
            (center_x - float(start["x"])) * axis_x
            + (center_y - float(start["y"])) * axis_y
        ) / axis_length_squared
        cross = axis_x * (center_y - float(start["y"])) - axis_y * (
            center_x - float(start["x"])
        )
        projection = max(0.0, min(1.0, projection))
        component_side = -1 if cross < 0 else 1
        if branches:
            destination = min(
                branches,
                key=lambda branch: (
                    abs(float(branch["anchor_projection"]) - projection)
                    + (0.08 if int(branch["side"]) != component_side else 0.0),
                    int(branch["branch_id"]),
                ),
            )
            for item in component_carriers:
                item["branch_id"] = int(destination["branch_id"])
            destination["edge_ids"].extend(component_edge_ids)
            destination["carrier_edges"].extend(component_carriers)
            destination["capacity"] = int(destination["capacity"]) + len(
                component_carriers
            )
            destination["includes_detached_components"] = True
        else:
            branches.append(
                {
                    "branch_id": branch_id,
                    "anchor_node": None,
                    "root_edge_id": seed_edge_id,
                    "anchor_projection": round(projection, 6),
                    "side": component_side,
                    "edge_ids": component_edge_ids,
                    "capacity": len(component_carriers),
                    "carrier_edges": component_carriers,
                    "detached_component": True,
                }
            )

    branches = [branch for branch in branches if branch["capacity"] > 0]
    for index, branch in enumerate(branches):
        old_id = branch["branch_id"]
        branch["branch_id"] = index
        for item in branch["carrier_edges"]:
            if item["branch_id"] == old_id:
                item["branch_id"] = index
    branches.sort(key=lambda item: (item["anchor_projection"], item["side"], item["branch_id"]))
    for index, branch in enumerate(branches):
        branch["branch_id"] = index
        for item in branch["carrier_edges"]:
            item["branch_id"] = index
    return {
        "target_resolution": target_resolution,
        "module_pitch_px": module_pitch_px,
        "minimum_stroke_px": minimum_stroke,
        "minimum_edge_px": minimum_edge,
        "main_capacity": sum(item["capacity"] for item in main_carriers),
        "payload_capacity": sum(branch["capacity"] for branch in branches),
        "carrier_edge_count": len(main_carriers)
        + sum(len(branch["carrier_edges"]) for branch in branches),
        "main_carriers": main_carriers,
        "branches": branches,
    }


def evaluate_exportability(
    graph: dict[str, Any] | None,
    protocol_required: int,
    payload_required: int,
    selected_resolution: int,
    module_pitch_px: float = 14.0,
    resolutions: tuple[int, ...] = (2048, 4096, 8192, 16384),
) -> dict[str, Any]:
    rows = []
    for resolution in resolutions:
        capacity = _branch_carrier_plan(
            graph,
            int(resolution),
            module_pitch_px=module_pitch_px,
        )
        protocol_available = int(capacity["main_capacity"])
        payload_available = int(capacity["payload_capacity"])
        protocol_pass = protocol_available >= int(protocol_required)
        payload_pass = payload_available >= int(payload_required)
        rows.append(
            {
                "resolution": int(resolution),
                "exportable": protocol_pass and payload_pass,
                "protocol": {
                    "required": int(protocol_required),
                    "available": protocol_available,
                    "pass": protocol_pass,
                    "usage": round(
                        int(protocol_required) / max(1, protocol_available), 4
                    ),
                },
                "payload": {
                    "required": int(payload_required),
                    "available": payload_available,
                    "pass": payload_pass,
                    "usage": round(
                        int(payload_required) / max(1, payload_available), 4
                    ),
                },
            }
        )

    selected = next(
        (row for row in rows if row["resolution"] == int(selected_resolution)),
        rows[-1],
    )
    minimum_safe = next(
        (row["resolution"] for row in rows if row["exportable"]),
        None,
    )
    failures = []
    if not selected["protocol"]["pass"]:
        failures.append("protocol")
    if not selected["payload"]["pass"]:
        failures.append("payload")
    if selected["exportable"]:
        recommendation = "Ready to export."
    elif minimum_safe is not None and minimum_safe > int(selected_resolution):
        recommendation = f"Increase raster target to at least {minimum_safe}px."
    elif "payload" in failures:
        recommendation = (
            "Shorten the text or select a specimen with more addressable branch edges."
        )
    else:
        recommendation = "Select a prepared specimen with a longer primary vein."
    return {
        "layout_safe": bool(selected["exportable"]),
        "raster_decode_status": "not_tested",
        "selected_resolution": int(selected["resolution"]),
        "minimum_safe_resolution": minimum_safe,
        "bottleneck": "+".join(failures) if failures else None,
        "recommendation": recommendation,
        "resolutions": rows,
    }


def analyze_document(
    raw_text: str,
    lexicon_path: Path,
    database_path: Path,
    sample_metrics: dict[str, Any] | None = None,
    sample_filename: str | None = None,
    sample_graph: dict[str, Any] | None = None,
    target_resolution: int = 8192,
) -> dict[str, Any]:
    raw_paragraphs = [item.strip() for item in re.split(r"\n\s*\n|\n", raw_text.strip()) if item.strip()]
    if not raw_paragraphs:
        raise ValueError("Enter at least one paragraph.")
    if len(raw_text) > 1600:
        raise ValueError("This prototype supports up to 1,600 characters per leaf.")

    paragraphs = []
    segments = []
    all_word_scores = []
    total_words = 0
    total_frames = 0
    canonical_paragraphs = []
    total_clauses = 0
    analysis_cache: dict[str, dict[str, Any]] = {}

    for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
        paragraph_segments = []
        for raw_segment in _split_sentences(raw_paragraph):
            canonical, tokens = _canonical_paragraph(raw_segment)
            canonical = canonical.upper()
            if not tokens:
                continue
            token_payloads = []
            word_index = 0
            clause_breaks = []
            for token_index, raw_token in enumerate(tokens):
                token = raw_token.upper()
                if WORD_PATTERN.match(raw_token):
                    word = re.sub(r"[^A-Za-z]", "", raw_token)
                    key = word.lower()
                    analysis = analysis_cache.get(key)
                    if analysis is None:
                        analysis = analyze_word(word, lexicon_path, database_path)
                        analysis_cache[key] = analysis
                    packet = encode_text(token)
                    all_word_scores.append(analysis["scores"])
                    token_payloads.append(
                        {
                            "type": "word",
                            "text": token,
                            "word_index": word_index,
                            "codec": packet,
                            "scores": analysis["scores"],
                            "morphemes": analysis["morphemes"],
                            "root": analysis["lexical_profile"].get("root", word.lower()),
                        }
                    )
                    word_index += 1
                    total_words += 1
                else:
                    packet = encode_token(token)
                    token_type = "number" if NUMBER_PATTERN.match(raw_token) else "punctuation"
                    token_payloads.append(
                        {
                            "type": token_type,
                            "text": token,
                            "codec": packet,
                            "marker": PUNCTUATION_MARKERS.get(
                                token,
                                {"code": "TOKEN", "topology": "one compact token ring"},
                            )
                            if token_type == "punctuation"
                            else None,
                        }
                    )
                    if token in CLAUSE_PUNCTUATION:
                        clause_breaks.append(
                            {
                                "token_index": token_index,
                                "symbol": token,
                                "level": CLAUSE_PUNCTUATION[token],
                            }
                        )
                total_frames += packet["frame_count"]

            checksum = _crc10(canonical.encode("utf-8"))
            segment = {
                "index": len(segments),
                "paragraph_index": paragraph_index,
                "paragraph_break_before": paragraph_index > 0 and not paragraph_segments,
                "text": canonical,
                "terminator": (
                    TERMINAL_PATTERN.search(canonical).group(0)
                    if TERMINAL_PATTERN.search(canonical)
                    else ""
                ),
                "word_count": word_index,
                "token_count": len(token_payloads),
                "clause_breaks": clause_breaks,
                "checksum": {"decimal": checksum, "bits": f"{checksum:010b}"},
                "boundary_marker": {
                    "code": "SEG",
                    "topology": "paired opposite forks enclosing one small loop",
                },
                "tokens": token_payloads,
            }
            segments.append(segment)
            paragraph_segments.append(segment)
            total_clauses += len(clause_breaks)

        canonical = " ".join(item["text"] for item in paragraph_segments)
        if not canonical:
            continue
        paragraphs.append(
            {
                "index": paragraph_index,
                "text": canonical,
                "segment_indices": [item["index"] for item in paragraph_segments],
                "segment_count": len(paragraph_segments),
                "word_count": sum(item["word_count"] for item in paragraph_segments),
            }
        )
        canonical_paragraphs.append(canonical)

    canonical_text = "\n\n".join(canonical_paragraphs)
    document_checksum = _crc10(canonical_text.encode("utf-8"))
    score_keys = (
        "composite_complexity",
        "information_complexity",
        "form_complexity",
        "morphological_complexity",
        "semantic_complexity",
        "semantic_confidence",
    )
    aggregate = {
        key: round(sum(item[key] for item in all_word_scores) / max(1, len(all_word_scores)), 2)
        for key in score_keys
    }
    target_resolution = max(2048, min(16384, int(target_resolution)))
    module_pitch_px = 14.0
    payload_codec = encode_lexical_payload(canonical_text, database_path, block_size=32)
    capacity = _branch_carrier_plan(
        sample_graph,
        target_resolution,
        module_pitch_px=module_pitch_px,
    )
    graph_profile = _graph_capacity_profile(sample_graph)
    payload_symbol_count = int(payload_codec["symbol_count"])
    payload_capacity = int(capacity["payload_capacity"])
    protocol_header: dict[str, Any] = {
        "version": "0.9.1",
        "graph_fingerprint": graph_profile["fingerprint"][:8],
        "lexicon_fingerprint": payload_codec["lexicon_fingerprint"][:4],
        "lexical_bit_length": int(payload_codec["lexical_bit_length"]),
        "payload_crc16": int(payload_codec["payload_crc16"]),
        "flags": 1,
    }
    protocol_codec = encode_branch_protocol(protocol_header)
    export_preflight = evaluate_exportability(
        sample_graph,
        int(protocol_codec["symbol_count"]),
        payload_symbol_count,
        target_resolution,
        module_pitch_px=module_pitch_px,
    )
    selected_preflight = next(
        row
        for row in export_preflight["resolutions"]
        if row["resolution"] == target_resolution
    )
    header_ready = bool(selected_preflight["protocol"]["pass"])
    payload_ready = bool(selected_preflight["payload"]["pass"])
    capacity_status = "ready" if header_ready and payload_ready else "insufficient"
    branch_assignments = []
    payload_cursor = 0
    block_ranges = []
    block_cursor = 0
    for block in payload_codec["blocks"]:
        block_ranges.append(
            {
                "block_index": int(block["index"]),
                "start_symbol": block_cursor,
                "end_symbol": block_cursor + int(block["symbol_count"]),
            }
        )
        block_cursor += int(block["symbol_count"])
    for branch in capacity["branches"]:
        if payload_cursor >= payload_symbol_count:
            break
        count = min(int(branch["capacity"]), payload_symbol_count - payload_cursor)
        start_symbol = payload_cursor
        end_symbol = start_symbol + count
        overlapping_blocks = [
            item["block_index"]
            for item in block_ranges
            if item["start_symbol"] < end_symbol and item["end_symbol"] > start_symbol
        ]
        branch_assignments.append(
            {
                "branch_id": int(branch["branch_id"]),
                "start_symbol": start_symbol,
                "symbol_count": count,
                "block_indices": overlapping_blocks,
            }
        )
        payload_cursor = end_symbol
    display_segments = []
    for segment in segments:
        display_segment = {key: value for key, value in segment.items() if key != "tokens"}
        display_segment["tokens"] = [
            {
                key: token[key]
                for key in ("type", "text", "word_index", "root", "marker")
                if key in token
            }
            for token in segment["tokens"]
        ]
        display_segments.append(display_segment)

    return {
        "model_version": "hierarchical-branch-document-0.9",
        "input": raw_text,
        "canonical_text": canonical_text,
        "summary": {
            "paragraph_count": len(paragraphs),
            "segment_count": len(segments),
            "clause_break_count": total_clauses,
            "word_count": total_words,
            "character_count": len(canonical_text),
            "encoded_frame_count": total_frames,
            "protocol_symbol_count": protocol_codec["symbol_count"],
            "payload_symbol_count": payload_symbol_count,
            "lexical_bit_count": payload_codec["lexical_bit_length"],
            "dictionary_token_count": payload_codec["dictionary_tokens"],
            "raw_token_count": payload_codec["raw_tokens"],
            "safe_carrier_count": capacity["carrier_edge_count"],
            "branch_count": len(capacity["branches"]),
            "used_branch_count": len(branch_assignments),
            "header_capacity": capacity["main_capacity"],
            "payload_capacity": payload_capacity,
            "payload_tile_repeats": 0,
            "payload_tail_symbols": 0,
            "capacity_status": capacity_status,
            "target_resolution": target_resolution,
            "module_pitch_px": module_pitch_px,
        },
        "scores": aggregate,
        "style_profile": _style_profile(sample_metrics, sample_filename),
        "surface_graph": sample_graph,
        "document_checksum": {
            "algorithm": "CRC-10/ATM",
            "decimal": document_checksum,
            "bits": f"{document_checksum:010b}",
        },
        "protocol_header": protocol_header,
        "protocol_codec": protocol_codec,
        "payload_codec": payload_codec,
        "export_preflight": export_preflight,
        "render_plan": {
            "carrier_plan": capacity,
            "branch_assignments": branch_assignments,
            "capacity_status": capacity_status,
            "protocol_color": "#3f7958",
            "payload_color": "#2f704b",
        },
        "hierarchy": {
            "level_0": "green locator primary vein / minimal Branch Protocol",
            "level_1": "secondary branch subtrees / addressed text lanes",
            "level_2": "green tiered lexical payload with UTF-8 escape",
            "level_3": "four edge states: two green tones × two natural line weights",
            "events": "punctuation and paragraph breaks remain inside the lexical payload",
            "ecc": "CRC-16 validates the protocol and lexical stream; parity branches are reserved",
            "tiling": "disabled; unused natural veins remain pale visual scaffold",
        },
        "paragraphs": paragraphs,
        "segments": display_segments,
    }


def decode_document(packet: dict[str, Any]) -> dict[str, Any]:
    if (
        packet.get("model_version") == "hierarchical-branch-document-0.9"
        and packet.get("protocol_codec")
        and packet.get("payload_codec")
    ):
        protocol_result = decode_packet(packet["protocol_codec"])
        payload_result = decode_packet(packet["payload_codec"])
        decoded_header = protocol_result.get("header", {})
        header_matches = decoded_header == packet.get("protocol_header", {})
        text = payload_result["text"]
        document_valid = (
            _crc10(text.encode("utf-8")) == int(packet["document_checksum"]["decimal"])
        )
        payload_metadata_valid = (
            int(decoded_header.get("lexical_bit_length", -1))
            == int(packet["payload_codec"].get("lexical_bit_length", -2))
            and int(decoded_header.get("payload_crc16", -1))
            == int(packet["payload_codec"].get("payload_crc16", -2))
            and decoded_header.get("lexicon_fingerprint")
            == str(packet["payload_codec"].get("lexicon_fingerprint", ""))[:4]
        )
        assignments = packet.get("render_plan", {}).get("branch_assignments", [])
        expected_cursor = 0
        assignment_valid = True
        for assignment in assignments:
            assignment_valid = assignment_valid and (
                int(assignment.get("start_symbol", -1)) == expected_cursor
                and int(assignment.get("symbol_count", 0)) > 0
            )
            expected_cursor += int(assignment.get("symbol_count", 0))
        carrier_plan = packet.get("render_plan", {}).get("carrier_plan", {})
        capacity_valid = (
            int(carrier_plan.get("main_capacity", 0))
            >= int(packet["protocol_codec"].get("symbol_count", 1))
            and int(carrier_plan.get("payload_capacity", 0))
            >= int(packet["payload_codec"].get("symbol_count", 1))
            and assignment_valid
            and expected_cursor == int(packet["payload_codec"].get("symbol_count", -1))
        )
        valid = (
            protocol_result["valid"]
            and payload_result["valid"]
            and header_matches
            and document_valid
            and payload_metadata_valid
            and capacity_valid
        )
        return {
            "text": text,
            "valid": valid,
            "packet_valid": (
                protocol_result["valid"]
                and payload_result["valid"]
                and header_matches
                and document_valid
                and payload_metadata_valid
            ),
            "document_checksum_valid": document_valid,
            "protocol_header_valid": protocol_result["valid"] and header_matches,
            "payload_valid": payload_result["valid"] and payload_metadata_valid,
            "capacity_valid": capacity_valid,
            "branch_assignment_valid": assignment_valid,
            "corrected_codewords": (
                int(protocol_result.get("corrected_codewords", 0))
                + int(payload_result.get("corrected_codewords", 0))
            ),
            "branch_packets": len(assignments),
            "segments": [],
            "tokens": [],
        }

    if packet.get("protocol_codec") and packet.get("payload_codec"):
        protocol_result = decode_packet(packet["protocol_codec"])
        payload_result = decode_packet(packet["payload_codec"])
        decoded_header = protocol_result.get("header", {})
        header_matches = decoded_header == packet.get("protocol_header", {})
        text = payload_result["text"]
        document_valid = (
            _crc10(text.encode("utf-8")) == int(packet["document_checksum"]["decimal"])
        )
        payload_metadata_valid = (
            int(decoded_header.get("payload_character_count", -1)) == len(text)
            and int(decoded_header.get("payload_compressed_bytes", -1))
            == int(packet["payload_codec"].get("compressed_bytes", -2))
            and int(decoded_header.get("payload_symbol_count", -1))
            == int(packet["payload_codec"].get("symbol_count", -2))
            and int(decoded_header.get("payload_crc16", -1))
            == int(packet["payload_codec"].get("payload_crc16", -2))
            and int(decoded_header.get("block_count", -1))
            == int(packet["payload_codec"].get("block_count", -2))
        )
        tile = decoded_header.get("tile", {})
        payload_symbol_count = max(
            1, int(decoded_header.get("payload_symbol_count", 0))
        )
        tile_plan_valid = (
            int(decoded_header.get("payload_capacity", -1))
            == int(tile.get("full_repeats", -1)) * payload_symbol_count
            + int(tile.get("tail_symbols", -1))
            and 0 <= int(tile.get("tail_symbols", -1)) < payload_symbol_count
            and int(decoded_header.get("header_tail_symbols", -1)) == 0
        )
        capacity_valid = (
            int(decoded_header.get("header_repeats", 0)) >= 1
            and int(tile.get("full_repeats", 0)) >= 1
        )
        expected_punctuation = [
            [index, character]
            for index, character in enumerate(text)
            if not character.isalnum() and character not in {" ", "\n", "'", "-"}
        ]
        punctuation_valid = (
            decoded_header.get("events", {}).get("punctuation") == expected_punctuation
        )
        segment_records = decoded_header.get("events", {}).get("segments", [])
        segment_valid = all(
            isinstance(item, list)
            and len(item) == 3
            and 0 <= int(item[0]) <= int(item[1]) <= len(text)
            for item in segment_records
        )
        valid = (
            protocol_result["valid"]
            and payload_result["valid"]
            and header_matches
            and document_valid
            and payload_metadata_valid
            and tile_plan_valid
            and capacity_valid
            and punctuation_valid
            and segment_valid
        )
        return {
            "text": text,
            "valid": valid,
            "packet_valid": (
                protocol_result["valid"]
                and payload_result["valid"]
                and header_matches
                and document_valid
                and payload_metadata_valid
                and tile_plan_valid
                and punctuation_valid
                and segment_valid
            ),
            "document_checksum_valid": document_valid,
            "protocol_header_valid": protocol_result["valid"] and header_matches,
            "payload_valid": payload_result["valid"] and payload_metadata_valid,
            "tile_plan_valid": tile_plan_valid,
            "capacity_valid": capacity_valid,
            "punctuation_table_valid": punctuation_valid,
            "segment_table_valid": segment_valid,
            "corrected_codewords": (
                int(protocol_result.get("corrected_codewords", 0))
                + int(payload_result.get("corrected_codewords", 0))
            ),
            "tile_redundancy": {
                "full_repeats": int(tile.get("full_repeats", 0)),
                "tail_symbols": int(tile.get("tail_symbols", 0)),
                "majority_vote_available": int(tile.get("full_repeats", 0)) >= 2,
            },
            "segments": [
                {
                    "text": text[int(item[0]) : int(item[1])],
                    "valid": True,
                }
                for item in segment_records
            ],
            "tokens": [],
        }

    if packet.get("segments"):
        decoded_segments = []
        valid = True
        token_results = []
        paragraph_groups: dict[int, list[str]] = {}
        for segment in packet["segments"]:
            tokens = []
            for token in segment.get("tokens", []):
                result = decode_packet(token["codec"])
                tokens.append(result["text"])
                valid = valid and result["valid"]
                token_results.append({"text": result["text"], "valid": result["valid"]})
            canonical, _ = _canonical_paragraph(" ".join(tokens))
            segment_valid = (
                _crc10(canonical.encode("utf-8")) == int(segment["checksum"]["decimal"])
            )
            valid = valid and segment_valid
            paragraph_index = int(segment.get("paragraph_index", 0))
            paragraph_groups.setdefault(paragraph_index, []).append(canonical)
            decoded_segments.append({"text": canonical, "valid": segment_valid})
        paragraphs = [
            " ".join(paragraph_groups[index]) for index in sorted(paragraph_groups)
        ]
        text = "\n\n".join(paragraphs)
        document_valid = (
            _crc10(text.encode("utf-8")) == int(packet["document_checksum"]["decimal"])
        )
        return {
            "text": text,
            "valid": valid and document_valid,
            "document_checksum_valid": document_valid,
            "segments": decoded_segments,
            "tokens": token_results,
        }

    decoded_paragraphs = []
    valid = True
    word_results = []
    for paragraph in packet.get("paragraphs", []):
        tokens = []
        for token in paragraph.get("tokens", []):
            result = decode_packet(token["codec"])
            tokens.append(result["text"])
            valid = valid and result["valid"]
            word_results.append({"text": result["text"], "valid": result["valid"]})
        canonical, _ = _canonical_paragraph(" ".join(tokens))
        paragraph_crc = _crc10(canonical.encode("utf-8"))
        paragraph_valid = paragraph_crc == int(paragraph["checksum"]["decimal"])
        valid = valid and paragraph_valid
        decoded_paragraphs.append({"text": canonical, "valid": paragraph_valid})

    text = "\n\n".join(item["text"] for item in decoded_paragraphs)
    document_valid = _crc10(text.encode("utf-8")) == int(packet["document_checksum"]["decimal"])
    valid = valid and document_valid
    return {
        "text": text,
        "valid": valid,
        "document_checksum_valid": document_valid,
        "paragraphs": decoded_paragraphs,
        "tokens": word_results,
    }
