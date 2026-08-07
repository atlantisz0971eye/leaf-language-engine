from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import struct
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


CODEBOOK_VERSION = "leaf-codebook-1.0"
TOKEN_CODEBOOK_VERSION = "leaf-token-codebook-2.0"
DOCUMENT_CODEBOOK_VERSION = "leaf-document-stream-3.0"
PROTOCOL_CODEBOOK_VERSION = "leaf-protocol-header-1.0"
MACRO_PAYLOAD_VERSION = "leaf-macro-payload-4.0"
LPH_LITE_VERSION = "leaf-protocol-lite-2.0"
LEXICAL_PAYLOAD_VERSION = "leaf-lexical-payload-5.0"
BRANCH_PROTOCOL_VERSION = "leaf-branch-protocol-3.0"
ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + (" ", "'", "-", "<START>", "<END>", "<ESC>")
VALUE_BY_SYMBOL = {symbol: value for value, symbol in enumerate(ALPHABET)}
SYMBOL_BY_VALUE = {value: symbol for symbol, value in VALUE_BY_SYMBOL.items()}
TEXT_SYMBOLS = set(ALPHABET[:29])
DOCUMENT_ESCAPES = {
    ".": 0,
    ",": 1,
    "?": 2,
    "!": 3,
    ";": 4,
    ":": 5,
    "—": 6,
    "–": 7,
    "(": 8,
    ")": 9,
    '"': 10,
    "\n": 11,
    "/": 12,
    "&": 13,
    "%": 14,
}
DOCUMENT_SYMBOL_BY_ESCAPE = {value: symbol for symbol, value in DOCUMENT_ESCAPES.items()}


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def _bits_to_bytes(bits: list[int], bit_length: int | None = None) -> bytes:
    bit_length = len(bits) if bit_length is None else bit_length
    selected = bits[:bit_length]
    selected.extend([0] * ((8 - len(selected) % 8) % 8))
    return bytes(
        sum(selected[index + offset] << (7 - offset) for offset in range(8))
        for index in range(0, len(selected), 8)
    )


def _hamming_15_11_encode(bits: list[int]) -> list[int]:
    padded = bits + [0] * ((11 - len(bits) % 11) % 11)
    encoded: list[int] = []
    data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    for offset in range(0, len(padded), 11):
        positions = [0] * 16
        for position, bit in zip(data_positions, padded[offset : offset + 11]):
            positions[position] = bit
        for parity_position in (1, 2, 4, 8):
            positions[parity_position] = sum(
                positions[position]
                for position in range(1, 16)
                if position & parity_position and position != parity_position
            ) % 2
        encoded.extend(positions[1:])
    return encoded


def _hamming_15_11_decode(bits: list[int], original_bit_length: int) -> tuple[list[int], int]:
    if len(bits) % 15:
        raise ValueError("Hamming payload does not contain complete 15-bit codewords.")
    decoded: list[int] = []
    corrected = 0
    data_positions = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    for offset in range(0, len(bits), 15):
        positions = [0] + bits[offset : offset + 15]
        syndrome = 0
        for parity_position in (1, 2, 4, 8):
            if sum(
                positions[position]
                for position in range(1, 16)
                if position & parity_position
            ) % 2:
                syndrome |= parity_position
        if syndrome:
            positions[syndrome] ^= 1
            corrected += 1
        decoded.extend(positions[position] for position in data_positions)
    return decoded[:original_bit_length], corrected


def _bits_to_macro_symbols(bits: list[int]) -> list[int]:
    padded = bits + [0] * (len(bits) % 2)
    return [padded[index] * 2 + padded[index + 1] for index in range(0, len(padded), 2)]


def _macro_symbols_to_bits(symbols: list[int], encoded_bit_length: int) -> list[int]:
    bits = [bit for symbol in symbols for bit in ((int(symbol) >> 1) & 1, int(symbol) & 1)]
    return bits[:encoded_bit_length]


def _encode_hamming_symbols(data: bytes) -> dict[str, Any]:
    raw_bits = _bytes_to_bits(data)
    encoded_bits = _hamming_15_11_encode(raw_bits)
    return {
        "raw_byte_count": len(data),
        "raw_bit_count": len(raw_bits),
        "encoded_bit_count": len(encoded_bits),
        "symbols": _bits_to_macro_symbols(encoded_bits),
    }


def _decode_hamming_symbols(packet: dict[str, Any]) -> tuple[bytes, int]:
    encoded = _macro_symbols_to_bits(
        [int(item) for item in packet.get("symbols", [])],
        int(packet["encoded_bit_count"]),
    )
    decoded, corrected = _hamming_15_11_decode(encoded, int(packet["raw_bit_count"]))
    return _bits_to_bytes(decoded, int(packet["raw_bit_count"])), corrected


def normalize_text(raw_text: str) -> str:
    text = re.sub(r"\s+", " ", raw_text.strip().upper())
    invalid = sorted(set(text) - TEXT_SYMBOLS)
    if invalid:
        shown = " ".join(repr(character) for character in invalid)
        raise ValueError(f"Unsupported character(s): {shown}. Use A-Z, spaces, apostrophes, or hyphens.")
    if not text:
        raise ValueError("Enter at least one supported character.")
    if len(text) > 80:
        raise ValueError("This version supports up to 80 characters.")
    return text


def _crc10(data: bytes) -> int:
    """CRC-10/ATM: width 10, polynomial 0x233, initial value 0."""
    crc = 0
    for byte in data:
        crc ^= byte << 2
        for _ in range(8):
            crc = ((crc << 1) ^ 0x233) & 0x3FF if crc & 0x200 else (crc << 1) & 0x3FF
    return crc


def _frame(value: int, role: str, symbol: str | None = None, index: int | None = None) -> dict[str, Any]:
    bits = f"{value:05b}"
    parity = bits.count("1") % 2
    return {
        "index": index,
        "role": role,
        "symbol": symbol if symbol is not None else (None if role == "checksum" else SYMBOL_BY_VALUE.get(value)),
        "value": value,
        "bits": bits,
        "even_parity_bit": parity,
        "vein_slots": ["right" if bit == "1" else "left" for bit in bits],
        "parity_vein": "right" if parity else "left",
    }


def encode_text(raw_text: str) -> dict[str, Any]:
    text = normalize_text(raw_text)
    checksum = _crc10(text.encode("ascii"))
    values = [VALUE_BY_SYMBOL[character] for character in text]
    frames = [_frame(VALUE_BY_SYMBOL["<START>"], "start", "<START>", 0)]
    frames.extend(_frame(value, "data", SYMBOL_BY_VALUE[value], index + 1) for index, value in enumerate(values))
    frames.append(_frame(VALUE_BY_SYMBOL["<ESC>"], "checksum_marker", "<ESC>", len(frames)))
    frames.append(_frame((checksum >> 5) & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(checksum & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(VALUE_BY_SYMBOL["<END>"], "end", "<END>", len(frames)))
    return {
        "codebook_version": CODEBOOK_VERSION,
        "normalized_text": text,
        "symbol_count": len(values),
        "frame_count": len(frames),
        "checksum": {"algorithm": "CRC-10/ATM", "decimal": checksum, "bits": f"{checksum:010b}"},
        "frames": frames,
        "visual_grammar": {
            "reading_direction": "petiole_to_tip",
            "frame_axis": "successive_nodes_on_primary_vein",
            "bit_zero": "left_side_vein",
            "bit_one": "right_side_vein",
            "sixth_slot": "even_parity",
        },
    }


def _values_from_frames(frames: Iterable[dict[str, Any] | int]) -> tuple[list[int], bool]:
    values = []
    parity_valid = True
    for item in frames:
        if isinstance(item, int):
            value = item
        else:
            value = int(item["value"])
            if "even_parity_bit" in item:
                expected = f"{value:05b}".count("1") % 2
                parity_valid = parity_valid and int(item["even_parity_bit"]) == expected
        if not 0 <= value < 32:
            raise ValueError(f"Frame value outside the 5-bit range: {value}")
        values.append(value)
    return values, parity_valid


def decode_frames(frames: Iterable[dict[str, Any] | int], codebook_version: str = CODEBOOK_VERSION) -> dict[str, Any]:
    if codebook_version != CODEBOOK_VERSION:
        raise ValueError(f"Unsupported codebook version: {codebook_version}")
    values, parity_valid = _values_from_frames(frames)
    if len(values) < 5:
        raise ValueError("The packet is too short.")
    if values[0] != VALUE_BY_SYMBOL["<START>"] or values[-1] != VALUE_BY_SYMBOL["<END>"]:
        raise ValueError("Start or end frame is missing.")
    try:
        marker_index = values.index(VALUE_BY_SYMBOL["<ESC>"], 1, len(values) - 1)
    except ValueError as exc:
        raise ValueError("Checksum marker is missing.") from exc
    if marker_index + 3 != len(values) - 1:
        raise ValueError("The checksum must contain exactly two 5-bit frames.")

    data_values = values[1:marker_index]
    if any(value >= 29 for value in data_values):
        raise ValueError("A control symbol appeared inside the text payload.")
    text = "".join(SYMBOL_BY_VALUE[value] for value in data_values)
    recorded_checksum = (values[marker_index + 1] << 5) | values[marker_index + 2]
    calculated_checksum = _crc10(text.encode("ascii"))
    return {
        "codebook_version": CODEBOOK_VERSION,
        "text": text,
        "valid": parity_valid and recorded_checksum == calculated_checksum,
        "parity_valid": parity_valid,
        "checksum_valid": recorded_checksum == calculated_checksum,
        "recorded_checksum": recorded_checksum,
        "calculated_checksum": calculated_checksum,
    }


def decode_packet(packet: dict[str, Any]) -> dict[str, Any]:
    version = str(packet.get("codebook_version", CODEBOOK_VERSION))
    if version == LEXICAL_PAYLOAD_VERSION:
        from branch_codec import decode_lexical_payload

        return decode_lexical_payload(packet)
    if version == BRANCH_PROTOCOL_VERSION:
        from branch_codec import decode_branch_protocol

        return decode_branch_protocol(packet)
    if version == MACRO_PAYLOAD_VERSION:
        return decode_macro_payload(packet)
    if version == LPH_LITE_VERSION:
        return decode_lph_lite(packet)
    if version == DOCUMENT_CODEBOOK_VERSION:
        return decode_document_stream(packet)
    if version == PROTOCOL_CODEBOOK_VERSION:
        return decode_protocol_header(packet)
    if version == TOKEN_CODEBOOK_VERSION:
        return decode_token_packet(packet)
    return decode_frames(packet.get("frames", []), version)


MACRO_GUARD_START = [0, 3, 1, 2, 3, 0, 2, 1]
MACRO_GUARD_END = list(reversed(MACRO_GUARD_START))
LPH_FIXED_FORMAT = ">2sB8sHHHHBBBBHHHHHHBHBBB"
LPH_FIXED_SIZE = struct.calcsize(LPH_FIXED_FORMAT)


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Varints cannot encode negative values.")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 28:
            raise ValueError("Varint is too large.")
    raise ValueError("Incomplete varint.")


def encode_macro_payload(raw_text: str, block_size: int = 48) -> dict[str, Any]:
    raw = raw_text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    chunks = [
        compressed[index : index + block_size]
        for index in range(0, len(compressed), block_size)
    ] or [b""]
    blocks = []
    total_symbols = 0
    for index, chunk in enumerate(chunks):
        body = struct.pack(">2sHHB", b"MV", index, len(chunks), len(chunk)) + chunk
        block_crc = _crc16(body)
        encoded = _encode_hamming_symbols(body + struct.pack(">H", block_crc))
        visual_symbols = MACRO_GUARD_START + encoded["symbols"] + MACRO_GUARD_END
        blocks.append(
            {
                "index": index,
                "total": len(chunks),
                "chunk_bytes": len(chunk),
                "crc16": block_crc,
                "raw_byte_count": encoded["raw_byte_count"],
                "raw_bit_count": encoded["raw_bit_count"],
                "encoded_bit_count": encoded["encoded_bit_count"],
                "guard_symbol_count": len(MACRO_GUARD_START),
                "symbols": visual_symbols,
                "symbol_count": len(visual_symbols),
            }
        )
        total_symbols += len(visual_symbols)
    return {
        "codebook_version": MACRO_PAYLOAD_VERSION,
        "compression": "zlib level 9",
        "ecc": "Hamming(15,11), one-bit correction per codeword",
        "block_crc": "CRC-16/CCITT-FALSE",
        "block_size": block_size,
        "original_text": raw_text,
        "original_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "payload_crc16": _crc16(raw),
        "block_count": len(blocks),
        "symbol_count": total_symbols,
        "guards": {"start": MACRO_GUARD_START, "end": MACRO_GUARD_END},
        "blocks": blocks,
    }


def decode_macro_payload(packet: dict[str, Any]) -> dict[str, Any]:
    decoded_chunks: dict[int, bytes] = {}
    corrected_codewords = 0
    blocks_valid = True
    expected_total = int(packet.get("block_count", 0))
    guard_length = len(MACRO_GUARD_START)
    for block in packet.get("blocks", []):
        visual_symbols = [int(item) for item in block.get("symbols", [])]
        guards_valid = (
            visual_symbols[:guard_length] == MACRO_GUARD_START
            and visual_symbols[-guard_length:] == MACRO_GUARD_END
        )
        encoded_packet = {
            "symbols": visual_symbols[guard_length:-guard_length],
            "encoded_bit_count": int(block["encoded_bit_count"]),
            "raw_bit_count": int(block["raw_bit_count"]),
        }
        decoded, corrected = _decode_hamming_symbols(encoded_packet)
        corrected_codewords += corrected
        if len(decoded) < LPH_FIXED_SIZE // 4:
            raise ValueError("Decoded macro block is too short.")
        body, recorded_crc = decoded[:-2], struct.unpack(">H", decoded[-2:])[0]
        magic, index, total, chunk_length = struct.unpack(">2sHHB", body[:7])
        chunk = body[7 : 7 + chunk_length]
        block_valid = (
            guards_valid
            and magic == b"MV"
            and total == expected_total
            and len(chunk) == chunk_length
            and recorded_crc == _crc16(body)
        )
        blocks_valid = blocks_valid and block_valid
        decoded_chunks[index] = chunk
    ordered = b"".join(decoded_chunks[index] for index in range(expected_total))
    raw = zlib.decompress(ordered)
    text = raw.decode("utf-8")
    payload_crc_valid = _crc16(raw) == int(packet["payload_crc16"])
    return {
        "codebook_version": MACRO_PAYLOAD_VERSION,
        "text": text,
        "valid": blocks_valid and payload_crc_valid and len(decoded_chunks) == expected_total,
        "blocks_valid": blocks_valid,
        "payload_crc_valid": payload_crc_valid,
        "corrected_codewords": corrected_codewords,
        "block_count": len(decoded_chunks),
    }


def _lph_binary(header: dict[str, Any]) -> bytes:
    events = header["events"]
    punctuation = events.get("punctuation", [])
    segments = events.get("segments", [])
    paragraphs = events.get("paragraph_breaks", [])
    fingerprint = bytes.fromhex(str(header["graph_fingerprint"]))
    if len(fingerprint) != 8:
        raise ValueError("LPH graph fingerprint must contain 16 hexadecimal characters.")
    fixed = struct.pack(
        LPH_FIXED_FORMAT,
        b"LP",
        8,
        fingerprint,
        int(header["payload_character_count"]),
        int(header["payload_compressed_bytes"]),
        int(header["payload_symbol_count"]),
        int(header["target_resolution"]),
        int(round(float(header["module_pitch_px"]) * 10)),
        int(header["block_size"]),
        int(header["block_count"]),
        int(header["tile"]["full_repeats"]),
        int(header["tile"]["tail_symbols"]),
        int(header["seed"]),
        int(header["carrier_edge_count"]),
        int(header["header_capacity"]),
        int(header["payload_capacity"]),
        int(header["payload_crc16"]),
        int(header["header_repeats"]),
        int(header["header_tail_symbols"]),
        len(punctuation),
        len(segments),
        len(paragraphs),
    )
    output = bytearray(fixed)
    previous = 0
    for position, symbol in punctuation:
        output.extend(_encode_varint(int(position) - previous))
        raw_symbol = str(symbol).encode("utf-8")
        output.append(len(raw_symbol))
        output.extend(raw_symbol)
        previous = int(position)
    previous = 0
    for start, end, terminator in segments:
        output.extend(_encode_varint(int(start) - previous))
        output.extend(_encode_varint(int(end) - int(start)))
        raw_terminator = str(terminator).encode("utf-8")
        output.append(len(raw_terminator))
        output.extend(raw_terminator)
        previous = int(start)
    previous = 0
    for position in paragraphs:
        output.extend(_encode_varint(int(position) - previous))
        previous = int(position)
    output.extend(struct.pack(">H", _crc16(bytes(output))))
    return bytes(output)


def encode_lph_lite(header: dict[str, Any]) -> dict[str, Any]:
    raw = _lph_binary(header)
    encoded = _encode_hamming_symbols(raw)
    symbols = MACRO_GUARD_START + encoded["symbols"] + MACRO_GUARD_END
    return {
        "codebook_version": LPH_LITE_VERSION,
        "ecc": "Hamming(15,11), one-bit correction per codeword",
        "crc": "CRC-16/CCITT-FALSE",
        "header": header,
        "binary_base64": base64.b64encode(raw).decode("ascii"),
        "raw_byte_count": encoded["raw_byte_count"],
        "raw_bit_count": encoded["raw_bit_count"],
        "encoded_bit_count": encoded["encoded_bit_count"],
        "guard_symbol_count": len(MACRO_GUARD_START),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "crc16": struct.unpack(">H", raw[-2:])[0],
    }


def decode_lph_lite(packet: dict[str, Any]) -> dict[str, Any]:
    symbols = [int(item) for item in packet.get("symbols", [])]
    guard_length = len(MACRO_GUARD_START)
    guards_valid = (
        symbols[:guard_length] == MACRO_GUARD_START
        and symbols[-guard_length:] == MACRO_GUARD_END
    )
    decoded, corrected = _decode_hamming_symbols(
        {
            "symbols": symbols[guard_length:-guard_length],
            "encoded_bit_count": int(packet["encoded_bit_count"]),
            "raw_bit_count": int(packet["raw_bit_count"]),
        }
    )
    if len(decoded) < LPH_FIXED_SIZE + 2:
        raise ValueError("Decoded LPH Lite packet is too short.")
    body, recorded_crc = decoded[:-2], struct.unpack(">H", decoded[-2:])[0]
    crc_valid = recorded_crc == _crc16(body)
    values = struct.unpack(LPH_FIXED_FORMAT, body[:LPH_FIXED_SIZE])
    (
        magic,
        version,
        fingerprint,
        payload_character_count,
        payload_compressed_bytes,
        payload_symbol_count,
        target_resolution,
        module_pitch_tenths,
        block_size,
        block_count,
        tile_repeats,
        tile_tail,
        seed,
        carrier_edge_count,
        header_capacity,
        payload_capacity,
        payload_crc16,
        header_repeats,
        header_tail,
        punctuation_count,
        segment_count,
        paragraph_count,
    ) = values
    offset = LPH_FIXED_SIZE
    punctuation = []
    previous = 0
    for _ in range(punctuation_count):
        delta, offset = _decode_varint(body, offset)
        length = body[offset]
        offset += 1
        symbol = body[offset : offset + length].decode("utf-8")
        offset += length
        previous += delta
        punctuation.append([previous, symbol])
    segments = []
    previous = 0
    for _ in range(segment_count):
        delta, offset = _decode_varint(body, offset)
        length, offset = _decode_varint(body, offset)
        term_length = body[offset]
        offset += 1
        terminator = body[offset : offset + term_length].decode("utf-8")
        offset += term_length
        start = previous + delta
        segments.append([start, start + length, terminator])
        previous = start
    paragraphs = []
    previous = 0
    for _ in range(paragraph_count):
        delta, offset = _decode_varint(body, offset)
        previous += delta
        paragraphs.append(previous)
    header = {
        "version": "0.8",
        "graph_fingerprint": fingerprint.hex(),
        "payload_character_count": payload_character_count,
        "payload_compressed_bytes": payload_compressed_bytes,
        "payload_symbol_count": payload_symbol_count,
        "target_resolution": target_resolution,
        "module_pitch_px": module_pitch_tenths / 10,
        "block_size": block_size,
        "block_count": block_count,
        "tile": {"full_repeats": tile_repeats, "tail_symbols": tile_tail},
        "seed": seed,
        "carrier_edge_count": carrier_edge_count,
        "header_capacity": header_capacity,
        "payload_capacity": payload_capacity,
        "payload_crc16": payload_crc16,
        "header_repeats": header_repeats,
        "header_tail_symbols": header_tail,
        "events": {
            "punctuation": punctuation,
            "segments": segments,
            "paragraph_breaks": paragraphs,
        },
    }
    return {
        "codebook_version": LPH_LITE_VERSION,
        "header": header,
        "valid": guards_valid and magic == b"LP" and version == 8 and crc_valid,
        "guards_valid": guards_valid,
        "crc_valid": crc_valid,
        "corrected_codewords": corrected,
        "recorded_crc16": recorded_crc,
        "calculated_crc16": _crc16(body),
    }


def encode_document_stream(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise ValueError("A document stream cannot be empty.")
    checksum = _crc10(raw_text.encode("utf-8"))
    frames = [_frame(VALUE_BY_SYMBOL["<START>"], "start", "<START>", 0)]
    for character in raw_text:
        if character in VALUE_BY_SYMBOL and character in TEXT_SYMBOLS:
            frames.append(_frame(VALUE_BY_SYMBOL[character], "data", character, len(frames)))
        elif character in DOCUMENT_ESCAPES:
            frames.append(_frame(VALUE_BY_SYMBOL["<ESC>"], "escape", "<ESC>", len(frames)))
            frames.append(
                _frame(DOCUMENT_ESCAPES[character], "data", character, len(frames))
            )
        else:
            encoded = character.encode("utf-8")
            if len(encoded) > 28:
                raise ValueError("One Unicode character exceeds the stream escape limit.")
            frames.append(_frame(VALUE_BY_SYMBOL["<ESC>"], "escape", "<ESC>", len(frames)))
            frames.append(_frame(29, "raw_marker", "<RAW>", len(frames)))
            frames.append(_frame(len(encoded), "raw_length", None, len(frames)))
            for byte in encoded:
                frames.append(_frame(byte // 29, "raw_data", None, len(frames)))
                frames.append(_frame(byte % 29, "raw_data", None, len(frames)))
    frames.append(_frame(VALUE_BY_SYMBOL["<ESC>"], "checksum_marker", "<ESC>", len(frames)))
    frames.append(_frame((checksum >> 5) & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(checksum & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(VALUE_BY_SYMBOL["<END>"], "end", "<END>", len(frames)))
    return {
        "codebook_version": DOCUMENT_CODEBOOK_VERSION,
        "text": raw_text,
        "character_count": len(raw_text),
        "frame_count": len(frames),
        "checksum": {"algorithm": "CRC-10/ATM", "decimal": checksum, "bits": f"{checksum:010b}"},
        "frames": frames,
    }


def decode_document_stream(packet: dict[str, Any]) -> dict[str, Any]:
    frames = packet.get("frames", [])
    values, parity_valid = _values_from_frames(frames)
    if len(values) < 5 or values[0] != VALUE_BY_SYMBOL["<START>"] or values[-1] != VALUE_BY_SYMBOL["<END>"]:
        raise ValueError("Document stream start or end frame is missing.")
    marker_index = len(values) - 4
    if values[marker_index] != VALUE_BY_SYMBOL["<ESC>"]:
        raise ValueError("Document stream checksum marker is missing.")
    values = values[1:marker_index]
    output: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value <= 28:
            output.append(SYMBOL_BY_VALUE[value])
            index += 1
            continue
        if value != VALUE_BY_SYMBOL["<ESC>"] or index + 1 >= len(values):
            raise ValueError("Invalid document stream escape.")
        escape_value = values[index + 1]
        if escape_value in DOCUMENT_SYMBOL_BY_ESCAPE:
            output.append(DOCUMENT_SYMBOL_BY_ESCAPE[escape_value])
            index += 2
            continue
        if escape_value != 29 or index + 2 >= len(values):
            raise ValueError("Unknown document stream escape.")
        byte_count = values[index + 2]
        payload_start = index + 3
        payload_end = payload_start + byte_count * 2
        if byte_count <= 0 or payload_end > len(values):
            raise ValueError("Incomplete document stream Unicode escape.")
        raw = bytearray()
        for pair_index in range(payload_start, payload_end, 2):
            high, low = values[pair_index], values[pair_index + 1]
            if high > 8 or low > 28 or high * 29 + low > 255:
                raise ValueError("Invalid byte in document stream Unicode escape.")
            raw.append(high * 29 + low)
        output.append(bytes(raw).decode("utf-8"))
        index = payload_end
    text = "".join(output)
    recorded_checksum = (int(packet["frames"][-3]["value"]) << 5) | int(
        packet["frames"][-2]["value"]
    )
    calculated_checksum = _crc10(text.encode("utf-8"))
    return {
        "codebook_version": DOCUMENT_CODEBOOK_VERSION,
        "text": text,
        "valid": parity_valid and recorded_checksum == calculated_checksum,
        "parity_valid": parity_valid,
        "checksum_valid": recorded_checksum == calculated_checksum,
        "recorded_checksum": recorded_checksum,
        "calculated_checksum": calculated_checksum,
    }


def encode_protocol_header(header: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(
        header, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    compressed = zlib.compress(serialized, level=9)
    checksum = _crc10(compressed)
    frames = [_frame(VALUE_BY_SYMBOL["<START>"], "start", "<START>", 0)]
    for byte in compressed:
        frames.append(_frame(byte // 29, "data", None, len(frames)))
        frames.append(_frame(byte % 29, "data", None, len(frames)))
    frames.append(_frame(VALUE_BY_SYMBOL["<ESC>"], "checksum_marker", "<ESC>", len(frames)))
    frames.append(_frame((checksum >> 5) & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(checksum & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(VALUE_BY_SYMBOL["<END>"], "end", "<END>", len(frames)))
    return {
        "codebook_version": PROTOCOL_CODEBOOK_VERSION,
        "compression": "zlib level 9",
        "uncompressed_bytes": len(serialized),
        "compressed_bytes": len(compressed),
        "frame_count": len(frames),
        "checksum": {"algorithm": "CRC-10/ATM", "decimal": checksum, "bits": f"{checksum:010b}"},
        "frames": frames,
    }


def decode_protocol_header(packet: dict[str, Any]) -> dict[str, Any]:
    frames = packet.get("frames", [])
    values, parity_valid = _values_from_frames(frames)
    if len(values) < 5 or values[0] != VALUE_BY_SYMBOL["<START>"] or values[-1] != VALUE_BY_SYMBOL["<END>"]:
        raise ValueError("Protocol header start or end frame is missing.")
    marker_index = len(values) - 4
    if values[marker_index] != VALUE_BY_SYMBOL["<ESC>"]:
        raise ValueError("Protocol header checksum marker is missing.")
    payload = values[1:marker_index]
    if len(payload) % 2:
        raise ValueError("Protocol header has an incomplete base-29 byte.")
    compressed = bytearray()
    for index in range(0, len(payload), 2):
        high, low = payload[index], payload[index + 1]
        value = high * 29 + low
        if high > 8 or low > 28 or value > 255:
            raise ValueError("Invalid protocol header byte.")
        compressed.append(value)
    recorded_checksum = (values[marker_index + 1] << 5) | values[marker_index + 2]
    calculated_checksum = _crc10(bytes(compressed))
    header = json.loads(zlib.decompress(bytes(compressed)).decode("utf-8"))
    return {
        "codebook_version": PROTOCOL_CODEBOOK_VERSION,
        "header": header,
        "valid": parity_valid and recorded_checksum == calculated_checksum,
        "parity_valid": parity_valid,
        "checksum_valid": recorded_checksum == calculated_checksum,
        "recorded_checksum": recorded_checksum,
        "calculated_checksum": calculated_checksum,
    }


def encode_token(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise ValueError("A token cannot be empty.")
    encoded = raw_text.encode("utf-8")
    values = []
    for byte in encoded:
        values.extend((byte // 29, byte % 29))
    checksum = _crc10(encoded)
    frames = [_frame(VALUE_BY_SYMBOL["<START>"], "start", "<START>", 0)]
    frames.extend(_frame(value, "data", None, index + 1) for index, value in enumerate(values))
    frames.append(_frame(VALUE_BY_SYMBOL["<ESC>"], "checksum_marker", "<ESC>", len(frames)))
    frames.append(_frame((checksum >> 5) & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(checksum & 0x1F, "checksum", None, len(frames)))
    frames.append(_frame(VALUE_BY_SYMBOL["<END>"], "end", "<END>", len(frames)))
    return {
        "codebook_version": TOKEN_CODEBOOK_VERSION,
        "encoding": "UTF-8 as two base-29 frames per byte",
        "text": raw_text,
        "byte_count": len(encoded),
        "frame_count": len(frames),
        "checksum": {"algorithm": "CRC-10/ATM", "decimal": checksum, "bits": f"{checksum:010b}"},
        "frames": frames,
    }


def decode_token_packet(packet: dict[str, Any]) -> dict[str, Any]:
    values, parity_valid = _values_from_frames(packet.get("frames", []))
    if len(values) < 5 or values[0] != VALUE_BY_SYMBOL["<START>"] or values[-1] != VALUE_BY_SYMBOL["<END>"]:
        raise ValueError("Start or end frame is missing.")
    marker_index = len(values) - 4
    if values[marker_index] != VALUE_BY_SYMBOL["<ESC>"]:
        raise ValueError("Checksum marker is missing.")
    payload = values[1:marker_index]
    if len(payload) % 2:
        raise ValueError("The UTF-8 payload has an incomplete base-29 pair.")
    raw = bytearray()
    for index in range(0, len(payload), 2):
        high, low = payload[index], payload[index + 1]
        if high > 8 or low > 28:
            raise ValueError("Invalid base-29 byte pair.")
        value = high * 29 + low
        if value > 255:
            raise ValueError("Decoded byte is outside the UTF-8 range.")
        raw.append(value)
    text = bytes(raw).decode("utf-8")
    recorded_checksum = (values[marker_index + 1] << 5) | values[marker_index + 2]
    calculated_checksum = _crc10(bytes(raw))
    return {
        "codebook_version": TOKEN_CODEBOOK_VERSION,
        "text": text,
        "valid": parity_valid and recorded_checksum == calculated_checksum,
        "parity_valid": parity_valid,
        "checksum_valid": recorded_checksum == calculated_checksum,
        "recorded_checksum": recorded_checksum,
        "calculated_checksum": calculated_checksum,
    }
