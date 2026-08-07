# Recognized Root Lexicon

**Model version:** Language Mapping MVP 0.1  
**Current recognized roots:** 25  
**Source file:** `language_data/root_lexicon.json`

This document lists the exact root keys currently recognized by the Leaf Language Engine. After the engine tests possible prefixes and suffixes, the remaining text must exactly match one of these keys to receive a curated semantic score. Otherwise, the root is marked as `estimated`.

The semantic complexity scores are artistic research hypotheses on a 0–100 scale. They are not objective linguistic rankings.

| Root key | Semantic complexity | Current gloss |
|---|---:|---|
| `art` | 62 | Human making and aesthetic expression |
| `bio` | 55 | Life |
| `break` | 44 | Separation or rupture |
| `chron` | 75 | Time |
| `complex` | 72 | Interwoven or composed of many parts |
| `connect` | 58 | Joining or relational linkage |
| `creat` | 60 | Bringing something into being |
| `form` | 48 | Shape or structure |
| `fract` | 55 | Breaking into parts |
| `grow` | 38 | Organic increase or development |
| `growth` | 40 | Organic increase, development, or expansion |
| `language` | 70 | Symbolic communication system |
| `leaf` | 26 | Concrete plant organ |
| `life` | 58 | Living existence |
| `light` | 48 | Illumination or conceptual clarity |
| `lingu` | 68 | Language or tongue |
| `memor` | 73 | Memory and mental retention |
| `memory` | 73 | Mental retention, recall, and reconstruction |
| `mort` | 70 | Death or mortality |
| `natur` | 47 | Nature or inherent character |
| `nect` | 58 | Binding or joining |
| `silence` | 64 | Absence of sound or speech |
| `simple` | 28 | Single or minimally composed |
| `time` | 74 | Temporal duration and order |
| `vein` | 31 | Vascular channel |

## Verified parsing examples

| Input word | Parsed structure | Recognized root |
|---|---|---|
| `recreation` | `re-` + `creat` + `-ion` | `creat` |
| `connection` | `connect` + `-ion` | `connect` |
| `complexity` | `complex` + `-ity` | `complex` |
| `transformation` | `trans-` + `form` + `-ation` | `form` |
| `growth` | `growth` | `growth` |
| `memory` | `memory` | `memory` |
| `silence` | `silence` | `silence` |

## Editing the lexicon

New roots can be added to `language_data/root_lexicon.json` using this structure:

```json
"root-key": {
  "score": 50,
  "gloss": "Short semantic description"
}
```

The root key should contain lowercase English letters only. Restart the local tool after editing the lexicon.
