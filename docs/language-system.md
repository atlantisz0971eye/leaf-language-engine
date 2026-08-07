# Leaf Language System 0.9.6

## Hierarchical branch grammar

MVP 0.9 removes the artificial first-third protocol region. Version 0.9.3 makes
the primary vein a continuous green locator rail. Natural branch subtrees are ordered from
petiole to apex and act as green text lanes. Punctuation and paragraph breaks
remain in the text stream; they are not drawn as separate icons.

The specimen still owns every visible node and edge. Text does not create a
branch. Unused edges remain a pale natural scaffold.

## Main-vein protocol

The compact Branch Protocol stores only:

- protocol version;
- graph and lexical-codebook fingerprints;
- character, byte, symbol, and branch counts;
- target resolution and primary-rail pitch;
- payload and capacity data;
- CRC-16 and deterministic seed.

The protocol does not duplicate sentence or punctuation tables. Version 0.9.1
reduces the header to a short graph fingerprint, lexical fingerprint, payload
bit length, payload CRC, flags, and header CRC. This reduces the tested header
from 188 to 72 two-bit cells.

The cells no longer grow perpendicular ticks. They are consecutive sections of
the specimen's primary vein. Two green tones and two widths encode the four
states. Width is measured relative to a natural taper from petiole to apex; the
current raster-tested variation is approximately -14% or +14%. Cell boundaries
prefer nearby natural main-vein nodes. Only one start ring and one terminal
double notch are added.

The decoder first detects the start ring and terminal double notch, then traces
the strongest continuous path between them. This establishes the primary vein
before any cell is read. The first four states are the fixed guard `00 11 10
01`; the final four states reverse that order. These guards establish reading
direction and calibrate the two relative tone and width levels from the PNG
itself. Machine-learning primary-vein classification is optional fallback, not
a protocol dependency.

## Reversible lexical codec

Canonical English is tokenized before visual encoding:

- the 256 most common words use 10 bits;
- the next dictionary tiers use 14 or 17 bits;
- a space uses 4 bits;
- common punctuation uses 8 bits;
- paragraph breaks use 4 bits;
- unknown words use a reversible 7-bit ASCII form;
- other characters use a UTF-8 escape.

The codebook begins with a stable common-word layer and continues with lexemes
from the offline WordNet/MorphyNet database. Its fingerprint is stored in the
main-vein protocol, so a decoder can reject the wrong dictionary.

## Four-state branch edges

The previous version added thousands of perpendicular micro-spurs. Version 0.9
instead turns each eligible existing branch edge into one two-bit state:

- `00`: light green, thin;
- `01`: dark green, thin;
- `10`: light green, thick;
- `11`: dark green, thick.

The node overlay supplies edge boundaries and reading addresses. Because the
state changes the appearance of an existing vein rather than adding a new
micro-vein, the result keeps the specimen's branching logic and is much less
dense.

Branch roots are read in their order along the primary vein. Every used branch
has a small start collar. Connected higher-order edges and detached skeleton
components are assigned to the nearest natural branch lane by primary-axis
projection.

## Integrity and capacity

The primary protocol and lexical stream carry CRC-16. A clean packet is fully
reversible; damage is detected rather than silently accepted. Dedicated parity
branches are reserved for the next error-recovery stage.

Tiling is disabled. A leaf is accepted only when:

- the complete protocol fits on the primary vein;
- the complete lexical stream fits on addressed branch edges.

The 8192 px benchmark for the current Acanthopanax sample carries roughly
eighty English words with the current 72-symbol main protocol and about 800
branch-edge states. Exact capacity varies with vocabulary and specimen
topology.

## Decode-safe surface renderer

Version 0.9.2 separates the data geometry from its visual material. The source
leaf alpha is aligned with the analyzed graph and used as a surface mask. A
pale leaf albedo, bounded procedural texture, diffuse light, a small specular
response, and vein relief are composited behind the protocol and payload
layers.

The interface has four diagnostic views:

- **Beauty**: optional surface color, texture, light, translucency, and vein relief;
- **Encoding**: the unchanged flat protocol, payload, and unused scaffold;
- **Normal**: an RGB normal-style preview of the leaf mask and vein height;
- **Data mask**: isolated protocol and payload carriers on black.

Relief is limited to the renderer's auxiliary shadow/highlight copies.
Translucency only changes the surface image opacity. Light angle only changes
the SVG lighting direction. None of these settings alter graph coordinates,
protocol cell boundaries, payload state values, or their protected final
strokes. Each rendered SVG records an encoding signature made from both
CRC-16 values and the protocol/payload symbol counts.

Displacement, blur on data strokes, and geometry-changing surface warps are
deliberately excluded. The normal view is currently a browser preview rather
than a standalone exported normal-map file.

The intended final artifact is the Encoding PNG. It must be independently
decoded from pixels; the in-memory packet verifier is not considered proof of
image decoding. Beauty remains an optional presentation preview until a full
PNG round-trip succeeds.

## Specimen ingestion

Version 0.9.4 adds the same PNG upload pipeline to Leaf Analysis and Language
Engine. Each uploaded specimen is validated, saved to the shared specimen
folder, analyzed, exported, and immediately added to both selectors.

The accepted input envelope is:

- PNG only, up to 32 MB and 120 megapixels;
- transparent leaf cutouts are preferred;
- centered cleared leaves on flat white or black backgrounds are also
  supported through border-color matte detection;
- images must be at least 128 × 128 pixels.

An upload generates its own `metrics.json` and `graph.json`. Language mapping
loads that selected specimen's exported graph, including its primary path,
branch anchors, topology, and capacity. Graph caches are keyed by specimen and
file timestamps; they never silently reuse the first specimen.

If the default threshold does not produce a usable primary path, the upload
pipeline can test alternate sensitivity and contrast profiles and retain the
best graph. A specimen is marked ready only when it contains a primary path
and at least eight graph edges. The five current NMNS-derived specimens all
pass the default profile.

## Encoding PNG export

Version 0.9.5 adds a server-side raster exporter for the final Encoding view.
It redraws the selected specimen graph from the verified packet rather than
taking a browser screenshot.

The exported PNG contains only:

- a white registration background;
- the pale unused specimen scaffold;
- four-state green payload edges;
- the four-state tapered green primary protocol;
- the start ring and terminal double notch.

Beauty and Normal layers are never composited into this file. The requested
2048, 4096, 8192, or 16384 value is the exact PNG width; height is 76% of that
width, matching the logical glyph canvas. Line widths, locator sizes, and graph
coordinates scale with the selected raster target.

Before export, the server runs the document decoder and requires valid protocol
and payload CRCs, complete branch assignments, and sufficient specimen
capacity. An incomplete packet is rejected rather than exported as a
plausible-looking leaf.

No packet or hidden decoding metadata is embedded in the PNG. Future raster
decoding must recover the protocol and payload from the visible pixels alone.

## Export preflight

Version 0.9.6 applies one shared `evaluate_exportability` function to packet
generation, the resolution selector, and PNG export gating. It compares exact
protocol and payload requirements with the selected specimen's safe carrier
capacity.

Every mapping evaluates 2048, 4096, 8192, and 16384 px at once. The result
contains:

- required and available primary-protocol cells;
- required and available payload edge states;
- pass/fail status for each resolution;
- the minimum safe resolution;
- the current bottleneck and a recommendation;
- separate `layout_safe` and `raster_decode_status` fields.

The interface presents two usage meters and a four-resolution matrix. Clicking
a matrix cell remaps the document at that resolution. `Export Encoding PNG` is
enabled only when the selected matrix cell is layout-safe.

Increasing resolution cannot create new veins. Once every eligible graph edge
is addressable, payload capacity reaches the specimen's topology ceiling. If
the text exceeds that ceiling at 16384 px, the preflight recommends a shorter
text or a specimen with more branch edges.

## Current decode boundary

Software packet round trips, CRC damage detection, graph-capacity checks, and
synthetic JPEG/blur/downsampling classification are implemented. The synthetic
test covers the legacy MVG-4 alphabet, four green branch-edge states, and the
continuous tapered main-vein protocol.

Direct decoding from a newly photographed leaf still requires image
registration against the specimen graph, primary-vein detection, branch-edge
segmentation, and tone/width classification.
