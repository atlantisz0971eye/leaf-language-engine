# Contour Colour Codec (CCC) v3

CCC v3 keeps the geometry and luminance texture of a real leaf rubbing. It adds
only a colour-modulated contour protocol. Text is stored by shifting the colour
of pixels that already belong to the source rubbing; it does not add interior
dots, strokes, or reconstructed veins.

## Visual grammar

- A continuous green/olive outline carries the compact protocol header.
- The original rubbing supplies every interior carrier pixel.
- A local pair of source-texture regions carries one bit by relative Lab-a
  chroma. Their luminance is not intentionally changed.
- Hamming(15,11) protects the text and CRC-32 validates the recovered UTF-8.

## JPG profile

Every usable source-texture region has an address derived from its stable grid
coordinate. The contour stores a two-bit address salt selected for that leaf.
Many independently addressed regions vote for each payload bit, so a missing or
newly detected region creates a local erasure instead of shifting the rest of
the message. No alpha channel or sidecar JSON is required. The JSON written
beside an export is only diagnostic metadata.

The 1558 v3 prototype was exported at JPG quality 96 and decoded successfully
after additional quality 92, 88, and 85 recompressions. Print/camera recovery,
rotation, and perspective rectification remain a separate profile.

## Texture preservation

For the 1558 prototype, the interior grayscale texture retained a luminance
correlation of 0.998394 against the protocol-only source, with a mean absolute
luminance difference of 0.736 on a 0–255 scale.

## Command-line use

```sh
python3 contour_color_codec.py encode rubbing_clean.png encoded.jpg \
  --silhouette silhouette.png --resolution 3072 --quality 96 \
  --text "Text to store"

python3 contour_color_codec.py decode encoded.jpg
```
