# Contour-Registered Texture Codec (CRTC) v1

CRTC replaces synthetic vein topology with two carriers already suggested by a
real rubbing:

- the inner edge of the silhouette carries a fixed-length protocol header;
- paired local optical-density marks inside the silhouette carry UTF-8 text.

The v1 profile is intended for lossless digital PNG files. The silhouette is
stored in the PNG alpha channel, so the exported PNG is the only object needed
for decoding; its adjacent JSON file is human-readable diagnostic metadata and
is not used by the decoder.

## Encoding

1. Extract and normalize the leaf silhouette.
2. Compress UTF-8 text with zlib when compression is beneficial.
3. Protect the payload with Hamming(15,11).
4. Write a three-copy, CRC-protected header just inside the real boundary.
5. Place each payload bit in a local pair of green micro-marks. The darker mark
   appearing first means `0`; the darker mark appearing second means `1`.
6. Preserve the source rubbing underneath and export a single RGBA PNG.

The header records the payload length, original text length, cell spacing,
compression flag, and text CRC-32. Positions are generated deterministically
from the silhouette and the CRC-derived seed.

## Decoding

1. Recover the silhouette from the PNG.
2. Traverse its contour in a canonical direction.
3. Majority-vote the three header copies and verify the header CRC.
4. Regenerate the interior cell positions.
5. Compare the optical density within each local pair.
6. Apply Hamming correction, decompress the bytes, and verify the text CRC-32.

## Current boundary

Version 1 is a digital round-trip prototype. Rotation, perspective correction,
alpha-free segmentation, and print/camera recovery belong to the later
print-safe profile. Those features can use the existing three-part start locator
without changing the payload grammar.

## Command-line use

```sh
python3 contour_texture_codec.py encode rubbing_clean.png encoded.png \
  --silhouette silhouette.png --resolution 2048 --text "Text to store"

python3 contour_texture_codec.py decode encoded.png
```
