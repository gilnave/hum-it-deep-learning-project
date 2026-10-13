# Third-party data and code

## CHAD

The project uses the **Cover and Hummings Aligned Dataset (CHAD)** introduced in:

Amatov, A. et al., “A Semi-Supervised Deep Learning Approach to Dataset Collection for
Query-by-Humming Task,” ISMIR 2023.

- Official code and metadata: <https://github.com/amanteur/CHAD>
- Official humming subset: <https://huggingface.co/datasets/amanteur/CHAD_hummings>
- Paper: <https://archives.ismir.net/ismir2023/paper/000077.pdf>
- Humming archive license: CC BY-NC 4.0
- Official repository code license: MIT

The official metadata points to reference and cover recordings hosted on YouTube. Those
recordings are not redistributed with this project. The preparation script downloads only the
aligned fragments needed for research when the corresponding source remains available. Users
are responsible for observing source-platform terms and applicable copyright rules.

The later personal-song demonstration is deliberately separate from the CHAD benchmark.
Personal audio files stay local and must not be committed, submitted, or redistributed unless
the user has permission to do so.

## Omnizart vocal-contour model

The general 30-song pilot uses the pretrained **Omnizart vocal-contour** network to extract
frame-level singing melody from each local studio recording.

- Official repository: <https://github.com/Music-and-Culture-Technology-Lab/omnizart>
- Pinned source commit: `bcd8cb44d4da66ce87df10b6abee5c35a8cc2886`
- Software license: MIT
- Official checkpoint release: `checkpoints-20211001`
- Vocal-contour weights SHA-256:
  `6b5c76f749ab552d7735fdcd472ebf29dd31d41609fe8b1160b9c037d6919339`
- Paper: Wu et al., “Omnizart: A General Toolbox for Automatic Music Transcription,”
  Journal of Open Source Software, 2021, DOI `10.21105/joss.03391`.

The checkpoint runs only during offline catalog preparation. Songs, humming recordings,
features, and contours remain on the local computer. The project stores provenance and hashes
for every generated contour. Omnizart's original one-frame sliding inference was replaced by
documented 50%-overlap inference so the local CPU can build the pilot in practical time; the
pretrained network, feature extraction, and pitch decoder remain unchanged.
