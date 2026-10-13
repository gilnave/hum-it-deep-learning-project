# Query by Humming Song Retrieval

Deep-learning course project for retrieving a song from a short hummed query.
The final system searches a fixed song catalog using Demucs vocal stems,
transposition-invariant chroma comparison, a learned late-interaction scorer,
and bounded chorus evidence.

## Method

1. Separate each catalog song's vocal stem with Demucs (`htdemucs`).
2. Divide each vocal stem into 24 twelve-second reference windows.
3. Convert the hum and reference windows into chroma features.
4. Circularly align pitch classes to handle humming in a different key.
5. Score the strongest query/reference interactions with the neural model.
6. Apply bounded additive chorus evidence and rank every candidate.

The checkpoint was trained on public CHAD training queries. CHAD validation was
used for checkpoint selection; CHAD test and personal recordings did not update
the neural weights.

## Important files

- `app_chad_vocals_compare.py` — demo launcher.
- `src/qbh/vocal_neural_runtime.py` — final retrieval runtime.
- `src/qbh/additive_chorus_runtime.py` — chorus-aware ranking runtime.
- `src/qbh/symbolic/vocal_neural.py` — model architecture.
- `src/qbh/symbolic/vocal_neural_data.py` — features and query preparation.
- `src/qbh/symbolic/audio_vocal_windows.py` — vocal reference windows.
- `src/qbh/symbolic/additive_chorus.py` — bounded chorus policy.
- `scripts/train_vocal_neural.py` — training entry point.
- `scripts/evaluate_vocal_neural_test.py` — main evaluation.
- `scripts/evaluate_results.py` — main results.

## Installation

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch torchaudio
python -m pip install -e .
```

Demucs is required only to regenerate vocal stems:

```bash
python -m pip install -e ".[separation]"
```

## Data and copyright

Audio recordings, full songs, personal hums, and generated vocal stems are not
included. Place legally obtained catalog audio under `data/catalog`, following
the filenames in `configs/catalog.json`.

Download CHAD from the sources in `THIRD_PARTY_DATA.md`. The checked-in vocal
manifest and chorus metadata provide provenance but do not redistribute audio.

The verified Demucs weight used in the experiment had SHA-256
`8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4`.
Place it at `models/demucs/955717e8-8726e21a.th` to reproduce stem extraction.

## Preprocessing and training

```bash
python scripts/extract_vocal_stems.py \
  --config configs/vocal_stems.json

python scripts/train_vocal_neural.py \
  --config configs/vocal_neural.yaml
```

Training also requires the CHAD-derived sequence artifacts named in the training
configuration. Generated datasets, feature caches, and stems remain ignored.

## Evaluation

```bash
python scripts/evaluate_vocal_neural_test.py
python scripts/evaluate_results.py --protocol path/to/protocol.json
```

## Main results

| Model | Top-1 | Top-3 | Top-10 | MRR |
|---|---:|---:|---:|---:|
| Current model | 21.95% | 36.59% | 48.78% | 0.3153 |
| Full-song model | 14.63% | 24.39% | 41.46% | 0.2369 |
| Model without bonuses | 7.32% | 9.76% | 17.07% | 0.1160 |

The values are stored in `results/main_results.json`.

## Demo

After supplying the catalog audio and generated stems:

```bash
python app_chad_vocals_compare.py --port 7866
```

Bibliographic entries are in `references.bib`; third-party data and software are
documented in `THIRD_PARTY_DATA.md`.
