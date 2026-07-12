# Kinetic Typography

Data and analysis code for a controlled experiment (N = 207) examining how background structural complexity and on-screen spatial position jointly shape attention, readability, and aesthetic evaluation of kinetic typography, supporting the manuscript *"Placing Kinetic Text on Complex Backgrounds: Divergent Effects on Attention, Legibility, and Aesthetics."*

## Contents

- `1A-statistical.py` — extracts response-frequency data from `RESULT.docx`, runs the statistical analyses, and generates all figures below.
- `RESULT.docx` — source tables (per background type and perceptual dimension) containing raw selection-frequency counts across the five spatial positions.
- `*.csv` — extracted/derived tabular data (background, dimension, spatial-position frequencies, totals) used as input to the statistical tests and plots.
- `statistical_summary_for_paper.txt` — plain-text summary of the key statistical results (bootstrap redistribution test, attention–readability correlation shift, entropy stability comparison).

## Figures

- `1A-perceptual.png` — selection intensity (%) by spatial position, dimension, and background type, with inter-dimension sensitivity (chi-square) p-values.
- `Redistribution.png` — deviation in selection intensity from the center position, by dimension and background.
- `Attention-Readability.png` — scatter plot of attention vs. readability selection percentages across positions and backgrounds.
- `Perceptual-Dispersion.png` — Shannon entropy of selection distributions across backgrounds, by dimension.

## Analysis pipeline

1. **Data extraction** — parses frequency tables from `RESULT.docx` into a tidy DataFrame (`Background`, `Dimension`, `Frequencies`, `Total`).
2. **Chi-square tests** — goodness-of-fit (within-dimension deviation from uniform selection) and independence (inter-dimension sensitivity within each background).
3. **Bootstrap validation** (1,000 resamples) — tests whether peripheral positions differ significantly from center under high background complexity.
4. **Correlation analysis** — Pearson correlation between attention and readability selection percentages, compared across background complexity levels.
5. **Entropy analysis** — Shannon entropy of selection distributions, used to compare the stability of aesthetic judgments against readability judgments across backgrounds.

## Requirements

```
pandas
numpy
scipy
matplotlib
python-docx
```

## Usage

```bash
python 1A-statistical.py
```

Requires `RESULT.docx` in the working directory. Outputs all figures (`.png`) and the statistical summary (`.txt`) to the same directory.

## Citation

If you use this data or code, please cite the associated manuscript (details to be added upon publication).
