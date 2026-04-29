# Figure Style Guide (Paper Plots)

This document defines the style rules every paper figure must follow.
Apply these rules to all new plots and to any plot edited going forward.

---

## 1. Color palette

### 1.1 Primary palette — Adobe pastel
Use these as the default colors for categorical encodings, attribute families,
app categories, channels, and any data-driven hue assignment.

| Variable | Hex | Sample |
|----------|-----|--------|
| `P_TEAL`   | `#7ADBC4` | teal-green |
| `P_YELLOW` | `#FAD765` | golden yellow |
| `P_ORANGE` | `#FA9F5C` | warm orange |
| `P_GREEN`  | `#98D198` | sage green |
| `P_BLUE`   | `#6C80FC` | periwinkle blue |
| `P_MAUVE`  | `#ACA4B3` | muted purple-gray |
| `P_SLATE`  | `#687692` | slate blue-gray |

### 1.2 Reserved stage colors
These are reserved exclusively for the **input → raw output → externalized**
flow comparison. Do not use them for any other categorical encoding.

| Stage | Variable | Hex |
|-------|----------|-----|
| Input  (GT label)        | `BLUE`   | `#3A7DC9` |
| Raw output (model)       | `ORANGE` | `#E8762C` |
| Externalized (channel)   | `RED`    | `#C93A3A` |

### 1.3 Verdict colors
Reserved for 3-class verdict encodings.

| Verdict | Hex |
|---------|-----|
| Confirmed leakage | `#C93A3A` (RED) |
| Possible leakage  | `#E8762C` (ORANGE) |
| No evidence       | `#2CA463` (GREEN) |

### 1.4 Encoding rules
- **Same metric, different conditions:** use one hue from the primary palette,
  vary alpha (`1.0` for the focal/dark condition, `0.40`–`0.45` for the
  comparison/light condition). Do not introduce a second hue family.
- **Different metrics on the same axis:** use different hues from the primary
  palette (e.g., `P_BLUE` vs `P_ORANGE`).
- **Annotations and reference lines** (mean lines, dashed thresholds, callout
  arrows): use `P_SLATE` so they read as connective markup, not a data series.
- **Tier ramps** (recall tiers, severity bands): use the ordered palette
  `P_GREEN → P_TEAL → P_YELLOW → P_ORANGE` (best → worst).

---

## 2. Background and chrome

- **Figure background:** white (`#FFFFFF`). Do not use the pale-blue
  (`#F0F4FA`) we used in earlier drafts; it does not survive the PDF
  conversion cleanly and clashes with the paper's white page background.
- **Axes background:** white. Set `ax.set_facecolor("white")`.
- **Grid:** `seaborn` whitegrid, light gray, behind data (`ax.set_axisbelow(True)`).
- **Spines:** hide top and right (`ax.spines["top"].set_visible(False)`,
  `ax.spines["right"].set_visible(False)`).
- **Bar edges:** white (`edgecolor="white"`), `linewidth=0.6`.
- **Fonts:** seaborn default with `sns.set_theme(style="whitegrid", font_scale=1.15)`.

---

## 3. One plot, one PNG

- **Do not concatenate panels into a single figure.** Each plot is one Axes
  object and one PNG. If you have a related set of panels (A / B / C), save
  each panel as its own file and reference them as separate figures in the
  paper.
- This rule applies even when two panels share an axis or legend. If a shared
  legend is needed, place the legend on each panel; do not share an Axes
  object across panels.
- Rationale: separately-saved PNGs let LaTeX position panels independently,
  scale them to different widths if needed, and keep figure captions tightly
  scoped to a single observation.

---

## 4. Two aspect ratios per plot

Every figure must be saved in **both** of these aspect ratios:

| Suffix | Size (inches) | Pixels @ 150 dpi | Use case |
|--------|---------------|------------------|----------|
| `_1x1` | `6.0 × 6.0`   | `900 × 900`      | Square panel; column-width LaTeX figures, dense vertical lists, square heatmaps. |
| `_2x1` | `12.0 × 6.0`  | `1800 × 900`     | Horizontal (`width = 2 × height`) panel; full-textwidth figures, horizontal bar groups, time-series. |

File naming:
```
analysis/attachments/<slug>_1x1.png
analysis/attachments/<slug>_2x1.png
```

Both files must contain identical content; only the aspect ratio differs.
Pick whichever ratio fits the paper layout at embed time, or use `_2x1` for
`figure*` (full-page-width) and `_1x1` for `figure` (column-width) blocks.

---

## 5. Saving

- DPI: `150`.
- Save with `bbox_inches="tight"` and explicit white facecolor:
  ```python
  fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
  ```
- Filename slugs must match the paper's caption labels (e.g., `judge_fig1_modality_gap`).

---

## 6. Annotation conventions

- **Bar value labels:** show every bar's exact value (`f"{v:.3f}"` for rates,
  `f"{v:.1%}"` for percentages). Place above the bar with `va="bottom"`.
- **Sample-size annotations:** `f"(n={n:,})"` next to the per-group label.
- **No emoji or unicode dingbats** in figure text. Plain ASCII only inside
  axis labels, titles, and legends; the paper PDF rendering chain doesn't
  carry custom fonts.

---

## 7. What to do when editing existing figures

- Apply Section 1 (palette), Section 2 (background), Section 4 (aspect
  ratios) to any figure you touch.
- If a figure currently has concatenated panels, **split it** into one PNG
  per panel before doing anything else.
- Update the LaTeX `\includegraphics` path to point at the new `_1x1` or
  `_2x1` file, depending on column vs. text-width context.
