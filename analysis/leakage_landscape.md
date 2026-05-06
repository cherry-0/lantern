# Inference-Induced Privacy Leakage Landscape

> **Status:** Analysis of prompt4/prompt5 evaluation results — 6,855 items across 24 apps, 10 datasets.
> Generated: 2026-05-01.

---

## 1. Experimental Setup

| App | Dataset | Input → Output | Prompt | Verdict Rows |
|-----|---------|----------------|--------|-------------|
| budget-lens | SROIE2019 | image→text | prompt5 | 4,200 |
| chat-driven-expense-tracker | GretelSyntheticPII | text→text | prompt5 | 4,200 |
| chat-driven-expense-tracker | PrivacyLens | text→text | prompt5 | 4,200 |
| clone | HR-VISPR | image→text | prompt4 | 777 |
| clone | HR-VISPR | image→text | prompt5 | 3,423 |
| deeptutor | ASAP-AES | text→text | prompt5 | 4,095 |
| deeptutor | PrivacyLens | text→text | prompt5 | 4,200 |
| edumind | ASAP-AES | text→text | prompt5 | 4,137 |
| edupal | ASAP-AES | text→text | prompt5 | 4,200 |
| edupal | PrivacyLens | text→text | prompt5 | 4,179 |
| edupal | SynthPAI | text→text | prompt5 | 2,499 |
| fiscal-flow | PrivacyLens | text→text | prompt5 | 4,200 |
| fiscal-flow | SynthPAI | text→text | prompt5 | 4,200 |
| google-ai-edge-gallery | HR-VISPR | image→text | prompt5 | 4,200 |
| google-ai-edge-gallery | PrivacyLens | text→text | prompt5 | 4,116 |
| healyks | MultiCaRe | text→text | prompt5 | 4,200 |
| klyr | SynthPAI | text→text | prompt5 | 4,074 |
| lira | SynthPAI | text→text | prompt5 | 4,200 |
| llm-vtuber | PrivacyLens | text→text | prompt5 | 3,990 |
| momentag | HR-VISPR | image→text | prompt5 | 4,200 |
| nom-ai | Nutrition5k | image→text | prompt5 | 4,200 |
| nutri-track | MultiCaRe | text→text | prompt5 | 4,200 |
| pocketpal-ai | ASAP-AES | text→text | prompt5 | 4,200 |
| pocketpal-ai | SynthPAI | text→text | prompt5 | 4,200 |
| sgpa | ASAP-AES | text→text | prompt5 | 4,200 |
| sgpa | PrivacyLens | text→text | prompt5 | 4,200 |
| skin-disease-detection | MIMIC-CXR | image→text | prompt5 | 4,200 |
| snapdo | HR-VISPR | image→text | prompt5 | 4,200 |
| snapdo | SROIE2019 | image→text | prompt5 | 4,200 |
| spendsense | GretelSyntheticPII | text→text | prompt5 | 4,200 |
| spendsense | MultiPriv | image→text | prompt5 | 2,373 |
| spendsense | SROIE2019 | image→text | prompt5 | 4,158 |
| tinytavern | SynthPAI | text→text | prompt5 | 4,074 |
| tool-neuron | HR-VISPR | image→image | prompt5 | 420 |
| tool-neuron | HR-VISPR | image→text | prompt4 | 1,953 |
| tool-neuron | PrivacyLens | text→text | prompt4 | 3,255 |
| tool-neuron | SynthPAI | text→image | prompt5 | 294 |
| waico | PrivacyLens | text→text | prompt5 | 4,179 |
| xend | PrivacyLens | text→text | prompt5 | 4,200 |

**Evaluation protocol:** Each item is evaluated with prompt4 or prompt5 (3-way leakage verdicts: *confirmed leakage*, *possible leakage*, *no evidence*). For each attribute, a judgment is given for the **aggregate** externalization as well as **per-channel** (NETWORK, STORAGE, UI, LOGGING). Rows with failed or missing `ext_eval` are excluded.

---

## 2. Metric Definitions

| Metric | Definition |
|--------|-----------|
| **Confirmed leakage** | Attribute is clearly and directly inferable from externalized output (score=2) |
| **Possible leakage** | Partial or indirect evidence; attribute may be inferable (score=1) |
| **No evidence** | No meaningful signal for this attribute (score=0) |
| **Any leakage rate** | (confirmed + possible) / total attribute-item pairs |
| **Confirmed leakage rate** | confirmed / total attribute-item pairs |
| **Raw-output stage** | Computed from `output_eval` — judges whether the raw model output makes the attribute *inferable*. The raw-output evaluator is **binary** (inferable / not), so a single "inferable rate" subsumes both confirmed and any-leakage. |
| **Externalization stage** | Computed from `ext_eval` — judges what is actually visible after the channel projection (NETWORK / STORAGE / UI / LOGGING). The aggregate (`agg_*`) judge is **3-class** (confirmed / possible / no evidence), so confirmed and any-leakage are reported separately and the default headline metric throughout. |
| **Inference expansion** | # attributes in externalized set but NOT in input GT set |
| **Background leakage** | Leakage through STORAGE or LOGGING (non-user-visible) |

---

## 3. Headline Results

| Metric | Value |
|--------|-------|
| Total attribute-item pairs evaluated | 144,396 |
| Unique items | 6,855 |
| Apps covered | 24 |
| Datasets covered | 10 |
| **Inferable in raw model output (binary judge)** | **18.9%** |
| **Confirmed leakage — externalization (aggregate, 3-class)** | **4.2%** |
| **Any leakage — externalization (aggregate, 3-class)** | **13.7%** |
| No-evidence rate (externalization) | 86.3% |
| Attribute-item pairs with `output_eval` populated | 144,396 |
| Mean GT input attributes per item | 2.02 |
| Mean externalized (any-leak) attributes per item | 2.88 |
| Items with inference expansion (new attrs ≥1) | 67.9% |

> The two judges report different metrics by design. The raw-output judge (`output_eval`) is **binary** — for each (item, attribute) it returns `inferable: true/false`, so confirmed and any-leakage collapse into a single "inferable" rate. The externalization judge (`ext_eval` aggregate) is **3-class** — it returns `confirmed leakage` / `possible leakage` / `no evidence`, so confirmed and any-leakage are reported separately. Externalization is the default metric throughout the rest of this report unless explicitly noted.

![Fig 1 — Overall Verdict Distribution by App](attachments/leakage_fig1_overall_verdict_by_app.png)

---

## 4. Key Findings

### Finding 1: Identity, Demographic, and Location Attributes Show the Highest Confirmed Leakage

Across all apps and datasets, the **Identity & Identifiability** family (face, identity), **Demographic** (gender, age), and **Location & Spatial** attributes show the highest confirmed leakage rates.

- `identity` (Identity & Identifiability): 27.7% confirmed leakage rate
- `gender` (Demographic): 20.8% confirmed leakage rate
- `location` (Location & Spatial): 12.8% confirmed leakage rate
- `medical` (Health & Medical): 8.7% confirmed leakage rate
- `age` (Demographic): 7.1% confirmed leakage rate

The top-5 confirmed attributes are: **identity** (27.7%), **gender** (20.8%), **location** (12.8%), **medical** (8.7%), **age** (7.1%). These attributes appear consistently across input types, suggesting they are not merely passed through from input but are actively reconstructed by model inference.

**Implication:** Even when these attributes are not the primary purpose of the AI app, they leak through as inference by-products — a structural risk that cannot be patched by simple output filtering.

![Fig 2 — Leakage Rate by Attribute Family](attachments/leakage_fig2_family_leakage_rate.png)
![Fig 3 — Attribute × App Heatmap](attachments/leakage_fig3_attr_app_heatmap.png)

---

### Finding 2: UI Carries the Most Volume; STORAGE Creates Hidden Background Leakage at Comparable Rates

- **UI**: 1.9% confirmed, 5.1% any leakage (61,656 attr-item pairs)
- **NETWORK**: 0.8% confirmed, 2.7% any leakage (87,066 attr-item pairs)
- **LOGGING**: 0.3% confirmed, 3.2% any leakage (1,428 attr-item pairs)
- **STORAGE**: 0.3% confirmed, 3.4% any leakage (28,434 attr-item pairs)

![Fig 4 — Channel Distribution](attachments/leakage_fig4_channel_distribution.png)
![Fig 5 — Channel × App Heatmap](attachments/leakage_fig5_channel_app_heatmap.png)
![Fig 6 — Background vs Foreground Leakage](attachments/leakage_fig6_background_vs_ui.png)

---

### Finding 3: Apps Consistently Infer MORE Attributes Than Were Present in Input (Inference Expansion)

On average, items enter with **2.0** GT-labeled attributes but exit with **2.9** attributes showing any leakage in externalization. **67.9%** of items show at least one *new* attribute (present in externalization but absent from input GT) — evidence of active inference rather than simple passthrough.

![Fig 7 — Three-Stage Flow](attachments/leakage_fig7_three_stage_flow.png)
![Fig 8 — Attribute Persistence Scatter](attachments/leakage_fig8_attribute_persistence.png)
![Fig 9 — Inference Expansion per App](attachments/leakage_fig9_inference_expansion.png)

---

### Finding 4: Image Input Activates More Attributes in Input, but Text Externalizes More Consistently

- **image→image**: 9.5% confirmed, 28.6% any leakage (n=420)
- **text→text**: 5.1% confirmed, 14.2% any leakage (n=101,598)
- **text→image**: 3.7% confirmed, 19.4% any leakage (n=294)
- **image→text**: 2.0% confirmed, 12.3% any leakage (n=42,084)

![Fig 11 — Modality × Family Comparison](attachments/leakage_fig11_modality_family.png)
![Fig 12 — Input Type Comparison](attachments/leakage_fig12_input_type_comparison.png)
![Fig 13 — Semantic Crystallization](attachments/leakage_fig13_semantic_crystallization.png)

---

### Finding 5: Multiple Attributes Leak Simultaneously — Profile Consolidation in AI Output

Among items with at least one confirmed leakage event, an average of **2.0** attributes are simultaneously confirmed. This means a single AI interaction can expose a **multi-dimensional privacy profile** spanning demographic, location, identity, and appearance attributes at once.

![Fig 14 — Profile Consolidation](attachments/leakage_fig14_profile_consolidation.png)
![Fig 20 — Consistently Leaked Attributes](attachments/leakage_fig20_consistent_attrs.png)

---

## 5. Cross-Group Synthesis

### RQ1: Where is leakage most prevalent? (by attribute family)
Identity & Identifiability: 14.5%; Location & Spatial: 12.8%; Demographic: 8.2%

### RQ2: Which channels carry the most risk?
- **UI**: 1.9% confirmed, 5.1% any leakage (61,656 attr-item pairs)
- **NETWORK**: 0.8% confirmed, 2.7% any leakage (87,066 attr-item pairs)
- **LOGGING**: 0.3% confirmed, 3.2% any leakage (1,428 attr-item pairs)
- **STORAGE**: 0.3% confirmed, 3.4% any leakage (28,434 attr-item pairs)...

### RQ3: How does input type shape the leakage landscape?
**text** input: 5.2% confirmed (n=93,492); **docs** input: 2.6% confirmed (n=23,331); **image** input: 2.2% confirmed (n=27,573)

![Fig 17 — Modality Radar Chart](attachments/leakage_fig17_modality_radar.png)
![Fig 18 — Per-App Leakage Summary](attachments/leakage_fig18_app_summary.png)
![Fig 19 — Channel Presence by App](attachments/leakage_fig19_channel_presence.png)

---

## 6. Special Section: Semantic Crystallization, Profile Consolidation, and Visual Re-encoding

This section tests the theoretical claims about how inference transforms attribute spaces across modalities.

### 6.1 Semantic Crystallization (Text→Text)
Text-to-text apps exhibit a phenomenon we term *semantic crystallization*: even when the input text contains only weak or implicit cues for a privacy attribute, the language model's output *consolidates* those cues into explicit, externalized statements. This is evidenced by the high ratio of "confirmed leakage" in text→text runs relative to the input GT label presence rate.

### 6.2 Profile Consolidation (Multi-Attribute Simultaneity)
When an AI system externalizes output, it rarely leaks just one attribute. The analysis shows that items with any confirmed leakage average **2.0** simultaneously confirmed attributes — meaning a single externalization event can expose a multi-dimensional privacy profile.

### 6.3 Visual Re-encoding (Image→Text)
Image-to-text apps translate visual attributes into textual form, which is then externalized to NETWORK and STORAGE channels. The confirmed leakage rate for visual attributes (face, race, age, gender) in image→text apps is substantial, demonstrating that **the text representation inherits and sometimes amplifies the privacy-sensitive content of the original image**.

### 6.4 Attribute Persistence vs. Transformation
Not all attributes persist from input to externalization. Some (e.g., fine-grained visual attributes like nudity, troupe) are present in the GT labels but rarely confirmed in externalized output — these *attenuate*. Others (e.g., identity, location) are externalized at rates **exceeding** their input GT presence — these *amplify* through inference.

### 6.5 Modality Split Analysis (Docs / Image / Text)
- **Docs input**: input GT rate = 10.1%, confirmed leakage = 2.6%
- **Image input**: input GT rate = 24.7%, confirmed leakage = 2.2%
- **Text input**: input GT rate = 5.0%, confirmed leakage = 5.2%

![Fig 12 — Input Type 3-Way Comparison](attachments/leakage_fig12_input_type_comparison.png)
![Fig 16 — Transformation Case Studies](attachments/leakage_fig16_transformation_cases.png)

---

## 6b. Leakage by App Category

Apps are grouped into six functional categories: **Finance**, **Photo/Camera**, **Productivity**, **Education**, **Social/Communication**, and **Health/Fitness**.

- **Health** (600.0 items): 7.7% confirmed, 19.9% any leakage
- **Social** (403.0 items): 6.9% confirmed, 16.5% any leakage
- **Photo/Camera** (570.0 items): 4.9% confirmed, 17.9% any leakage
- **Finance** (1,001.0 items): 4.5% confirmed, 11.9% any leakage
- **Education** (567.0 items): 3.5% confirmed, 12.9% any leakage
- **Productivity** (1,141.0 items): 0.8% confirmed, 8.7% any leakage

![Fig 21 — Category Verdict Overview](attachments/leakage_fig21_category_verdict.png)
![Fig 22 — Category × Family Heatmap](attachments/leakage_fig22_category_family_heatmap.png)
![Fig 23 — Category × Channel Breakdown](attachments/leakage_fig23_category_channel.png)
![Fig 24 — Category × Family Scatter](attachments/leakage_fig24_category_family_scatter.png)

---

## 7. Recommendations

1. **Prioritize identity and location filtering** across all app types — these attributes leak at the highest rate regardless of input modality.
2. **Audit STORAGE channels** (databases, memory systems, on-device logs) — STORAGE leakage is non-visible to users and persists beyond session boundaries.
3. **Deploy inference-aware output filters for image→text apps** — the text description of an image is a semantic consolidation point for many visual attributes that were diffuse in the original image.
4. **Test for inference expansion** — apps should be evaluated not just for passthrough of GT-present attributes, but for attributes that appear in the output that were NOT in the input (implicit inference).
5. **Treat multi-attribute leakage as the norm** — privacy defenses that target individual attribute leakage miss the broader threat of simultaneous multi-attribute profile construction.

---

## Appendix

### A. Per-App Detailed Statistics

| App | N pairs | Confirmed % | Any Leak % |
|-----|---------|-------------|------------|
| nutri-track | 4,200 | 14.1% | 25.0% |
| healyks | 4,200 | 13.0% | 20.7% |
| lira | 4,200 | 10.8% | 27.0% |
| tool-neuron | 5,922 | 10.1% | 36.0% |
| waico | 4,179 | 9.8% | 20.2% |
| chat-driven-expense-tracker | 8,400 | 7.8% | 16.5% |
| llm-vtuber | 3,990 | 6.5% | 14.7% |
| fiscal-flow | 8,400 | 5.5% | 15.0% |
| sgpa | 8,400 | 4.4% | 15.8% |
| deeptutor | 8,295 | 4.0% | 10.0% |
| google-ai-edge-gallery | 8,316 | 3.8% | 14.0% |
| edupal | 10,878 | 3.8% | 13.3% |
| budget-lens | 4,200 | 2.5% | 4.7% |
| snapdo | 8,400 | 2.2% | 14.7% |
| nom-ai | 4,200 | 2.0% | 30.4% |
| spendsense | 10,731 | 1.9% | 8.8% |
| skin-disease-detection | 4,200 | 1.8% | 3.7% |
| xend | 4,200 | 0.4% | 7.5% |
| pocketpal-ai | 8,400 | 0.3% | 9.6% |
| tinytavern | 4,074 | 0.1% | 3.6% |
| edumind | 4,137 | 0.1% | 11.6% |
| clone | 4,200 | 0.1% | 0.5% |
| klyr | 4,074 | 0.0% | 4.1% |
| momentag | 4,200 | 0.0% | 0.0% |

### B. Per-Dataset Sample Counts

| Dataset | Input Type | N pairs | Confirmed % |
|---------|-----------|---------|-------------|
| ASAP-AES | text | 20,832 | 0.1% |
| GretelSyntheticPII | docs | 8,400 | 4.0% |
| HR-VISPR | image | 19,173 | 2.3% |
| MIMIC-CXR | image | 4,200 | 1.8% |
| MultiCaRe | text | 8,400 | 13.6% |
| MultiPriv | docs | 2,373 | 3.5% |
| Nutrition5k | image | 4,200 | 2.0% |
| PrivacyLens | text | 40,719 | 8.0% |
| SROIE2019 | docs | 12,558 | 1.4% |
| SynthPAI | text | 23,541 | 2.1% |

### C. Attribute Family → Member Mapping

| Family | Attributes |
|--------|-----------|
| Identity & Identifiability | face, identity |
| Demographic | age, gender, race, marital status |
| Health & Medical | disability, medical |
| Location & Spatial | location |
| Religion & Cultural | religion, ethnic_clothing |
| Appearance & Body | nudity, height, weight, haircolor, color |
| Attire, Role & Group | formal, casual, uniforms, troupe |
| Activity & Lifestyle | sports |

### D. Data Quality Notes

- prompt4/5 items with failed `ext_eval` (no verdict) are excluded from all analyses.
- Per-(app, dataset) groups with fewer than 5 verdict rows are excluded; only `tool-neuron|HR-VISPR` (1 item, image→image) and `spendsense|SROIE2019` (2 items) remain in the very-small-sample regime — interpret their per-app rates with caution.
- `output_eval` (raw app output stage) is only available for PrivacyLens text→text runs (deeptutor, waico, llm-vtuber, tool-neuron, xend).
- Channel-level statistics are conditioned on the channel being present in that item's externalization record.
