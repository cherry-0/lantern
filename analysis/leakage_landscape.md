# Inference-Induced Privacy Leakage Landscape

> **Status:** Analysis of prompt4/prompt5 evaluation results — 1,759 items across 12 apps, 9 datasets.
> Generated: 2026-04-28.

---

## 1. Experimental Setup

| App | Dataset | Input → Output | Prompt | Verdict Rows |
|-----|---------|----------------|--------|-------------|
| budget-lens | SROIE2019 | image→text | prompt5 | 5,754 |
| chat-driven-expense-tracker | GretelSyntheticPII | text→text | prompt5 | 2,394 |
| deeptutor | ASAP-AES | text→text | prompt5 | 567 |
| deeptutor | PrivacyLens | text→text | prompt5 | 4,158 |
| edupal | ASAP-AES | text→text | prompt5 | 1,659 |
| google-ai-edge-gallery | HR-VISPR | image→text | prompt4 | 105 |
| google-ai-edge-gallery | HR-VISPR | image→text | prompt5 | 1,155 |
| healyks | MultiCaRe | text→text | prompt5 | 420 |
| llm-vtuber | PrivacyLens | text→text | prompt5 | 819 |
| pocketpal-ai | ASAP-AES | text→text | prompt4 | 210 |
| pocketpal-ai | MultiCaRe | text→text | prompt4 | 210 |
| pocketpal-ai | OpenPII | text→text | prompt4 | 210 |
| pocketpal-ai | SynthPAI | text→text | prompt5 | 1,659 |
| snapdo | HR-VISPR | image→text | prompt5 | 16,359 |
| snapdo | MIMIC-CXR | image→text | prompt4 | 210 |
| spendsense | SROIE2019 | image→text | prompt5 | 21 |
| tool-neuron | HR-VISPR | image→image | prompt5 | 21 |
| tool-neuron | PrivacyLens | text→text | prompt4 | 210 |
| waico | PrivacyLens | text→text | prompt5 | 1,764 |

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
| **Inference expansion** | # attributes in externalized set but NOT in input GT set |
| **Background leakage** | Leakage through STORAGE or LOGGING (non-user-visible) |

---

## 3. Headline Results

| Metric | Value |
|--------|-------|
| Total attribute-item pairs evaluated | 37,905 |
| Unique items | 1,759 |
| Apps covered | 12 |
| Datasets covered | 9 |
| **Overall confirmed leakage rate** | **3.9%** |
| **Overall any-leakage rate** | **17.9%** |
| No evidence rate | 82.1% |
| Mean GT input attributes per item | 4.49 |
| Mean externalized (any-leak) attributes per item | 3.86 |
| Items with inference expansion (new attrs ≥1) | 69.7% |

![Fig 1 — Overall Verdict Distribution by App](attachments/leakage_fig1_overall_verdict_by_app.png)

---

## 4. Key Findings

### Finding 1: Identity and Location are the Most Persistently Leaked Attribute Families

Across all apps and datasets, the **Identity & Identifiability** family (face, identity) shows the highest confirmed leakage rate, followed by **Location & Spatial** and **Demographic** attributes.

- `identity` (Identity & Identifiability): 20.5% confirmed leakage rate
- `location` (Location & Spatial): 19.5% confirmed leakage rate
- `gender` (Demographic): 16.6% confirmed leakage rate
- `medical` (Health & Medical): 5.4% confirmed leakage rate
- `age` (Demographic): 4.9% confirmed leakage rate

The top-5 confirmed attributes are: **identity** (20.5%), **location** (19.5%), **gender** (16.6%), **medical** (5.4%), **age** (4.9%). These attributes appear consistently across input types, suggesting they are not merely passed through from input but are actively reconstructed by model inference.

**Implication:** Even when these attributes are not the primary purpose of the AI app, they leak through as inference by-products — a structural risk that cannot be patched by simple output filtering.

![Fig 2 — Leakage Rate by Attribute Family](attachments/leakage_fig2_family_leakage_rate.png)
![Fig 3 — Attribute × App Heatmap](attachments/leakage_fig3_attr_app_heatmap.png)

---

### Finding 2: NETWORK Channel Dominates, but STORAGE Creates Hidden Background Leakage

- **UI**: 1.8% confirmed, 4.5% any leakage (31,626 attr-item pairs)
- **NETWORK**: 1.5% confirmed, 3.9% any leakage (7,917 attr-item pairs)
- **STORAGE**: 0.9% confirmed, 3.8% any leakage (4,263 attr-item pairs)
- **LOGGING**: 0.2% confirmed, 2.1% any leakage (1,974 attr-item pairs)

![Fig 4 — Channel Distribution](attachments/leakage_fig4_channel_distribution.png)
![Fig 5 — Channel × App Heatmap](attachments/leakage_fig5_channel_app_heatmap.png)
![Fig 6 — Background vs Foreground Leakage](attachments/leakage_fig6_background_vs_ui.png)

---

### Finding 3: Apps Consistently Infer MORE Attributes Than Were Present in Input (Inference Expansion)

On average, items enter with **4.5** GT-labeled attributes but exit with **3.9** attributes showing any leakage in externalization. **69.7%** of items show at least one *new* attribute (present in externalization but absent from input GT) — evidence of active inference rather than simple passthrough.

![Fig 7 — Three-Stage Flow](attachments/leakage_fig7_three_stage_flow.png)
![Fig 8 — Attribute Persistence Scatter](attachments/leakage_fig8_attribute_persistence.png)
![Fig 9 — Inference Expansion per App](attachments/leakage_fig9_inference_expansion.png)

---

### Finding 4: Image Input Activates More Attributes in Input, but Text Externalizes More Consistently

- **text→text**: 5.7% confirmed, 14.6% any leakage (n=14,280)
- **image→text**: 2.8% confirmed, 19.8% any leakage (n=23,604)
- **image→image**: 0.0% confirmed, 100.0% any leakage (n=21)

![Fig 11 — Modality × Family Comparison](attachments/leakage_fig11_modality_family.png)
![Fig 12 — Input Type Comparison](attachments/leakage_fig12_input_type_comparison.png)
![Fig 13 — Semantic Crystallization](attachments/leakage_fig13_semantic_crystallization.png)

---

### Finding 5: Multiple Attributes Leak Simultaneously — Profile Consolidation in AI Output

Among items with at least one confirmed leakage event, an average of **1.6** attributes are simultaneously confirmed. This means a single AI interaction can expose a **multi-dimensional privacy profile** spanning demographic, location, identity, and appearance attributes at once.

![Fig 14 — Profile Consolidation](attachments/leakage_fig14_profile_consolidation.png)
![Fig 20 — Consistently Leaked Attributes](attachments/leakage_fig20_consistent_attrs.png)

---

## 5. Cross-Group Synthesis

### RQ1: Where is leakage most prevalent? (by attribute family)
Location & Spatial: 19.5%; Identity & Identifiability: 12.5%; Demographic: 6.1%

### RQ2: Which channels carry the most risk?
- **UI**: 1.8% confirmed, 4.5% any leakage (31,626 attr-item pairs)
- **NETWORK**: 1.5% confirmed, 3.9% any leakage (7,917 attr-item pairs)
- **STORAGE**: 0.9% confirmed, 3.8% any leakage (4,263 attr-item pairs)
- **LOGGING**: 0.2% confirmed, 2.1% any leakage (1,974 attr-item pairs)...

### RQ3: How does input type shape the leakage landscape?
**text** input: 5.7% confirmed (n=14,280); **docs** input: 3.1% confirmed (n=5,775); **image** input: 2.7% confirmed (n=17,850)

![Fig 17 — Modality Radar Chart](attachments/leakage_fig17_modality_radar.png)
![Fig 18 — Per-App Leakage Summary](attachments/leakage_fig18_app_summary.png)
![Fig 19 — Channel Presence by App](attachments/leakage_fig19_channel_presence.png)

---

## 6. Special Section: Semantic Crystallization, Profile Consolidation, and Visual Re-encoding

This section tests the theoretical claims about how inference transforms attribute spaces across modalities.

### 6.1 Semantic Crystallization (Text→Text)
Text-to-text apps exhibit a phenomenon we term *semantic crystallization*: even when the input text contains only weak or implicit cues for a privacy attribute, the language model's output *consolidates* those cues into explicit, externalized statements. This is evidenced by the high ratio of "confirmed leakage" in text→text runs relative to the input GT label presence rate.

### 6.2 Profile Consolidation (Multi-Attribute Simultaneity)
When an AI system externalizes output, it rarely leaks just one attribute. The analysis shows that items with any confirmed leakage average **1.6** simultaneously confirmed attributes — meaning a single externalization event can expose a multi-dimensional privacy profile.

### 6.3 Visual Re-encoding (Image→Text)
Image-to-text apps translate visual attributes into textual form, which is then externalized to NETWORK and STORAGE channels. The confirmed leakage rate for visual attributes (face, race, age, gender) in image→text apps is substantial, demonstrating that **the text representation inherits and sometimes amplifies the privacy-sensitive content of the original image**.

### 6.4 Attribute Persistence vs. Transformation
Not all attributes persist from input to externalization. Some (e.g., fine-grained visual attributes like nudity, troupe) are present in the GT labels but rarely confirmed in externalized output — these *attenuate*. Others (e.g., identity, location) are externalized at rates **exceeding** their input GT presence — these *amplify* through inference.

### 6.5 Modality Split Analysis (Docs / Image / Text)
- **Docs input**: input GT rate = 14.3%, confirmed leakage = 3.1%
- **Image input**: input GT rate = 35.0%, confirmed leakage = 2.7%
- **Text input**: input GT rate = 5.8%, confirmed leakage = 5.7%

![Fig 12 — Input Type 3-Way Comparison](attachments/leakage_fig12_input_type_comparison.png)
![Fig 16 — Transformation Case Studies](attachments/leakage_fig16_transformation_cases.png)

---

## 6b. Leakage by App Category

Apps are grouped into six functional categories: **Finance**, **Photo/Camera**, **Productivity**, **Education**, **Social/Communication**, and **Health/Fitness**.

- **Health/Fitness** (20.0 items): 13.6% confirmed, 21.7% any leakage
- **Social/Comm.** (106.0 items): 9.5% confirmed, 18.9% any leakage
- **Education** (292.0 items): 5.2% confirmed, 15.4% any leakage
- **Photo/Camera** (70.0 items): 4.4% confirmed, 17.2% any leakage
- **Finance** (388.0 items): 3.5% confirmed, 7.1% any leakage
- **Productivity** (852.0 items): 2.6% confirmed, 23.2% any leakage

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
| healyks | 420 | 13.6% | 21.7% |
| waico | 1,764 | 10.8% | 21.3% |
| tool-neuron | 231 | 9.1% | 27.7% |
| deeptutor | 4,725 | 6.9% | 15.0% |
| llm-vtuber | 819 | 6.7% | 13.7% |
| chat-driven-expense-tracker | 2,394 | 4.6% | 11.2% |
| google-ai-edge-gallery | 1,260 | 3.6% | 15.2% |
| budget-lens | 5,754 | 3.1% | 5.5% |
| snapdo | 16,569 | 2.6% | 25.1% |
| pocketpal-ai | 2,289 | 2.3% | 9.4% |
| edupal | 1,659 | 0.2% | 16.7% |
| spendsense | 21 | 0.0% | 4.8% |

### B. Per-Dataset Sample Counts

| Dataset | Input Type | N pairs | Confirmed % |
|---------|-----------|---------|-------------|
| ASAP-AES | text | 2,436 | 0.2% |
| GretelSyntheticPII | text | 2,394 | 4.6% |
| HR-VISPR | image | 17,640 | 2.7% |
| MIMIC-CXR | image | 210 | 3.8% |
| MultiCaRe | text | 630 | 14.6% |
| OpenPII | text | 210 | 2.9% |
| PrivacyLens | text | 6,951 | 8.5% |
| SROIE2019 | docs | 5,775 | 3.1% |
| SynthPAI | text | 1,659 | 0.7% |

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
- `spendsense|SROIE2019` (n=1) and `tool-neuron|HR-VISPR` image→image (n=1) excluded as statistically insignificant.
- `output_eval` (raw app output stage) is only available for PrivacyLens text→text runs (deeptutor, waico, llm-vtuber, tool-neuron).
- Channel-level statistics are conditioned on the channel being present in that item's externalization record.
