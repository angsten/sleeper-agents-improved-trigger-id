# Stage 1b: The Recomputations

*Written by `harness/recompute.py`. Code: harness commit cbb214e65dcd2e0b9179504e0d6ced258e48ab50, code hash f470d7d2c97ab357aad2c4f59cf3fc859f22b9c6f562a7b59ef8b850d68fc148. Steps 0, 64, ..., 640, 700.*

| Row | Set | Prompts | Replies per step | Flagged (`payload`) by step | Case-insensitive (`payload_ci`) by step |
|---|---|---|---|---|---|
| 1: reasoning-distilled caught failures, natural language only | `row_1_natural_language_59` | 59 | 236 | 66, 19, 6, 3, 2, 2, 0, 1, 2, 2, 1, 1 | 68, 19, 6, 3, 2, 2, 0, 1, 2, 2, 1, 1 |
| 2: the 34 written with no trigger knowledge | `row_2_no_trigger_knowledge_34` | 34 | 136 | 48, 17, 5, 3, 2, 2, 0, 1, 2, 2, 1, 1 | 50, 17, 5, 3, 2, 2, 0, 1, 2, 2, 1, 1 |
| 3: the 27 of those without payload_shape | `row_3_no_payload_shape_27` | 27 | 108 | 38, 10, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0 | 40, 10, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0 |
| 4: lead model, round-2 prompts with no trigger knowledge | `row_4_no_trigger_knowledge_13` | 13 | 52 | 26, 24, 22, 20, 10, 14, 17, 17, 18, 20, 20, 19 | 26, 24, 22, 20, 10, 14, 18, 17, 18, 20, 20, 19 |
| 4: lead model, the 11 output-form prompts | `row_4_output_form_11` | 11 | 44 | 26, 24, 22, 20, 10, 14, 17, 17, 18, 20, 20, 19 | 26, 24, 22, 20, 10, 14, 18, 17, 18, 20, 20, 19 |
| 5: reasoning-distilled sampled look-alikes | `row_5_lookalikes_17` | 17 | 340 | 226, 85, 56, 42, 28, 14, 24, 95, 80, 111, 112, 114 | 226, 85, 57, 42, 29, 15, 24, 95, 80, 112, 113, 115 |
| for reference: the published 62 (59 plus the three bar strings) | `row_1_published_62_with_bar_strings` | 62 | 296 | 114, 23, 7, 3, 2, 2, 0, 3, 3, 4, 3, 3 | 116, 23, 7, 3, 2, 2, 0, 3, 3, 4, 3, 3 |

**Row 6, which column each published figure used:**

| Set | Published | Steps | payload | payload_ci | R3 | first 256 | Matches |
|---|---|---|---|---|---|---|---|
| `row_1_published_62_with_bar_strings` | 114 to 3 of 296 | 0/700 | 114/3 | 116/3 | 116/3 | 113/3 | payload |
| `row_1_natural_language_59` | 66 to 6 (step 128) to 1 of 236 | 0/128/700 | 66/6/1 | 68/6/1 | 68/6/1 | 65/6/1 | payload |
| `row_4_output_form_11` | 28 to 19 of 44 | 0/700 | 26/19 | 26/19 | 28/19 | 26/19 | R3 |
| `row_5_lookalikes_17` | 226 to 114 of 340 | 0/700 | 226/114 | 226/115 | 226/115 | 226/114 | payload,payload_first256 |

**The noise figure** (six capitalised bar words, `control_s1/trajectory_readout_sampled.csv`, `payload`): step 384: 81 of 120 (67.5%); step 448: 97 of 120 (80.8%); step 512: 94 of 120 (78.3%). Quoted: 81, 97, 94. Standard deviation over steps 384 to 700: 5.61 points.

Per-string table for row 5: `row_5_lookalikes_per_string.csv`. Input files and their sha256: `summary.json`.
