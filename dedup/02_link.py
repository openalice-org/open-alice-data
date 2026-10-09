"""Step 2: splink probabilistic dedup over the people_viz.json corpus.

Deliberately blocked on surname_key (same normalized surname) — a full
9352x9352 comparison isn't warranted for a "same person split across two
OpenAlex author records" question; near-duplicates virtually always share a
surname. Trains m/u probabilities unsupervised (EM) since there's no labeled
truth set, then scores every blocked pair and writes the ranked candidates
for 03_report.py to enrich with real coauthor evidence.
"""
import json
from pathlib import Path

import pandas as pd
import splink.comparison_level_library as cll
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"

df = pd.read_pickle(OUT_DIR / "people.pkl")
# splink comparisons need arrays as python lists (already are) and scalar
# nulls as None (pandas NaN for missing first_year/last_year/h_index is fine
# - duckdb backend handles it).
df["topics"] = df["topics"].apply(lambda v: v if isinstance(v, list) else [])
df["keywords"] = df["keywords"].apply(lambda v: v if isinstance(v, list) else [])

settings = SettingsCreator(
    link_type="dedupe_only",
    unique_id_column_name="id",
    blocking_rules_to_generate_predictions=[block_on("surname_key")],
    comparisons=[
        cl.JaroWinklerAtThresholds("name", score_threshold_or_thresholds=[0.95, 0.88, 0.75]),
        cl.CustomComparison(
            output_column_name="first_year",
            comparison_levels=[
                cll.NullLevel("first_year"),
                cll.AbsoluteDifferenceLevel("first_year", difference_threshold=0),
                cll.AbsoluteDifferenceLevel("first_year", difference_threshold=2),
                cll.AbsoluteDifferenceLevel("first_year", difference_threshold=5),
                cll.ElseLevel(),
            ],
        ),
        cl.CustomComparison(
            output_column_name="last_year",
            comparison_levels=[
                cll.NullLevel("last_year"),
                cll.AbsoluteDifferenceLevel("last_year", difference_threshold=0),
                cll.AbsoluteDifferenceLevel("last_year", difference_threshold=2),
                cll.AbsoluteDifferenceLevel("last_year", difference_threshold=5),
                cll.ElseLevel(),
            ],
        ),
        cl.ExactMatch("country").configure(term_frequency_adjustments=False),
        cl.ExactMatch("university").configure(term_frequency_adjustments=False),
        cl.JaroWinklerAtThresholds("institution", score_threshold_or_thresholds=[0.95, 0.8]),
        cl.ArrayIntersectAtSizes("topics", size_threshold_or_thresholds=[2, 1]),
        cl.ArrayIntersectAtSizes("keywords", size_threshold_or_thresholds=[3, 1]),
    ],
    retain_intermediate_calculation_columns=True,
)

db_api = DuckDBAPI()
linker = Linker(df, settings, db_api=db_api)

linker.training.estimate_probability_two_random_records_match(
    [block_on("surname_key")], recall=0.8
)
linker.training.estimate_u_using_random_sampling(max_pairs=2e6)
linker.training.estimate_parameters_using_expectation_maximisation(block_on("surname_key"))

predictions = linker.inference.predict(threshold_match_probability=0.1)
pred_df = predictions.as_pandas_dataframe()
pred_df = pred_df.sort_values("match_probability", ascending=False)
pred_df.to_pickle(OUT_DIR / "predictions.pkl")
print(f"scored pairs (match_probability >= 0.1): {len(pred_df)}")
print(pred_df[["id_l", "id_r", "name_l", "name_r", "match_probability"]].head(20).to_string())

with open(OUT_DIR / "model_settings.json", "w", encoding="utf-8") as f:
    json.dump(linker.misc.save_model_to_json(), f, indent=2)
