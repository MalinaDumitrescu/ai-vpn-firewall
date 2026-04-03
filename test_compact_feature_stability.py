import pandas as pd
import unittest

from src.pipeline.feature_pipeline import COMPACT_FEATURES, FeaturePipeline


def _base_row(flow_id: str, capture_id: str, label: int, split: str, dataset: str) -> dict:
    return {
        "flow_id": flow_id,
        "capture_id": capture_id,
        "source_file": f"{capture_id}.pcap",
        "source_capture_id": capture_id,
        "label": label,
        "split": split,
        "dataset": dataset,
    }


class CompactFeatureStabilityTests(unittest.TestCase):
    def test_direction_features_not_zeroed_by_per_capture_normalization(self) -> None:
        rows = []

        # Capture A is almost fully upload-heavy, capture B is download-heavy.
        # If direction features are per-capture normalized, both collapse near zero.
        for i in range(2):
            r = _base_row(f"a{i}", "cap_a", 0, "train", "vnat")
            r.update(
                {
                    "sz_coef_variation": 0.2 + (0.01 * i),
                    "sz_p25_median_ratio": 0.8,
                    "sz_p75_median_ratio": 1.2,
                    "sz_iqr_norm_median": 0.4,
                    "dispersion_symmetry": 0.1,
                    "direction_balance_bytes": 0.99,
                    "direction_balance_packets": 1.0,
                }
            )
            rows.append(r)

        for i in range(2):
            r = _base_row(f"b{i}", "cap_b", 1, "train", "vnat")
            r.update(
                {
                    "sz_coef_variation": 0.3 + (0.01 * i),
                    "sz_p25_median_ratio": 0.85,
                    "sz_p75_median_ratio": 1.15,
                    "sz_iqr_norm_median": 0.35,
                    "dispersion_symmetry": -0.2,
                    "direction_balance_bytes": -0.98,
                    "direction_balance_packets": -1.0,
                }
            )
            rows.append(r)

        df = pd.DataFrame(rows)

        pipe = FeaturePipeline().fit(df)
        xt = pipe.transform(df)

        for col in ("direction_balance_bytes", "direction_balance_packets"):
            self.assertIn(col, xt.columns)
            # Signal should remain separable across captures.
            self.assertGreater(xt[col].nunique(), 1)


    def test_compact_features_present_in_pipeline_order(self) -> None:
        pipe = FeaturePipeline(
            feature_cols=list(COMPACT_FEATURES),
            scale_cols=list(COMPACT_FEATURES),
            passthrough_cols=[],
        )
        self.assertEqual(pipe.feature_cols, list(COMPACT_FEATURES))


if __name__ == "__main__":
    unittest.main()


