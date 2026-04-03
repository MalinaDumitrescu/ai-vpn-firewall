import numpy as np
import pandas as pd
import unittest

from src.eval.calibration import fit_calibrator_from_df


class CalibrationFallbackTests(unittest.TestCase):
    def test_calibration_falls_back_when_val_has_single_class(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["val", "val", "val", "train", "train", "train", "train"],
                "label": [0, 0, 0, 0, 1, 0, 1],
                "p_raw": [0.1, 0.2, 0.3, 0.05, 0.9, 0.2, 0.85],
            }
        )

        cal = fit_calibrator_from_df(
            df,
            prob_col="p_raw",
            label_col="label",
            split_col="split",
            fit_split="val",
            method="platt",
            fallback_splits=("train",),
        )

        self.assertEqual(cal.metadata["fit_split_requested"], "val")
        self.assertEqual(cal.metadata["fit_split_used"], "train")
        self.assertTrue(cal.metadata["fallback_used"])
        self.assertEqual(set(cal.metadata["class_counts"].keys()), {0, 1})

        out = cal.predict(np.array([0.1, 0.8], dtype=float))
        self.assertEqual(out.shape, (2,))
        self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_calibration_raises_if_no_candidate_has_two_classes(self) -> None:
        df = pd.DataFrame(
            {
                "split": ["val", "val", "train", "train"],
                "label": [0, 0, 0, 0],
                "p_raw": [0.1, 0.2, 0.3, 0.4],
            }
        )

        with self.assertRaisesRegex(ValueError, "at least 2 classes"):
            fit_calibrator_from_df(
                df,
                prob_col="p_raw",
                label_col="label",
                split_col="split",
                fit_split="val",
                method="isotonic",
                fallback_splits=("train",),
            )


if __name__ == "__main__":
    unittest.main()


