import pandas as pd
from obesity_predictor.core.validation.drift_detector import DriftDetector

def test_drift_detector(tmp_path):
    df_ref = pd.DataFrame({"Age": [20, 30], "Weight": [70, 80], "NObeyesdad": [0, 1]})
    df_new = pd.DataFrame({"Age": [22, 35], "Weight": [68, 85], "NObeyesdad": [0, 1]})
    detector = DriftDetector(target_column="NObeyesdad")
    out_file = tmp_path / "drift.html"
    detector.run(df_ref, df_new, output_path=str(out_file))
    assert out_file.exists()
