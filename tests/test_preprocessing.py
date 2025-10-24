import pytest
import pandas as pd
from obesity_predictor.core.preprocessing.train_preprocessor import TrainPreprocessor

def test_preprocessing_fit_transform():
    df = pd.DataFrame({
        "Age": [25, 40],
        "Height": [170, 180],
        "Weight": [70, 90],
        "Gender": ["Male", "Female"]
    })
    prep = TrainPreprocessor()
    prep.fit(df)
    transformed = prep.transform(df)
    assert not transformed.empty
    assert prep.fitted
