import pytest
import pandas as pd
from obesity_predictor.core.model.xgboost_trainer import XGBoostTrainer

def test_xgboost_trainer_train(monkeypatch):
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y = [0, 1]
    trainer = XGBoostTrainer({"iterations": 1})
    model = trainer.train(X, y, X, y)
    assert model is not None
