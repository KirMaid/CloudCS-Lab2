# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict[str, int | float]:
    return {
        "culmen_length_mm": 36.7,
        "culmen_depth_mm": 19.3,
        "flipper_length_mm": 193.0,
        "body_mass_g": 3450.0,
        "sex": 1,
        "island_Biscoe": 0,
        "island_Dream": 0,
        "island_Torgersen": 1
    }


def test_make_inference(monkeypatch, create_data):
    def mock_get_predictions(_, data: pd.DataFrame) -> list[str]:
        assert create_data == {
            key: value[0] for key, value in data.to_dict("list").items()
        }
        return ["Adelie"]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    result = make_inference(in_model, create_data)
    assert result == {"species": "Adelie"}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    assert filepath_and_data[1] == load_model(filepath_and_data[0])
