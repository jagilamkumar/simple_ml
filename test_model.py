
import joblib
import pandas as pd


def test_model_prediction():
    model = joblib.load("model.joblib")

    house = pd.DataFrame({"size_sqft": [900]})
    prediction = model.predict(house)

    assert len(prediction) == 1
    assert 0 < prediction[0] < 100
