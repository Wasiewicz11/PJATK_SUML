import pickle
import pandas as pd


def predict(passenger, model_path):
    model = pickle.load(open(model_path, 'rb'))

    data = pd.DataFrame([{
        "Pclass": passenger.pclass,
        "Sex": passenger.sex,
        "Age": passenger.age,
        "SibSp": passenger.sibsp,
        "Parch": passenger.parch,
        "Fare": passenger.fare,
        "Embarked": passenger.embarked,
    }])

    survived = model.predict(data)
    confidence = model.predict_proba(data)

    return int(survived[0]), round(float(confidence[0][survived[0]]) * 100, 2)
