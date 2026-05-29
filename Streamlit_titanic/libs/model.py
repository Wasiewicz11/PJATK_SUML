import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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


def train(n_estimators, max_depth, path2train, path2test, path2labels, path2pickle):
    train_df = pd.read_csv(path2train)
    test_df = pd.read_csv(path2test)
    labels_df = pd.read_csv(path2labels)
    test_df = test_df.merge(labels_df[['PassengerId', 'Survived']], on='PassengerId')

    X_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_train = train_df['Survived']
    X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_test = test_df['Survived']

    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['Sex', 'Embarked'])
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }

    pickle.dump(model, open(path2pickle, 'wb'))

    return metrics
