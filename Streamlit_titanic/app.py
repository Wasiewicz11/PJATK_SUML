import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path

startTime = datetime.now()
# import znanych nam bibliotek

model_filename = "our_titanic_model.pkl"
model_path = Path(__file__).parent / "models" / model_filename
model = pickle.load(open(model_path, 'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {"female": "Kobieta", "male": "Mężczyzna"}
pclass_d = {1: "Pierwsza", 2: "Druga", 3: "Trzecia"}
embarked_d = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}


# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem
def main():
    st.set_page_config(page_title="Czy przeżyłbyś katastrofę?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image(
        "https://media1.popsugar-assets.com/files/thumbor/7CwCuGAKxTrQ4wPyOBpKjSsd1JI/fit-in/2048xorig/filters:format_auto-!!-:strip_icc-!!-/2017/04/19/743/n/41542884/5429b59c8e78fbc4_MCDTITA_FE014_H_1_.JPG")

    with overview:
        st.title("Czy przeżyłbyś katastrofę?")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        pclass_radio = st.radio("Klasa", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        embarked_radio = st.radio("Port zaokrętowania", list(embarked_d.keys()), index=2,
                                  format_func=lambda x: embarked_d[x])

    with right:
        age_slider = st.slider("Wiek", value=50, min_value=1, max_value=100)
        sibsp_slider = st.slider("# Liczba rodzeństwa i/lub partnera", min_value=0, max_value=8)
        parch_slider = st.slider("# Liczba rodziców i/lub dzieci", min_value=0, max_value=6)
        fare_slider = st.slider("Cena biletu", min_value=0, max_value=500, step=10)

    data = pd.DataFrame([{
        "Pclass": pclass_radio,
        "Sex": sex_radio,
        "Age": age_slider,
        "SibSp": sibsp_slider,
        "Parch": parch_slider,
        "Fare": fare_slider,
        "Embarked": embarked_radio,
    }])
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.header("Czy dana osoba przeżyje? {0}".format("Tak" if survival[0] == 1 else "Nie"))
        st.subheader("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()

## Źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic), zastosowanie przez Adama Ramblinga
