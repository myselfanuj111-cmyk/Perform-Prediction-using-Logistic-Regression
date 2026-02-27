
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Titanic Survival", layout="centered")
st.title("🚢 Titanic Survival Prediction")

file = st.file_uploader("Upload Titanic CSV", type="csv")

if file is not None:

    df = pd.read_csv(file)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df.dropna(subset=["Embarked"], inplace=True)

    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    features = [
        "Pclass", "Age", "SibSp", "Parch", "Fare",
        "Sex_male", "Embarked_Q", "Embarked_S"
    ]

    X = df[features]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    st.success(f"Accuracy: {acc:.2f}")


    cm = confusion_matrix(y_test, pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)


    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values[:50], label="Actual")
    ax2.plot(pred[:50], label="Predicted")

    ax2.set_title("Actual vs Predicted (First 50)")
    ax2.legend()

    st.pyplot(fig2)
