import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import streamlit as st

def split_data(df):
    x = df.drop(["Species"], axis=1)
    y = df["Species"]
    return train_test_split(x, y, test_size=0.2, random_state=23)

def train_model(x_train, y_train):
    grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 3, 5]
    }
    dt = DecisionTreeClassifier(random_state=10)
    grid_search = GridSearchCV(dt, grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, x_train, y_train, x_test, y_test):
    y_pred = model.predict(x_test)
    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    cv_scores = cross_val_score(model, x_test, y_test, cv=5, scoring='accuracy')
    return train_acc, test_acc, cm, class_report, cv_scores, y_pred

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_decision_tree(model, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names, ax=ax)
    return fig

# Streamlit UI

st.title("Iris Flower Classification App")

uploaded_file = st.file_uploader("Upload Iris CSV file", type=["csv"], key="iris_uploader")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Processing code here...
    x_train, x_test, y_train, y_test = split_data(df.drop(['Id'], axis=1))
    model = train_model(x_train, y_train)
    train_acc, test_acc, cm, class_report, cv_scores, y_pred = evaluate_model(model, x_train, y_train, x_test, y_test)

    st.markdown("---")
    st.write("### Model Evaluation")
    st.write(f"Training Accuracy: {train_acc:.2f}")
    st.write(f"Test Accuracy: {test_acc:.2f}")
    st.write("Classification Report:")
    st.text(class_report)

    st.markdown("---")
    st.write("### Cross-validation Scores")
    st.write(f"Scores: {cv_scores}")
    st.write(f"Mean Accuracy: {cv_scores.mean():.2f}")

    st.markdown("---")
    st.write("### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm, model.classes_))

    st.markdown("---")
    st.write("### Decision Tree Visualization")
    st.pyplot(plot_decision_tree(model, x_train.columns, model.classes_))


    # Collect each input separately from user
    sepal_length = st.number_input("Sepal Length", min_value=0.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0)
    petal_length = st.number_input("Petal Length", min_value=0.0)
    petal_width = st.number_input("Petal Width", min_value=0.0)

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]


#     a = np.array()

#     df1 = pd.DataFrame({
#     "Sepal_L": [st.number_input("Sepal Length", min_value=0.0)],
#     "Sepal_W": [st.number_input("Sepal Width", min_value=0.0)],
#     "Petal_L": [st.number_input("Petal Length", min_value=0.0)],
#     "Petal_W": [st.number_input("Petal Width", min_value=0.0)],
# })
    

    predict1 = st.button("Predict Species")
    if predict1:
      st.write(f""{model.predict(input_data)})
