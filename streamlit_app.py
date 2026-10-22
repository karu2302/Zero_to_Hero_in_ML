import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------- Custom Styling ----------
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🧠 TensorFlow Playground</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
        .css-18e3th9 {
            background-color: #f5f5f5;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar Inputs ----------
st.sidebar.title("Hyperparameter Tuning")

dataset = st.sidebar.selectbox("Dataset", ["moons", "circles", "blobs"])
activation = st.sidebar.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
lr = st.sidebar.number_input("Learning Rate", value=0.01, format="%.4f")
batch_size = st.sidebar.number_input("Batch Size", value=32, min_value=1)
train_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.7)

# ---------- Data Generation ----------
def generate_data(dataset_name):
    if dataset_name == "moons":
        return make_moons(n_samples=1000, noise=0.2, random_state=42)
    elif dataset_name == "circles":
        return make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    elif dataset_name == "blobs":
        return make_blobs(n_samples=1000, centers=2, cluster_std=1.5, random_state=42)
    else:
        raise ValueError("Unsupported dataset")

X, y = generate_data(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=42)

# ---------- Model Definition ----------
def build_model(activation, lr):
    model = Sequential([
        Dense(10, input_shape=(2,), activation=activation),
        Dense(10, activation=activation),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_model(activation, lr)

# ---------- Training ----------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=50,
    verbose=0
)

# ---------- Plot Decision Boundary ----------
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid, verbose=0).reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, levels=25, cmap="RdBu", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors='w')
    plt.title("Decision Boundary")
    st.pyplot(plt.gcf())
    plt.clf()

# ---------- Plot Loss ----------
def plot_loss_curves(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Testing Loss")
    st.pyplot(plt.gcf())
    plt.clf()

# ---------- Outputs ----------
st.subheader("🔍 Decision Boundary")
plot_decision_boundary(model, X, y)

st.subheader("📉 Training vs Testing Error")
plot_loss_curves(history)