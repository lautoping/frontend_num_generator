import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo entrenado (Keras)
model = tf.keras.models.load_model("tensorflow_mnist.h5")

st.set_page_config(page_title="MNIST Classifier", layout="centered")
st.title("🖊️ Clasificador de Dígitos Manuscritos MNIST")
st.write("Elegí un número entre 1 y 9 para ver ejemplos reales del dataset MNIST y la predicción del modelo.")

# Menú desplegable para elegir el dígito
digit = st.selectbox("Seleccioná un dígito (1-9):", list(range(1, 10)))

# Cargar el dataset MNIST de prueba
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test[..., None]  # (N, 28, 28, 1)

# Seleccionar una imagen real del dígito elegido
indices = np.where(y_test == digit)[0]
if len(indices) > 0:
    idx = indices[0]
    img = x_test[idx]
    img_arr = img[None, ...]  # (1, 28, 28, 1)
    pred = model.predict(img_arr)
    pred_label = np.argmax(pred, axis=1)[0]
    st.image(img.squeeze(), caption=f"Predicción: {pred_label}", width=128, channels="GRAY")
    st.write(f"El modelo predice: **{pred_label}**")
else:
    st.warning("No se encontró una imagen para ese dígito en el dataset.")
