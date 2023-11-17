# Librerías necesarias para el desarrollo del proyecto.
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Cargar el modelo de TensorFlow previamente entrenado.
model = tf.keras.models.load_model("./models/model.h5")

# Permitir el uso de gráficos en la aplicación.
st.set_option('deprecation.showPyplotGlobalUse', False)

# Función para predecir la clase de una imagen.
def predict_image_class(image):

    # Preprocesar la imagen según los requisitos del modelo.
    image = np.array(image)
    image = tf.image.resize(image, (128, 128))
    image = (image / 255.0)

    # Realizar la predicción.
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Obtener la etiqueta de clase con mayor probabilidad.
    class_index = np.argmax(prediction)
    return class_index

# Función para generar gráficos exploratorios de la imagen.
def generate_image_plots(image):

    st.subheader("Exploración de la Imagen")

    # Mostrar la imagen.
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Histograma de los valores de píxeles en escala de grises.
    st.subheader("Histograma de Píxeles")
    grayscale_image = image.convert('L')
    pixel_values = np.array(grayscale_image)
    plt.figure(figsize=(6, 4))
    sns.histplot(pixel_values.ravel(), kde=True, color='blue')
    st.pyplot()

    # Mapa de calor de los valores de píxeles.
    st.subheader("Mapa de Calor de Píxeles")
    plt.figure(figsize=(6, 6))
    sns.heatmap(pixel_values, cmap='viridis', cbar=True)
    st.pyplot()

    # Histograma de los valores de píxeles en escala de grises.
    grayscale_image = image.convert('L')
    pixel_values = np.array(grayscale_image)
    plt.figure(figsize=(6, 4))
    sns.histplot(pixel_values.ravel(), kde=True, color='blue')
    st.pyplot()

    # Histogramas de colores (rojo, verde, azul).
    st.subheader("Histogramas de Colores")
    r, g, b = image.split()
    r_values = np.array(r)
    g_values = np.array(g)
    b_values = np.array(b)
    plt.figure(figsize=(6, 4))
    plt.subplot(131)
    sns.histplot(r_values.ravel(), kde=True, color='red')
    plt.title("Rojo")
    plt.subplot(132)
    sns.histplot(g_values.ravel(), kde=True, color='green')
    plt.title("Verde")
    plt.subplot(133)
    sns.histplot(b_values.ravel(), kde=True, color='blue')
    plt.title("Azul")
    st.pyplot()

    # Visualización en escala de grises.
    st.subheader("Visualización en Escala de Grises")
    st.image(grayscale_image, caption="Imagen en Escala de Grises", use_column_width=True)

# Configuración de la aplicación Streamlit.
st.title("Clasificador de Imágenes")
st.write("Carga una imagen en formato .tif y te diré a qué clase pertenece")

# Cargar una imagen de entrada.
uploaded_image = st.file_uploader("Cargar imagen .tif", type=["tif", "tiff"])

# Subida de la imagen a utilizar.
if (uploaded_image is not None):

    # Carga de la imagen a predecir.
    image = Image.open(uploaded_image)

    # Realizar la predicción.
    class_index = predict_image_class(image)

    # Diccionario de clases.
    classes = ["Renal Cortex", "Renal Medulla", "Renal Papilla"]

    # Muestra la clase predicha.
    st.write(f"Clase predicha: {classes[class_index]}")

    # Generar gráficos exploratorios.
    generate_image_plots(image)
