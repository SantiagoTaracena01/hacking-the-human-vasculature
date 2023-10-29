# Librerías necesarias para el desarrollo del proyecto.
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo de TensorFlow previamente entrenado.
model = 0 # tf.keras.models.load_model('ruta_del_modelo')

# Función para predecir la clase de una imagen.
def predict_image_class(image):

    # Preprocesar la imagen según los requisitos del modelo.
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = (image / 255.0)

    # Realizar la predicción.
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Obtener la etiqueta de clase con mayor probabilidad.
    class_index = np.argmax(prediction)
    return class_index

# Configuración de la aplicación Streamlit.
st.title("Clasificador de Imágenes")
st.write("Carga una imagen en formato .tif y te diré a qué clase pertenece")

# Cargar una imagen de entrada.
uploaded_image = st.file_uploader("Cargar imagen .tif", type=["tif", "tiff"])

# Subida de la imagen a utilizar.
if (uploaded_image is not None):

    # Carga de la imagen a predecir.
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Realizar la predicción.
    class_index = predict_image_class(image)

    # Muestra la clase predicha.
    st.write(f"Clase predicha: {class_index}")
