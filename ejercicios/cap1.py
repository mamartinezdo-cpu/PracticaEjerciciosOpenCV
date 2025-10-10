import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Capítulo 1 - Introducción a OpenCV")
    st.subheader("Transformaciones básicas: traslación y rotación")

    st.write("""
    En este ejercicio aplicaremos **traslación** y **rotación** a una imagen usando OpenCV.  
    Sube una imagen y observa el resultado de las transformaciones.
    """)

    # Subida de imagen
    uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leer imagen en formato OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Mostrar imagen original
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="🖼️ Imagen original", use_container_width=True)

        # Obtener dimensiones
        num_rows, num_cols = img.shape[:2]

        # Crear matrices de transformación
        translation_matrix = np.float32([
            [1, 0, int(0.5 * num_cols)],
            [0, 1, int(0.5 * num_rows)]
        ])
        rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)

        # Aplicar transformaciones
        img_translation = cv2.warpAffine(img, translation_matrix, (2 * num_cols, 2 * num_rows))
        img_rotation = cv2.warpAffine(img_translation, rotation_matrix, (2 * num_cols, 2 * num_rows))

        # Mostrar resultados
        st.image(cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB), caption="Imagen trasladada", use_container_width=True)
        st.image(cv2.cvtColor(img_rotation, cv2.COLOR_BGR2RGB), caption="Imagen rotada", use_container_width=True)

        # Descargar resultado final
        _, buffer = cv2.imencode(".jpg", img_rotation)
        st.download_button(
            label="Descargar imagen rotada",
            data=buffer.tobytes(),
            file_name="imagen_rotada.jpg",
            mime="image/jpeg"
        )
    else:
        st.info("Sube una imagen para comenzar.")
