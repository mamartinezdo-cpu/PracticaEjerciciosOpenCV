import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Detección de Puntos Clave con ORB")
    st.write("Este ejercicio utiliza el detector ORB (Oriented FAST and Rotated BRIEF) para encontrar características distintivas en una imagen.")

    uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leer imagen en formato OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if input_image is None:
            st.error("No se pudo leer la imagen. Verifica el formato.")
            return

        # Convertir a escala de grises
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Crear el detector ORB
        orb = cv2.ORB_create()

        # Detectar puntos clave
        keypoints = orb.detect(gray_image, None)

        # Calcular descriptores
        keypoints, descriptors = orb.compute(gray_image, keypoints)

        # Dibujar los puntos clave sobre la imagen original
        output_image = cv2.drawKeypoints(input_image, keypoints, None, color=(0, 255, 0), flags=0)

        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Puntos clave detectados con ORB", use_container_width=True)

        # Mostrar detalles
        st.write(f"**Puntos detectados:** {len(keypoints)}")
        st.write("El detector ORB identifica esquinas y regiones con alta variación de intensidad, útiles en visión por computadora para reconocimiento y seguimiento.")

    else:
        st.info("Sube una imagen para comenzar el análisis con ORB.")

