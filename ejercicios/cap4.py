import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.title("Detección de Rostro con Máscara (Hannibal)")
    st.write("Usa el clasificador Haar Cascade para detectar rostros y colocar una máscara sobre ellos.")

    # Cargar el clasificador de rostros
    face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml')
    if face_cascade.empty():
        st.error("No se pudo cargar el clasificador de rostros.")
        return

    # Cargar la imagen de máscara
    face_mask = cv2.imread('./images/mask_hannibal.png')
    if face_mask is None:
        st.error("No se encontró la imagen './images/mask_hannibal.png'.")
        return

    # Iniciar cámara
    run_camera = st.checkbox("Activar cámara")

    if run_camera:
        cap = cv2.VideoCapture(0)  # Usa 0 o 1 según tu dispositivo
        scaling_factor = 0.5
        stframe = st.empty()  # contenedor de video

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("No se pudo acceder a la cámara.")
                break

            frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)

            for (x, y, w, h) in face_rects:
                if h <= 0 or w <= 0:
                    continue

                # Ajustar el tamaño de la máscara
                h_new, w_new = int(1.4 * h), int(1.0 * w)
                y_new = max(0, y - int(0.1 * h_new))
                x_new = max(0, x)

                face_mask_small = cv2.resize(face_mask, (w_new, h_new), interpolation=cv2.INTER_AREA)

                # Convertir a escala de grises para crear máscara binaria
                gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_mask, 180, 255, cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)

                # ROI dentro de la imagen principal (asegurando límites)
                roi = frame[y_new:y_new + h_new, x_new:x_new + w_new]
                if roi.shape[0] <= 0 or roi.shape[1] <= 0:
                    continue

                try:
                    masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)
                    masked_frame = cv2.bitwise_and(roi, roi, mask=mask_inv)
                    combined = cv2.add(masked_face, masked_frame)
                    frame[y_new:y_new + h_new, x_new:x_new + w_new] = combined
                except cv2.error:
                    continue

            # Mostrar el frame en Streamlit
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Botón para detener
            if not run_camera:
                break

        cap.release()

