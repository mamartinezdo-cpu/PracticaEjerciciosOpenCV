import streamlit as st
import cv2
import numpy as np
import tempfile
import time

# ===================== FUNCIONES AUXILIARES =====================

def get_frame(cap, scaling_factor=0.5):
    """Captura y redimensiona un fotograma del video."""
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

def process_video(video_source, use_camera=False):
    """Aplica sustracción de fondo MOG2 sobre un video o cámara."""
    if use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_source)

    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
    history = 100

    frame_placeholder = st.empty()
    mask_placeholder = st.empty()

    while True:
        frame = get_frame(cap, 0.7)
        if frame is None:
            break

        mask = bgSubtractor.apply(frame, learningRate=1.0 / history)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = cv2.bitwise_and(frame, mask_rgb)

        # Mostrar resultados
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="📷 Frame Original", use_container_width=True)
        mask_placeholder.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), caption="🎯 Objetos en Movimiento (MOG2)", use_container_width=True)

        time.sleep(0.03)

        # Detener al presionar el botón
        if st.session_state.get("stop", False):
            break

    cap.release()

# ===================== INTERFAZ STREAMLIT =====================

def main():
    st.title("🎥 Detección de Movimiento con Sustracción de Fondo (MOG2)")
    st.markdown("""
    Este módulo aplica el algoritmo **MOG2** para detectar objetos en movimiento en un video o transmisión de cámara.
    """)

    option = st.radio("Selecciona la fuente de video:", ("Subir video", "Usar cámara web"))

    if option == "Subir video":
        uploaded_file = st.file_uploader("Sube un archivo de video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            if st.button("Iniciar detección"):
                st.session_state["stop"] = False
                process_video(tfile.name, use_camera=False)

            if st.button("Detener"):
                st.session_state["stop"] = True

    elif option == "Usar cámara web":
        st.warning("Al presionar iniciar, se activará tu cámara.")
        if st.button("Iniciar cámara"):
            st.session_state["stop"] = False
            process_video(None, use_camera=True)

        if st.button("Detener"):
            st.session_state["stop"] = True

if __name__ == "__main__":
    main()
