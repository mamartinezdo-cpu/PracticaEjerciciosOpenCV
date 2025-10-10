import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

# ===================== CLASES DETECTORAS =====================

class DenseDetector:
    def __init__(self, step_size=20, feature_scale=20, img_bound=20):
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound

    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints

class SIFTDetector:
    def __init__(self):
        # Se requiere tener instalado opencv-contrib-python
        self.detector = cv2.SIFT_create()

    def detect(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_image, None)

# ===================== INTERFAZ STREAMLIT =====================

def main():
    st.title("Detección de Características: Dense vs SIFT")
    st.markdown("""
    Este ejercicio compara dos técnicas de detección de puntos clave:
    - **Dense Feature Detector:** distribuye puntos de forma uniforme en toda la imagen.
    - **SIFT (Scale-Invariant Feature Transform):** detecta puntos clave según contrastes, bordes y escala.
    """)

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convertir a imagen temporal para OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        image = cv2.imread(tfile.name)
        if image is None:
            st.error("No se pudo leer la imagen. Asegúrate de subir un formato válido.")
            return

        # Mostrar imagen original
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📸 Imagen Original", use_container_width=True)

        # Parámetros ajustables
        st.sidebar.header("Parámetros del Detector Denso")
        step_size = st.sidebar.slider("Step Size", 5, 50, 20, 5)
        feature_scale = st.sidebar.slider("Feature Scale", 5, 50, 20, 5)
        img_bound = st.sidebar.slider("Image Bound", 0, 50, 5, 5)

        # Detectores
        dense_detector = DenseDetector(step_size, feature_scale, img_bound)
        sift_detector = SIFTDetector()

        # Detección Dense
        dense_kp = dense_detector.detect(image)
        dense_output = cv2.drawKeypoints(image, dense_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Detección SIFT
        sift_kp = sift_detector.detect(image)
        sift_output = cv2.drawKeypoints(image, sift_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Mostrar resultados
        st.subheader("Resultados de la Detección de Características")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(dense_output, cv2.COLOR_BGR2RGB), caption=f"Dense Detector ({len(dense_kp)} puntos)", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(sift_output, cv2.COLOR_BGR2RGB), caption=f"SIFT Detector ({len(sift_kp)} puntos)", use_container_width=True)

        # Información adicional
        st.success(f"Detección completada con {len(dense_kp)} puntos densos y {len(sift_kp)} puntos SIFT.")

if __name__ == "__main__":
    main()
