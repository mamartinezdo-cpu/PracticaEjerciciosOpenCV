import streamlit as st
import cv2
import numpy as np

# ===================== FUNCIONES AUXILIARES =====================

def get_all_contours(img):
    """Convierte la imagen a escala de grises, aplica umbral y obtiene los contornos."""
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)

    # Compatibilidad con versiones de OpenCV
    contours_info = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    return contours

def draw_convexity_defects(img):
    """Dibuja contornos y sus defectos de convexidad."""
    output = np.copy(img)
    contours = get_all_contours(img)

    for contour in contours:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            continue

        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            continue

        # Dibujar cada defecto de convexidad
        for i in range(defects.shape[0]):
            start_defect, end_defect, far_defect, _ = defects[i, 0]
            start = tuple(contour[start_defect][0])
            end = tuple(contour[end_defect][0])
            far = tuple(contour[far_defect][0])

            # Contorno negro
            cv2.drawContours(output, [contour], -1, color=(0, 0, 0), thickness=2)
            # Puntos rojos en defectos
            cv2.circle(output, far, 5, [255, 0, 0], -1)
            # Línea verde entre puntos
            cv2.line(output, start, end, (0, 255, 0), 2)

    return output

# ===================== INTERFAZ STREAMLIT =====================

def main():
    st.title("Detección de Defectos de Convexidad")
    st.markdown("Este módulo detecta **defectos de convexidad** en los contornos de una imagen.")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        if st.button("Detectar Defectos de Convexidad"):
            result = draw_convexity_defects(img)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Defectos detectados", use_container_width=True)
            st.success("Detección completada correctamente.")

if __name__ == "__main__":
    main()
