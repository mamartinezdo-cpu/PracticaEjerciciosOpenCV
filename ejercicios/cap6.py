import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# ===================== FUNCIONES AUXILIARES =====================

def overlay_vertical_seam(img, seam):
    img_seam_overlay = np.copy(img)
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0)
    return img_seam_overlay

def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]
    seam = np.zeros(rows)
    dist_to = np.zeros((rows, cols)) + float('inf')
    dist_to[0, :] = energy[0, :]
    edge_to = np.zeros((rows, cols))

    for row in range(rows - 1):
        for col in range(cols):
            for offset in [-1, 0, 1]:
                c = col + offset
                if 0 <= c < cols:
                    new_dist = dist_to[row, col] + energy[row + 1, c]
                    if new_dist < dist_to[row + 1, c]:
                        dist_to[row + 1, c] = new_dist
                        edge_to[row + 1, c] = offset

    seam[-1] = np.argmin(dist_to[-1])
    for i in range(rows - 2, -1, -1):
        seam[i] = seam[i + 1] + edge_to[i + 1, int(seam[i + 1])]
    return seam

def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]
    output = np.zeros((rows, cols - 1, 3), dtype=np.uint8)
    for row in range(rows):
        col = int(seam[row])
        output[row, :, :] = np.delete(img[row, :, :], col, axis=0)
    return output

def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    output = np.zeros((rows, cols + 1, 3), dtype=np.uint8)
    for row in range(rows):
        col = int(seam[row])
        for i in range(3):
            v1 = img[row, max(col - 1, 0), i]
            v2 = img[row, min(col + 1, cols - 1), i]
            output[row, :col] = img[row, :col]
            output[row, col] = (int(v1) + int(v2)) // 2
            output[row, col + 1:] = img[row, col:]
    return output

# ===================== INTERFAZ STREAMLIT =====================

def main():
    st.title("Eliminación y Adición de Costuras (Seam Carving)")
    st.markdown("Sube una imagen y elige cuántas costuras deseas eliminar o añadir verticalmente.")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    num_seams = st.slider("Número de costuras a procesar", 1, 50, 10)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        if st.button("Procesar Imagen"):
            img = np.copy(img_input)
            img_output = np.copy(img_input)
            img_overlay_seam = np.copy(img_input)
            energy = compute_energy_matrix(img)

            progress_bar = st.progress(0)

            for i in range(num_seams):
                seam = find_vertical_seam(img, energy)
                img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
                img = remove_vertical_seam(img, seam)
                img_output = add_vertical_seam(img_output, seam, i)
                energy = compute_energy_matrix(img)
                progress_bar.progress((i + 1) / num_seams)

            st.success(f"Procesamiento completado ({num_seams} costuras).")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(img_overlay_seam, cv2.COLOR_BGR2RGB), caption="Costuras detectadas", use_container_width=True)
            with col3:
                st.image(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB), caption="Imagen modificada", use_container_width=True)

if __name__ == "__main__":
    main()
