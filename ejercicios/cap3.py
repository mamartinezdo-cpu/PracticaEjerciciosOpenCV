import streamlit as st
import cv2
import numpy as np

def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)

    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)

    for _ in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)

    # 🩹 FIX: redimensionar la máscara al tamaño de la imagen de salida
    mask_resized = cv2.resize(mask, (img_output.shape[1], img_output.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Combinar bordes con la imagen suavizada
    cartoon = cv2.bitwise_and(img_output, img_output, mask=mask_resized)
    return cartoon


def main():
    st.title("Capítulo 3 - Efecto Caricatura (Cartoonizer)")
    st.write("""
    Este ejercicio transforma una imagen en un **efecto de caricatura o boceto**, usando filtros bilaterales, 
    detección de bordes y umbralización.
    """)

    # Subir imagen
    uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Mostrar imagen original
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        # Selección de modo
        mode = st.radio("🎭 Modo de caricatura", ("Caricatura con color", "Boceto sin color"))

        # Parámetros ajustables
        ksize = st.slider("Tamaño del filtro (ksize)", min_value=3, max_value=15, step=2, value=5)

        # Procesar imagen
        if st.button("🪄 Aplicar efecto"):
            sketch_mode = (mode == "Boceto sin color")
            cartoonized = cartoonize_image(img, ksize=ksize, sketch_mode=sketch_mode)

            st.image(cv2.cvtColor(cartoonized, cv2.COLOR_BGR2RGB),
                     caption=f"Resultado - {mode}",
                     use_container_width=True)

            # Botón de descarga
            _, buffer = cv2.imencode(".jpg", cartoonized)
            st.download_button(
                label="Descargar imagen caricaturizada",
                data=buffer.tobytes(),
                file_name=f"cartoonized_{'sketch' if sketch_mode else 'color'}.jpg",
                mime="image/jpeg"
            )

            # Explicación
            with st.expander("Explicación del proceso"):
                st.markdown("""
                - Se convierte la imagen a **escala de grises** y se aplica un **filtro mediano** para reducir ruido.  
                - Se detectan los bordes con **Laplacian** y se umbraliza para obtener un **boceto**.  
                - Se aplica **filtro bilateral repetido** para suavizar colores sin perder bordes.  
                - Finalmente, se combinan los bordes con la imagen suavizada para lograr el efecto de caricatura.
                """)

    else:
        st.info("Sube una imagen para comenzar.")

