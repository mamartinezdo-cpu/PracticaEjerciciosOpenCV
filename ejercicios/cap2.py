import streamlit as st
import cv2
import numpy as np

def main():
    st.title("Capítulo 2 - Operaciones básicas con imágenes")
    st.subheader("Aplicación de filtro de desenfoque por movimiento (Motion Blur)")

    st.write("""
    En este ejercicio aplicaremos un **filtro de desenfoque por movimiento** a una imagen.  
    Usaremos un *kernel personalizado* para simular el efecto de movimiento.
    """)

    # Subir imagen
    uploaded_file = st.file_uploader("Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leer la imagen con OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Mostrar imagen original
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", use_container_width=True)

        # Selector de tamaño del kernel
        size = st.slider("Tamaño del desenfoque (kernel size)", min_value=3, max_value=50, value=15, step=2)

        # Generar kernel de desenfoque por movimiento
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # Aplicar el filtro
        output = cv2.filter2D(img, -1, kernel_motion_blur)

        # Mostrar imagen resultante
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption=f"Imagen con Motion Blur (kernel={size})", use_container_width=True)

        # Opción para descargar
        _, buffer = cv2.imencode(".jpg", output)
        st.download_button(
            label="Descargar imagen con desenfoque",
            data=buffer.tobytes(),
            file_name=f"motion_blur_{size}.jpg",
            mime="image/jpeg"
        )

        # Explicación breve
        with st.expander("Explicación del proceso"):
            st.markdown("""
            - Se crea una **matriz (kernel)** de tamaño `size x size` con ceros.  
            - La fila central se llena de unos para simular el movimiento lineal.  
            - El kernel se normaliza dividiendo entre el tamaño (`size`).  
            - Luego, `cv2.filter2D()` aplica este kernel sobre la imagen, generando el efecto de **desenfoque direccional**.
            """)
    else:
        st.info("Sube una imagen para comenzar.")
