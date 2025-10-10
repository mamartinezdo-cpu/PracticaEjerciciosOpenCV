import streamlit as st
import sys
import os 

# Añadir ruta actual ANTES de importar los módulos
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importar los módulos de ejercicios
from ejercicios import cap1, cap2, cap3, cap4, cap5, cap6, cap7, cap8, cap9

# Configuración general de la página
st.set_page_config(
    page_title="Proyecto OpenCV - Ejercicios",
    page_icon="👁️",
    layout="centered"
)

# --- Menú lateral ---
st.sidebar.title("📚 Menú de Ejercicios")
st.sidebar.markdown("Selecciona un capítulo del libro de OpenCV:")

opciones = [
    "Inicio",
    "Capítulo 1 - Introducción a OpenCV",
    "Capítulo 2 - Operaciones básicas con imágenes",
    "Capítulo 3 - Filtros y suavizado",
    "Capítulo 4 - Detección de bordes",
    "Capítulo 5 - Transformaciones geométricas",
    "Capítulo 6 - Espacios de color y umbralización",
    "Capítulo 7 - Detección de características",
    "Capítulo 8 - Segmentación de imágenes",
    "Capítulo 9 - Morfología digital",
 #   "Capítulo 10 - Video y movimiento",
 #   "Capítulo 11 - Proyecto final con OpenCV"
]

seleccion = st.sidebar.selectbox("Ejercicio:", opciones)

# --- Contenido dinámico ---
if seleccion == "Inicio":
    st.title("👁️ Proyecto de Ejercicios con OpenCV")
    st.markdown("---")
    st.subheader("Visualización y procesamiento de imágenes con OpenCV")

    st.write("""
    ¡Bienvenido!  
    Este proyecto contiene **11 ejercicios prácticos** basados en los capítulos del libro de OpenCV.  
    Aquí podrás **ver, ejecutar y analizar** ejemplos interactivos de visión por computadora.
    """)

    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg",
        width=200,
        caption="OpenCV - Open Source Computer Vision Library"
    )

elif seleccion == "Capítulo 1 - Introducción a OpenCV":
    cap1.main()

elif seleccion == "Capítulo 2 - Operaciones básicas con imágenes":
    cap2.main()

elif seleccion == "Capítulo 3 - Filtros y suavizado":
    cap3.main()

elif seleccion == "Capítulo 4 - Detección de bordes":
    cap4.main()

elif seleccion == "Capítulo 5 - Transformaciones geométricas":
    cap5.main()

elif seleccion == "Capítulo 6 - Espacios de color y umbralización":
    cap6.main()

elif seleccion == "Capítulo 7 - Detección de características":
    cap7.main()

elif seleccion == "Capítulo 8 - Segmentación de imágenes":
    cap8.main()

elif seleccion == "Capítulo 9 - Morfología digital":
    cap9.main()

#elif seleccion == "Capítulo 10 - Video y movimiento":
#    cap10.main()

#elif seleccion == "Capítulo 11 - Proyecto final con OpenCV":
#    cap11.main()

# --- Pie de página ---
st.markdown("---")
st.caption("Desarrollado por Mario Martínez — Proyecto OpenCV con Streamlit")
