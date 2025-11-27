import streamlit as st
import pandas as pd
import pickle
import base64

# Para cargar el logo
with open("datom.png", "rb") as logo:
    logo_datom = base64.b64encode(logo.read()).decode()

st.markdown(
    f"""
    <style>
    .logo-container {{
        width: 100%;
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
    }}

    .logo-img {{
        width: 180px;
    }}

    /* Ajuste responsivo para celulares */
    @media (max-width: 600px) {{
        .logo-img {{
            width: 120px;
        }}
    }}

    /* Ajuste para pantallas muy pequeñas */
    @media (max-width: 400px) {{
        .logo-img {{
            width: 90px;
        }}
    }}
    </style>

    <div class="logo-container">
        <img src="data:image/png;base64,{logo_datom}" class="logo-img">
    </div>
    """,
    unsafe_allow_html=True
)

alcaldias = ["TLAHUAC", "CUAUHTEMOC", "IZTAPALAPA", "GUSTAVO A. MADERO", "ALVARO OBREGON", "BENITO JUAREZ",
             "MILPA ALTA", "COYOACAN", "MAGDALENA CONTRERAS", "MIGUEL HIDALGO", "TLALPAN", "IZTACALCO",
             "VENUSTIANO CARRANZA", "AZCAPOTZALCO", "XOCHIMILCO", "AZCAPOTZALCO"]

vialidades = ["EJE VIAL", "VIA PRIMARIA", "VIA SECUNDARIA", "VAC ANULAR", "VAC RADIAL", "VAC VIADUCTO", "ACCESO CARRETERO"]

intersecciones = ['CRUZ', 'T', 'RECTA', 'RAMAS MULTIPLES', 'CURVA', 'Y', 'DESNIVEL', 'GLORIETA', 'GAZA', 'LATERALES', 'NO ESPECIFICADO']

# Esto carga el modelo descargado desde Colab
with open("modelo.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Title de la página
st.set_page_config(page_title="Predicción de Accidentes CDMX", layout="centered")
st.title("Predicción de Accidentes CDMX")

st.markdown("Completa los datos del accidente para predecir la severidad estimada.")

# Inputs del usuario
tipo_evento = st.selectbox("Tipo de evento", ["CHOQUE", "ATROPELLO", "DERRAPE", "VOLCADURA", "CAIDA CICLISTA", "CAIDA PASAJERO"])
alcaldia = st.selectbox("Alcaldía", alcaldias)
hora = st.number_input("Hora del evento (0-23)", min_value=0, max_value=23, value=12)
dia_semana = st.selectbox("Día de la semana", ["LUNES","MARTES","MIERCOLES","JUEVES","VIERNES","SABADO","DOMINGO"])
clasificacion_vialidad = st.selectbox("Clasificación de la vialidad", vialidades)
tipo_interseccion = st.selectbox("Tipo de intersección", intersecciones)
interseccion_semaforizada = st.selectbox("Intersección semaforizada", ["SI", "NO"])
tiene_moto = st.checkbox("Hay moto")
tiene_bici = st.checkbox("Hay bicicleta")
tiene_peaton = st.checkbox("Hay peatón")
solo_autos = st.checkbox("Solo autos")
edad_promedio = st.number_input("Edad promedio", min_value=0, max_value=100, value=30)

if st.button("Predecir"):
    # Crear df desde inputs
    if 6 <= hora < 12:
        turno = "MANANA"
    elif 12 <= hora < 18:
        turno = "TARDE"
    else:
        turno = "NOCHE"
    
    df_test = pd.DataFrame([{
        "tipo_evento": tipo_evento,
        "alcaldia": alcaldia,
        "hora": hora,
        "dia_semana": dia_semana,
        "turno": turno,
        "clasificacion_de_la_vialidad": clasificacion_vialidad,
        "tipo_de_interseccion": tipo_interseccion,
        "interseccion_semaforizada": interseccion_semaforizada,
        "tiene_moto": int(tiene_moto),
        "tiene_bici": int(tiene_bici),
        "tiene_peaton": int(tiene_peaton),
        "solo_autos": int(solo_autos),
        "edad_promedio": edad_promedio
    }])

    # Features derivadas
    df_test["hora_pico"] = df_test["hora"].apply(lambda h: 1 if h in [7,8,9,10,17,18,19,20] else 0)
    df_test["hora_noche"] = df_test["hora"].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)
    df_test["interseccion_riesgo_medio"] = df_test["tipo_de_interseccion"].isin(["GLORIETA", "Y"]).astype(int)
    df_test["interseccion_riesgo_alto"] = df_test["tipo_de_interseccion"].isin(["CURVA", "DESNIVEL", "RAMAS MULTIPLES"]).astype(int)

    # Predicción
    pred = pipeline.predict(df_test)
    #prob = pipeline.predict_proba(df_test)[:,1]  # probabilidad de clase positiva

    # Mostrar resultados
    st.success(f"Predicción de severidad: {pred[0]}")
    #st.info(f"Probabilidad estimada: {prob[0]:.2f}")

st.markdown("""
    <div style="
        background-color: #1F2126;
        padding: 16px;
        border-radius: 10px;
    ">
    <h3 style="margin-top: 0;">Severidad</h3>
    <ul>
    <li><b>0 – Muy baja:</b> No hay personas lesionadas, no participan moto/bici/peatón y solo hay un afectado. La intersección es simple (CRUZ, T o RECTA).</li>
    <li><b>1 – Baja:</b> Hay solo una persona lesionada y ninguna persona fallecida.</li>
    <li><b>2 – Media:</b> Hay 2 lesionados o 2 afectados; o bien participa una moto o una bici, o la intersección es de riesgo medio (GLORIETA, Y).</li>
    <li><b>3 – Alta:</b> Hay 3 o más lesionados/afectados; o participa un peatón con lesionados; o una moto con ≥2 lesionados; o la intersección es de riesgo alto (CURVA, DESNIVEL, RAMAS MULTIPLES).</li>
    <li><b>4 – Muy alta:</b> Ocurre al menos una muerte.</li>
    </ul>
    </div>
""", unsafe_allow_html=True)


