import streamlit as st
import joblib
import numpy as np
import pandas as pd

modelo = joblib.load("modelo.joblib")
features = joblib.load("features.joblib")

st.set_page_config(
    page_title="Passos Mágicos — Risco Educacional",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Passos Mágicos")
st.subheader("Modelo Preditivo de Risco Educacional")
st.markdown(
    "Preencha os indicadores do aluno para obter a probabilidade de risco educacional."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    IAN = st.selectbox("IAN — Adequação ao Nível", options=[2.5, 5.0, 10.0])
    IDA = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 6.0, 0.1)
    IEG = st.slider("IEG — Engajamento", 0.0, 10.0, 6.0, 0.1)
    IAA = st.slider("IAA — Autoavaliação", 0.0, 10.0, 6.0, 0.1)
    IPS = st.slider("IPS — Psicossocial", 0.0, 10.0, 6.0, 0.1)

with col2:
    IPP = st.slider("IPP — Psicopedagógico", 0.0, 10.0, 6.0, 0.1)
    IPV = st.slider("IPV — Ponto de Virada", 0.0, 10.0, 6.0, 0.1)
    ANO = st.selectbox("Ano de referência", options=[2022, 2023, 2024])
    PEDRA = st.selectbox(
        "Pedra (classificação atual)",
        options=["Quartzo", "Ágata", "Ametista", "Topázio"]
    )

st.divider()

pedra_map = {"Quartzo": 0, "Ágata": 1, "Ametista": 2, "Topázio": 3}
PEDRA_COD = pedra_map[PEDRA]

DELTA_IAA_IDA = IAA - IDA
IEG_x_IPV = IEG * IPV
MEDIA_PSICO = (IPS + IPP) / 2

entrada = pd.DataFrame([{
    "IAN": IAN,
    "IDA": IDA,
    "IEG": IEG,
    "IAA": IAA,
    "IPS": IPS,
    "IPV": IPV,
    "DELTA_IAA_IDA": DELTA_IAA_IDA,
    "IEG_x_IPV": IEG_x_IPV,
    "MEDIA_PSICO": MEDIA_PSICO,
    "PEDRA_COD": PEDRA_COD,
    "ANO": ANO
}])[features]

if st.button("Calcular risco", use_container_width=True):
    prob = modelo.predict_proba(entrada)[0][1]

    risco_regra = (IAN == 2.5) or (IDA < 5.0)

    st.divider()

    if risco_regra:
        criterio = []
        if IAN == 2.5:
            criterio.append("IAN = 2.5 (defasagem severa)")
        if IDA < 5.0:
            criterio.append(f"IDA = {IDA:.1f} (abaixo do limiar crítico)")
        st.error(
            f"⚠️ Aluno em risco educacional — probabilidade predita: {prob*100:.1f}%. "
            f"Critério crítico acionado: {' | '.join(criterio)}."
        )
    else:
        if prob >= 0.5:
            st.error(f"⚠️ Aluno em risco educacional — probabilidade predita: {prob*100:.1f}%.")
        else:
            st.success(f"✅ Aluno sem risco educacional — probabilidade de risco: {prob*100:.1f}%.")

    st.markdown("**Detalhamento dos indicadores informados:**")
    st.dataframe(entrada.T.rename(columns={0: "Valor"}), use_container_width=True)
