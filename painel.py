import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv(
    "AC.csv",
    sep=";",
    encoding="latin1"
)

st.title("Previsão do Resultado das Eleições 2026 Aprendizado de maquina com base em dados de 2022")

cargo = st.selectbox(
    "Selecione o Cargo",
    sorted(df["DS_CARGO"].unique())
)

resultado = (
    df[df["DS_CARGO"] == cargo]
    .groupby(
        ["NR_VOTAVEL", "NM_VOTAVEL"]
    )["QT_VOTOS"]
    .sum()
    .reset_index()
    .sort_values(
        "QT_VOTOS",
        ascending=False
    )
)

st.dataframe(
    resultado,
    use_container_width=True,
    hide_index=True
)

fig = px.bar(
    resultado.head(10),
    x="NM_VOTAVEL",
    y="QT_VOTOS",
    color="QT_VOTOS",
    text="QT_VOTOS",
    title=f"Top 10 - {cargo}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)