import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Football Historical Analytics", layout="wide")

LIGAS = {
    "Premier League (Inglaterra)": "E0",
    "LaLiga (España)": "SP1",
    "Serie A (Italia)": "I1",
    "Bundesliga (Alemania)": "D1",
    "Ligue 1 (Francia)": "F1",
}

TEMPORADAS = {
    "2026/2027": "2627",
    "2025/2026": "2526",
    "2024/2025": "2425",
    "2023/2024": "2324",
    "2022/2023": "2223",
}


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


@st.cache_data(ttl=3600)
def cargar_datos_historicos(codigo_liga: str, codigos_temp: list[str]) -> pd.DataFrame:
    frames = []
    for temp in codigos_temp:
        url = f"https://www.football-data.co.uk/mmz4281/{temp}/{codigo_liga}.csv"
        try:
            df = pd.read_csv(url, encoding="latin1")
            df["Temporada"] = f"20{temp[:2]}/20{temp[2:]}"
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    cols_base = [
        "Temporada", "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HC", "AC",
    ]
    cols_disponibles = [c for c in cols_base if c in data.columns]
    data = data[cols_disponibles].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    return data.sort_values("Date", ascending=False)


def calcular_metricas_poisson(df: pd.DataFrame, local: str, visita: str):
    media_goles_local = df["FTHG"].mean()
    media_goles_visita = df["FTAG"].mean()

    df_local = df[df["HomeTeam"] == local]
    df_visita = df[df["AwayTeam"] == visita]

    ataque_local = (df_local["FTHG"].mean() / media_goles_local) if len(df_local) else 1.0
    defensa_local = (df_local["FTAG"].mean() / media_goles_visita) if len(df_local) else 1.0

    ataque_visita = (df_visita["FTAG"].mean() / media_goles_visita) if len(df_visita) else 1.0
    defensa_visita = (df_visita["FTHG"].mean() / media_goles_local) if len(df_visita) else 1.0

    lambda_local = ataque_local * defensa_visita * media_goles_local
    lambda_visita = ataque_visita * defensa_local * media_goles_visita

    max_goles = 7
    matriz = np.outer(
        [poisson_pmf(i, lambda_local) for i in range(max_goles)],
        [poisson_pmf(j, lambda_visita) for j in range(max_goles)],
    )
    prob_1 = np.tril(matriz, -1).sum()
    prob_x = np.trace(matriz)
    prob_2 = np.triu(matriz, 1).sum()

    return lambda_local, lambda_visita, prob_1, prob_x, prob_2, matriz


st.title("Panel de Análisis Histórico de Fútbol")

with st.sidebar:
    st.header("Filtros Históricos")
    liga_nombre = st.selectbox("Competición", list(LIGAS.keys()))
    temps_sel = st.multiselect(
        "Temporadas a incluir",
        list(TEMPORADAS.keys()),
        default=["2026/2027", "2025/2026"],
    )

if not temps_sel:
    st.warning("Selecciona al menos una temporada en el panel lateral.")
    st.stop()

df_hist = cargar_datos_historicos(LIGAS[liga_nombre], [TEMPORADAS[t] for t in temps_sel])

if df_hist.empty:
    st.error("No se pudieron recuperar datos para la combinación seleccionada.")
    st.stop()

equipos = sorted(df_hist["HomeTeam"].unique())
col1, col2 = st.columns(2)
equipo_local = col1.selectbox("Equipo Local", equipos, index=0)
equipo_visita = col2.selectbox("Equipo Visitante", equipos, index=min(1, len(equipos) - 1))

lam_l, lam_v, p1, px_prob, p2, matriz_prob = calcular_metricas_poisson(df_hist, equipo_local, equipo_visita)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Goles Esperados Local (λ)", f"{lam_l:.2f}")
m2.metric("Goles Esperados Visita (λ)", f"{lam_v:.2f}")
m3.metric("Prob. Victoria Local", f"{p1*100:.1f}%")
m4.metric("Prob. Empate", f"{px_prob*100:.1f}%")
m5.metric("Prob. Victoria Visita", f"{p2*100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Matriz de Marcadores", "Historial Directo (H2H)", "Tendencia de Goles"])

with tab1:
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=matriz_prob * 100,
            x=[f"Visita {i}" for i in range(7)],
            y=[f"Local {i}" for i in range(7)],
            text=np.round(matriz_prob * 100, 1),
            texttemplate="%{text}%",
            colorscale="Blues",
        )
    )
    fig_heat.update_layout(title="Probabilidad de Marcador Exacto (%)", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    h2h = df_hist[
        ((df_hist["HomeTeam"] == equipo_local) & (df_hist["AwayTeam"] == equipo_visita))
        | ((df_hist["HomeTeam"] == equipo_visita) & (df_hist["AwayTeam"] == equipo_local))
    ]
    st.dataframe(h2h, use_container_width=True, hide_index=True)

with tab3:
    ultimos = df_hist[(df_hist["HomeTeam"] == equipo_local) | (df_hist["AwayTeam"] == equipo_local)].head(15).copy()
    ultimos["Goles_Equipo"] = np.where(ultimos["HomeTeam"] == equipo_local, ultimos["FTHG"], ultimos["FTAG"])
    fig_line = px.bar(
        ultimos.sort_values("Date"),
        x="Date",
        y="Goles_Equipo",
        color="Temporada",
        title=f"Últimos 15 partidos de {equipo_local} (Goles anotados)",
    )
    st.plotly_chart(fig_line, use_container_width=True)
