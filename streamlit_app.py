import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai

st.set_page_config(page_title="Football Historical Analytics", layout="wide")

MODELO_GEMINI = "gemini-3.8-flash"

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
    prob_btts = matriz[1:, 1:].sum()
    prob_over25 = sum(
        matriz[i, j] for i in range(max_goles) for j in range(max_goles) if (i + j) >= 3
    )

    return lambda_local, lambda_visita, prob_1, prob_x, prob_2, prob_btts, prob_over25, matriz


def construir_contexto_ia(
    liga: str,
    temps: list[str],
    local: str,
    visita: str,
    lam_l: float,
    lam_v: float,
    p1: float,
    px_prob: float,
    p2: float,
    p_btts: float,
    p_over25: float,
    matriz: np.ndarray,
    h2h_df: pd.DataFrame,
    ult_local: pd.DataFrame,
    ult_visita: pd.DataFrame,
) -> str:
    indices_top = np.dstack(np.unravel_index(np.argsort(matriz.ravel())[::-1][:3], matriz.shape))[0]
    top_marcadores = ", ".join(
        [f"{i}-{j} ({matriz[i, j]*100:.1f}%)" for i, j in indices_top]
    )

    cuota_1 = f"{1/p1:.2f}" if p1 > 0 else "N/A"
    cuota_x = f"{1/px_prob:.2f}" if px_prob > 0 else "N/A"
    cuota_2 = f"{1/p2:.2f}" if p2 > 0 else "N/A"

    resumen_h2h = (
        h2h_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(5).to_string(index=False)
        if not h2h_df.empty
        else "Sin enfrentamientos directos en las temporadas seleccionadas."
    )
    resumen_l = ult_local[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(5).to_string(index=False)
    resumen_v = ult_visita[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(5).to_string(index=False)

    return f"""
Actúa como un analista cuantitativo de fútbol experto en modelos de Poisson, mercados asiáticos y valor esperado (EV).
Responde siempre en español, de forma directa, objetiva y respaldada estrictamente en los datos del partido:

- Competición: {liga} | Temporadas: {', '.join(temps)}
- Partido: {local} (Local) vs {visita} (Visitante)
- Goles Esperados (Lambda): {local} = {lam_l:.2f} | {visita} = {lam_v:.2f}
- Probabilidades Modelo 1X2: Local {p1*100:.1f}% (Cuota justa: {cuota_1}) | Empate {px_prob*100:.1f}% (Cuota justa: {cuota_x}) | Visita {p2*100:.1f}% (Cuota justa: {cuota_2})
- Mercados de Goles: Over 2.5 = {p_over25*100:.1f}% | Ambos Marcan (BTTS) = {p_btts*100:.1f}%
- Top 3 Marcadores Exactos: {top_marcadores}

Últimos 5 partidos de {local}:
{resumen_l}

Últimos 5 partidos de {visita}:
{resumen_v}

Historial Directo (H2H):
{resumen_h2h}
"""


st.title("Panel de Análisis Histórico de Fútbol")

with st.sidebar:
    st.header("Filtros Históricos")
    liga_nombre = st.selectbox("Competición", list(LIGAS.keys()))
    temps_sel = st.multiselect(
        "Temporadas a incluir",
        list(TEMPORADAS.keys()),
        default=["2026/2027", "2025/2026"],
    )
    st.divider()
    st.header("Configuración IA")
    api_key_secret = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key_input = st.text_input(
        "Gemini API Key",
        value=api_key_secret,
        type="password",
        help="Obtén tu clave gratuita en Google AI Studio o guárdala en Streamlit Secrets como GEMINI_API_KEY.",
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

lam_l, lam_v, p1, px_prob, p2, p_btts, p_over25, matriz_prob = calcular_metricas_poisson(
    df_hist, equipo_local, equipo_visita
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Goles Esperados Local (λ)", f"{lam_l:.2f}")
m2.metric("Goles Esperados Visita (λ)", f"{lam_v:.2f}")
m3.metric("Prob. Victoria Local", f"{p1*100:.1f}%")
m4.metric("Prob. Empate", f"{px_prob*100:.1f}%")
m5.metric("Prob. Victoria Visita", f"{p2*100:.1f}%")

h2h = df_hist[
    ((df_hist["HomeTeam"] == equipo_local) & (df_hist["AwayTeam"] == equipo_visita))
    | ((df_hist["HomeTeam"] == equipo_visita) & (df_hist["AwayTeam"] == equipo_local))
]
ultimos_local = df_hist[(df_hist["HomeTeam"] == equipo_local) | (df_hist["AwayTeam"] == equipo_local)].head(15).copy()
ultimos_visita = df_hist[(df_hist["HomeTeam"] == equipo_visita) | (df_hist["AwayTeam"] == equipo_visita)].head(15).copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Matriz de Marcadores",
    "Historial Directo (H2H)",
    "Tendencia de Goles",
    "Asistente Analítico IA",
])

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
    st.dataframe(h2h, use_container_width=True, hide_index=True)

with tab3:
    ultimos_local["Goles_Equipo"] = np.where(
        ultimos_local["HomeTeam"] == equipo_local, ultimos_local["FTHG"], ultimos_local["FTAG"]
    )
    fig_line = px.bar(
        ultimos_local.sort_values("Date"),
        x="Date",
        y="Goles_Equipo",
        color="Temporada",
        title=f"Últimos 15 partidos de {equipo_local} (Goles anotados)",
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab4:
    st.subheader(f"Análisis Inteligente: {equipo_local} vs {equipo_visita}")

    partido_id = f"{liga_nombre}_{equipo_local}_{equipo_visita}_{'_'.join(temps_sel)}"
    if st.session_state.get("partido_actual") != partido_id:
        st.session_state["partido_actual"] = partido_id
        st.session_state["mensajes_ia"] = []

    contexto_sistema = construir_contexto_ia(
        liga_nombre, temps_sel, equipo_local, equipo_visita,
        lam_l, lam_v, p1, px_prob, p2, p_btts, p_over25,
        matriz_prob, h2h, ultimos_local, ultimos_visita,
    )

    if not api_key_input:
        st.info("Introduce tu Gemini API Key en el menú lateral para activar el chat analítico en vivo.")
        with st.expander("Ver datos estructurados listos para copiar"):
            st.code(contexto_sistema, language="text")
    else:
        client = genai.Client(api_key=api_key_input)

        if st.button("Generar diagnóstico rápido del partido", type="primary"):
            prompt_diagnostico = (
                "Realiza un diagnóstico breve del partido en 3 puntos: "
                "1) Lectura de fuerzas ofensivas/defensivas y cuota justa 1X2, "
                "2) Tendencia en mercados de goles (Over/Under 2.5 y BTTS), y "
                "3) Principal riesgo estadístico según la racha reciente."
            )
            with st.spinner("Analizando métricas de Poisson y rachas..."):
                try:
                    resp = client.models.generate_content(
                        model=MODELO_GEMINI,
                        contents=f"{contexto_sistema}\n\nSolicitud del usuario: {prompt_diagnostico}",
                    )
                    st.session_state["mensajes_ia"].append({"role": "assistant", "content": resp.text})
                except Exception as e:
                    st.error(f"Error al consultar Gemini: {e}")

        for msg in st.session_state["mensajes_ia"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pregunta = st.chat_input("Ej: ¿Hay valor si la casa paga 2.15 por el local o mejor entro al Over 2.5?")
        if pregunta:
            st.session_state["mensajes_ia"].append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)

            with st.chat_message("assistant"):
                with st.spinner("Calculando respuesta..."):
                    try:
                        historial_chat = "\n".join(
                            [f"{m['role'].upper()}: {m['content']}" for m in st.session_state["mensajes_ia"]]
                        )
                        resp = client.models.generate_content(
                            model=MODELO_GEMINI,
                            contents=f"{contexto_sistema}\n\nConversación previa:\n{historial_chat}",
                        )
                        st.markdown(resp.text)
                        st.session_state["mensajes_ia"].append({"role": "assistant", "content": resp.text})
                    except Exception as e:
                        st.error(f"Error al consultar Gemini: {e}")
