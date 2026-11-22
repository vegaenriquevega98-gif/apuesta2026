import math
import time
import difflib
from datetime import datetime, timezone, timedelta
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai

st.set_page_config(page_title="TuApuestas - Análisis Histórico & IA", layout="wide")

MODELO_GEMINI_DEFAULT = "gemini-3.8-flash"
ZONA_HORARIA_LOCAL = timezone(timedelta(hours=-4))  # Hora Venezuela (UTC-4)

LIGAS = {
    "Premier League (Inglaterra)": {"csv": "E0", "espn": "eng.1"},
    "LaLiga (España)": {"csv": "SP1", "espn": "esp.1"},
    "Serie A (Italia)": {"csv": "I1", "espn": "ita.1"},
    "Bundesliga (Alemania)": {"csv": "D1", "espn": "ger.1"},
    "Ligue 1 (Francia)": {"csv": "F1", "espn": "fra.1"},
}

TEMPORADAS = {
    "2026/2027": "2627",
    "2025/2026": "2526",
    "2024/2025": "2425",
    "2023/2024": "2324",
    "2022/2023": "2223",
}

ALIASES_EQUIPOS = {
    "manchester city": "Man City",
    "manchester united": "Man United",
    "nottingham forest": "Nott'm Forest",
    "newcastle united": "Newcastle",
    "wolverhampton wanderers": "Wolves",
    "tottenham hotspur": "Tottenham",
    "west ham united": "West Ham",
    "brighton & hove albion": "Brighton",
    "leicester city": "Leicester",
    "ipswich town": "Ipswich",
    "atlético madrid": "Ath Madrid",
    "atletico madrid": "Ath Madrid",
    "athletic club": "Ath Bilbao",
    "athletic bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad",
    "real betis": "Betis",
    "rayo vallecano": "Vallecano",
    "celta vigo": "Celta",
    "deportivo alavés": "Alaves",
    "deportivo alaves": "Alaves",
    "rcd espanyol": "Espanol",
    "internazionale": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "as roma": "Roma",
    "hellas verona": "Verona",
    "bayern munich": "Bayern Munich",
    "bayer leverkusen": "Leverkusen",
    "borussia dortmund": "Dortmund",
    "borussia mönchengladbach": "M'gladbach",
    "borussia monchengladbach": "M'gladbach",
    "eintracht frankfurt": "Ein Frankfurt",
    "vfb stuttgart": "Stuttgart",
    "sc freiburg": "Freiburg",
    "paris saint-germain": "Paris SG",
    "as monaco": "Monaco",
    "olympique marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "stade rennais": "Rennes",
    "as saint-étienne": "St Etienne",
    "as saint-etienne": "St Etienne",
}


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def consultar_gemini(client: genai.Client, modelo: str, prompt: str, reintentos: int = 2) -> str:
    ultimo_error = None
    for intento in range(reintentos):
        try:
            resp = client.models.generate_content(
                model=modelo.strip(),
                contents=prompt,
            )
            return resp.text
        except Exception as e:
            ultimo_error = e
            texto_error = str(e)
            es_temporal = ("503" in texto_error) or ("UNAVAILABLE" in texto_error) or ("429" in texto_error)
            if es_temporal and intento < reintentos - 1:
                time.sleep(1.5 * (intento + 1))
                continue
            raise ultimo_error


def generar_dictamen_local(
    local: str,
    visita: str,
    m: dict,
    ult_local: pd.DataFrame,
    ult_visita: pd.DataFrame,
) -> str:
    """Motor cuantitativo de respaldo cuando la API externa devuelve 503."""
    def cuota(p: float) -> str:
        return f"{1/p:.2f}" if p > 0.01 else "N/A"

    matriz = m["matriz"]
    idx_top = np.unravel_index(np.argmax(matriz), matriz.shape)
    marcador_top = f"{idx_top[0]}-{idx_top[1]} ({matriz[idx_top]*100:.1f}%)"

    candidatos_principales = [
        (f"Victoria Local ({local})", m["p1"], "resultado"),
        (f"Victoria Visitante ({visita})", m["p2"], "resultado"),
        ("Más de 2.5 Goles (Over 2.5)", m["over25"], "goles"),
        ("Menos de 2.5 Goles (Under 2.5)", m["under25"], "goles"),
        ("Ambos Equipos Marcan (BTTS Sí)", m["btts"], "goles"),
    ]
    # Si ningún 1X2 supera 55%, incluir Doble Oportunidad como candidato principal
    if max(m["p1"], m["p2"]) < 0.55:
        candidatos_principales.extend([
            (f"Doble Oportunidad 1X ({local} o Empate)", m["p1x"], "resultado"),
            (f"Doble Oportunidad X2 (Empate o {visita})", m["px2"], "resultado"),
        ])

    candidatos_principales.sort(key=lambda x: x[1], reverse=True)
    pick_nombre, pick_prob, pick_tipo = candidatos_principales[0]

    respaldos = [
        (f"Doble Oportunidad 1X ({local} o Empate)", m["p1x"], "resultado"),
        (f"Doble Oportunidad X2 (Empate o {visita})", m["px2"], "resultado"),
        ("Más de 1.5 Goles (Over 1.5)", m["over15"], "goles"),
        ("Menos de 3.5 Goles (Under 3.5)", m["under35"], "goles"),
    ]
    respaldos_filtrados = [r for r in respaldos if r[0] != pick_nombre]
    respaldos_filtrados.sort(key=lambda x: x[1], reverse=True)
    alt_nombre, alt_prob, _ = respaldos_filtrados[0]

    # Evaluar coherencia con últimos 5 partidos
    u_l = ult_local.head(5)
    u_v = ult_visita.head(5)
    goles_favor_l = np.where(u_l["HomeTeam"] == local, u_l["FTHG"], u_l["FTAG"]).mean() if len(u_l) else m["lam_l"]
    goles_favor_v = np.where(u_v["HomeTeam"] == visita, u_v["FTHG"], u_v["FTAG"]).mean() if len(u_v) else m["lam_v"]

    if pick_prob >= 0.65:
        confianza = "Alto"
    elif pick_prob >= 0.52:
        confianza = "Medio"
    else:
        confianza = "Bajo (partido de alta paridad)"

    total_lambda = m["lam_l"] + m["lam_v"]
    riesgo = (
        f"La proyección total de goles es ajustada ({total_lambda:.2f}); cualquier expulsión o gol temprano altera la distribución."
        if 2.2 <= total_lambda <= 2.8
        else f"{local} promedia {goles_favor_l:.1f} goles en sus últimos 5 juegos y {visita} {goles_favor_v:.1f}; vigilar rotaciones de alineación."
    )

    return (
        "> *Nota: Los servidores de Gemini presentaron saturación (503). Dictamen emitido automáticamente por el motor cuantitativo local.*\n\n"
        f"1. **PICK MÁS PROBABLE (Veredicto Principal):** **{pick_nombre}** — Probabilidad matemática: **{pick_prob*100:.1f}%** | Cuota mínima rentable (Fair Odds): **{cuota(pick_prob)}**.\n"
        f"2. **ALTERNATIVA CONSERVADORA:** **{alt_nombre}** — Probabilidad: **{alt_prob*100:.1f}%** | Cuota mínima rentable: **{cuota(alt_prob)}**.\n"
        f"3. **MARCADOR EXACTO PROYECTADO Y NIVEL DE CONFIANZA:** **{marcador_top}** | Nivel de confianza: **{confianza}**.\n"
        f"4. **JUSTIFICACIÓN Y RIESGO CLAVE:** Con fuerzas esperadas de **{m['lam_l']:.2f} goles** para {local} y **{m['lam_v']:.2f} goles** para {visita}, el modelo concentra la mayor masa de probabilidad en este escenario. **Riesgo:** {riesgo}"
    )


def mapear_equipo(nombre_api: str, equipos_csv: list[str]) -> str | None:
    if not nombre_api or not equipos_csv:
        return None
    clave = nombre_api.strip().lower()
    if clave in ALIASES_EQUIPOS and ALIASES_EQUIPOS[clave] in equipos_csv:
        return ALIASES_EQUIPOS[clave]
    for eq in equipos_csv:
        if eq.lower() == clave:
            return eq
    coincidencias = difflib.get_close_matches(nombre_api, equipos_csv, n=1, cutoff=0.45)
    return coincidencias[0] if coincidencias else None


@st.cache_data(ttl=900)
def cargar_partidos_fecha(espn_slug: str, fecha_yyyymmdd: str) -> list[dict]:
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_slug}/scoreboard?dates={fecha_yyyymmdd}"
    partidos = []
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            competidores = comp.get("competitors", [])
            local_raw, visita_raw = "", ""
            for c in competidores:
                nombre = c.get("team", {}).get("displayName", "")
                if c.get("homeAway") == "home":
                    local_raw = nombre
                elif c.get("homeAway") == "away":
                    visita_raw = nombre

            fecha_iso = ev.get("date", "")
            hora_local_str = ""
            if fecha_iso:
                try:
                    dt_utc = datetime.fromisoformat(fecha_iso.replace("Z", "+00:00"))
                    dt_loc = dt_utc.astimezone(ZONA_HORARIA_LOCAL)
                    hora_local_str = dt_loc.strftime("%I:%M %p")
                except Exception:
                    hora_local_str = fecha_iso

            estado = ev.get("status", {}).get("type", {}).get("shortDetail", "")
            if local_raw and visita_raw:
                partidos.append({
                    "local_api": local_raw,
                    "visita_api": visita_raw,
                    "hora": hora_local_str,
                    "estado": estado,
                })
    except Exception:
        return []
    return partidos


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
    prob_over15 = sum(
        matriz[i, j] for i in range(max_goles) for j in range(max_goles) if (i + j) >= 2
    )
    prob_over25 = sum(
        matriz[i, j] for i in range(max_goles) for j in range(max_goles) if (i + j) >= 3
    )
    prob_under35 = sum(
        matriz[i, j] for i in range(max_goles) for j in range(max_goles) if (i + j) <= 3
    )
    prob_under25 = 1.0 - prob_over25
    prob_1x = prob_1 + prob_x
    prob_x2 = prob_x + prob_2

    return {
        "lam_l": lambda_local,
        "lam_v": lambda_visita,
        "p1": prob_1,
        "px": prob_x,
        "p2": prob_2,
        "p1x": prob_1x,
        "px2": prob_x2,
        "btts": prob_btts,
        "over15": prob_over15,
        "over25": prob_over25,
        "under25": prob_under25,
        "under35": prob_under35,
        "matriz": matriz,
    }


def construir_contexto_ia(
    liga: str,
    temps: list[str],
    local: str,
    visita: str,
    m: dict,
    h2h_df: pd.DataFrame,
    ult_local: pd.DataFrame,
    ult_visita: pd.DataFrame,
) -> str:
    matriz = m["matriz"]
    indices_top = np.dstack(np.unravel_index(np.argsort(matriz.ravel())[::-1][:3], matriz.shape))[0]
    top_marcadores = ", ".join([f"{i}-{j} ({matriz[i, j]*100:.1f}%)" for i, j in indices_top])

    def cuota(p: float) -> str:
        return f"{1/p:.2f}" if p > 0.01 else "N/A"

    resumen_h2h = (
        h2h_df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(5).to_string(index=False)
        if not h2h_df.empty
        else "Sin enfrentamientos directos en las temporadas seleccionadas."
    )
    resumen_l = ult_local[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(5).to_string(index=False)
    resumen_v = ult_visita[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(5).to_string(index=False)

    return f"""
Actúa como un analista cuantitativo de fútbol experto en modelos de Poisson, probabilidad implícita y gestión de riesgo.
Tu objetivo es emitir un DICTAMEN ESTADÍSTICO DECISIVO sobre el escenario más probable del partido, respaldado estrictamente por estos números:

- Competición: {liga} | Temporadas: {', '.join(temps)}
- Partido: {local} (Local) vs {visita} (Visitante)
- Goles Esperados (Lambda): {local} = {m['lam_l']:.2f} | {visita} = {m['lam_v']:.2f} (Total esperado: {m['lam_l'] + m['lam_v']:.2f})
- Probabilidades 1X2 y Cuota Justa:
  * Local ({local}): {m['p1']*100:.1f}% (Cuota justa: {cuota(m['p1'])})
  * Empate: {m['px']*100:.1f}% (Cuota justa: {cuota(m['px'])})
  * Visitante ({visita}): {m['p2']*100:.1f}% (Cuota justa: {cuota(m['p2'])})
- Doble Oportunidad:
  * 1X: {m['p1x']*100:.1f}% (Cuota justa: {cuota(m['p1x'])})
  * X2: {m['px2']*100:.1f}% (Cuota justa: {cuota(m['px2'])})
- Mercados de Goles:
  * Over 1.5: {m['over15']*100:.1f}% (Cuota justa: {cuota(m['over15'])})
  * Over 2.5: {m['over25']*100:.1f}% (Cuota justa: {cuota(m['over25'])})
  * Under 2.5: {m['under25']*100:.1f}% (Cuota justa: {cuota(m['under25'])})
  * Ambos Marcan (BTTS Sí): {m['btts']*100:.1f}% (Cuota justa: {cuota(m['btts'])})
- Top 3 Marcadores Exactos: {top_marcadores}

Últimos 5 partidos de {local}:
{resumen_l}

Últimos 5 partidos de {visita}:
{resumen_v}

Historial Directo (H2H):
{resumen_h2h}
"""


st.title("Panel de Análisis Histórico & Dictamen IA")

hoy_local = datetime.now(ZONA_HORARIA_LOCAL).date()

with st.sidebar:
    st.header("1. Jornada y Competición")
    fecha_sel = st.date_input("Fecha de partidos", value=hoy_local)
    liga_nombre = st.selectbox("Competición", list(LIGAS.keys()))
    temps_sel = st.multiselect(
        "Temporadas históricas",
        list(TEMPORADAS.keys()),
        default=["2026/2027", "2025/2026"],
    )

    st.divider()
    st.header("2. Configuración del Asistente IA")
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        api_key_input = st.secrets["GEMINI_API_KEY"]
        st.success("Gemini API conectada")
    else:
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            help="Guárdala en Streamlit Secrets como GEMINI_API_KEY.",
        )
    modelo_gemini_sel = st.text_input("Modelo Gemini", value=MODELO_GEMINI_DEFAULT)

if not temps_sel:
    st.warning("Selecciona al menos una temporada histórica en el panel lateral.")
    st.stop()

cfg_liga = LIGAS[liga_nombre]
df_hist = cargar_datos_historicos(cfg_liga["csv"], [TEMPORADAS[t] for t in temps_sel])

if df_hist.empty:
    st.error("No se pudieron recuperar datos históricos para esta liga.")
    st.stop()

equipos = sorted(df_hist["HomeTeam"].unique())

# --- DETECCIÓN DE PARTIDOS DE LA FECHA ---
partidos_hoy = cargar_partidos_fecha(cfg_liga["espn"], fecha_sel.strftime("%Y%m%d"))

st.subheader(f"Cartelera detectada ({fecha_sel.strftime('%d/%m/%Y')}) — {liga_nombre}")

idx_local_default = 0
idx_visita_default = min(1, len(equipos) - 1)

if partidos_hoy:
    opciones_jornada = ["Selección manual de equipos"] + [
        f"{p['hora']} | {p['local_api']} vs {p['visita_api']} ({p['estado']})"
        for p in partidos_hoy
    ]
    partido_elegido = st.selectbox(
        "Selecciona un partido programado para cargar su análisis automáticamente:",
        opciones_jornada,
        index=1,
    )
    if partido_elegido != "Selección manual de equipos":
        pos = opciones_jornada.index(partido_elegido) - 1
        elegido = partidos_hoy[pos]
        loc_mapeado = mapear_equipo(elegido["local_api"], equipos)
        vis_mapeado = mapear_equipo(elegido["visita_api"], equipos)
        if loc_mapeado in equipos:
            idx_local_default = equipos.index(loc_mapeado)
        if vis_mapeado in equipos:
            idx_visita_default = equipos.index(vis_mapeado)
else:
    st.info(f"No hay partidos programados de {liga_nombre} para el {fecha_sel.strftime('%d/%m/%Y')}. Puedes elegir cualquier cruce manualmente abajo.")

col1, col2 = st.columns(2)
equipo_local = col1.selectbox("Equipo Local", equipos, index=idx_local_default)
equipo_visita = col2.selectbox("Equipo Visitante", equipos, index=idx_visita_default)

metricas = calcular_metricas_poisson(df_hist, equipo_local, equipo_visita)
matriz_prob = metricas["matriz"]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("λ Goles Local", f"{metricas['lam_l']:.2f}")
m2.metric("λ Goles Visita", f"{metricas['lam_v']:.2f}")
m3.metric("Local (1)", f"{metricas['p1']*100:.1f}%")
m4.metric("Empate (X)", f"{metricas['px']*100:.1f}%")
m5.metric("Visita (2)", f"{metricas['p2']*100:.1f}%")
m6.metric("Over 2.5", f"{metricas['over25']*100:.1f}%")

h2h = df_hist[
    ((df_hist["HomeTeam"] == equipo_local) & (df_hist["AwayTeam"] == equipo_visita))
    | ((df_hist["HomeTeam"] == equipo_visita) & (df_hist["AwayTeam"] == equipo_local))
]
ultimos_local = df_hist[(df_hist["HomeTeam"] == equipo_local) | (df_hist["AwayTeam"] == equipo_local)].head(15).copy()
ultimos_visita = df_hist[(df_hist["HomeTeam"] == equipo_visita) | (df_hist["AwayTeam"] == equipo_visita)].head(15).copy()

tab1, tab2, tab3, tab4 = st.tabs([
    "Dictamen Analítico IA",
    "Matriz de Marcadores",
    "Historial Directo (H2H)",
    "Tendencia de Goles",
])

with tab1:
    st.subheader(f"Dictamen Estadístico: {equipo_local} vs {equipo_visita}")

    partido_id = f"{liga_nombre}_{equipo_local}_{equipo_visita}_{'_'.join(temps_sel)}"
    if st.session_state.get("partido_actual") != partido_id:
        st.session_state["partido_actual"] = partido_id
        st.session_state["mensajes_ia"] = []

    contexto_sistema = construir_contexto_ia(
        liga_nombre, temps_sel, equipo_local, equipo_visita,
        metricas, h2h, ultimos_local, ultimos_visita,
    )

    if st.button("Emitir Dictamen Más Probable", type="primary"):
        prompt_dictamen = (
            "Emite un DICTAMEN FINAL estructurado y directo con este formato exacto:\n"
            "1. **PICK MÁS PROBABLE (Veredicto Principal):** Indica el mercado con mayor respaldo matemático "
            "(1X2, Doble Oportunidad, Over/Under 2.5 o BTTS), su probabilidad %, y la cuota mínima rentable para entrar.\n"
            "2. **ALTERNATIVA CONSERVADORA:** Una jugada de respaldo de menor varianza.\n"
            "3. **MARCADOR EXACTO PROYECTADO Y NIVEL DE CONFIANZA:** (Bajo / Medio / Alto) según coherencia entre Poisson y racha de últimos 5 partidos.\n"
            "4. **JUSTIFICACIÓN Y RIESGO CLAVE:** Explica brevemente por qué los números favorecen este dictamen y qué factor podría invalidarlo."
        )
        with st.spinner("Calculando veredicto estadístico más probable..."):
            if not api_key_input:
                respuesta = generar_dictamen_local(equipo_local, equipo_visita, metricas, ultimos_local, ultimos_visita)
                st.session_state["mensajes_ia"] = [{"role": "assistant", "content": respuesta}]
            else:
                client = genai.Client(api_key=api_key_input)
                try:
                    respuesta = consultar_gemini(
                        client,
                        modelo_gemini_sel,
                        f"{contexto_sistema}\n\nInstrucción:\n{prompt_dictamen}",
                    )
                    st.session_state["mensajes_ia"] = [{"role": "assistant", "content": respuesta}]
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                        respuesta_local = generar_dictamen_local(
                            equipo_local, equipo_visita, metricas, ultimos_local, ultimos_visita
                        )
                        st.session_state["mensajes_ia"] = [{"role": "assistant", "content": respuesta_local}]
                    else:
                        st.error(f"Error al consultar Gemini: {e}")

    for msg in st.session_state["mensajes_ia"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Pregunta por una cuota específica (ej: 'La casa paga 1.95 al Over 2.5, ¿tiene valor?')...")
    if pregunta:
        st.session_state["mensajes_ia"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Evaluando consulta..."):
                if not api_key_input:
                    respuesta = generar_dictamen_local(equipo_local, equipo_visita, metricas, ultimos_local, ultimos_visita)
                    st.markdown(respuesta)
                    st.session_state["mensajes_ia"].append({"role": "assistant", "content": respuesta})
                else:
                    client = genai.Client(api_key=api_key_input)
                    try:
                        historial_chat = "\n".join(
                            [f"{m['role'].upper()}: {m['content']}" for m in st.session_state["mensajes_ia"]]
                        )
                        respuesta = consultar_gemini(
                            client,
                            modelo_gemini_sel,
                            f"{contexto_sistema}\n\nConversación previa:\n{historial_chat}",
                        )
                        st.markdown(respuesta)
                        st.session_state["mensajes_ia"].append({"role": "assistant", "content": respuesta})
                    except Exception as e:
                        if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                            def f_cuota(p: float) -> str:
                                return f"{1/p:.2f}" if p > 0.01 else "N/A"
                            tabla_ev = (
                                "> *Servidores de Gemini saturados (503). Aquí tienes las cuotas mínimas rentables (Fair Odds) para comparar directamente contra tu casa de apuestas (si la casa paga MÁS que la cuota justa, hay Valor Esperado positivo):*\n\n"
                                f"- **Local ({equipo_local}):** {metricas['p1']*100:.1f}% → Cuota justa **{f_cuota(metricas['p1'])}**\n"
                                f"- **Empate (X):** {metricas['px']*100:.1f}% → Cuota justa **{f_cuota(metricas['px'])}**\n"
                                f"- **Visita ({equipo_visita}):** {metricas['p2']*100:.1f}% → Cuota justa **{f_cuota(metricas['p2'])}**\n"
                                f"- **Doble Oportunidad 1X:** {metricas['p1x']*100:.1f}% → Cuota justa **{f_cuota(metricas['p1x'])}**\n"
                                f"- **Doble Oportunidad X2:** {metricas['px2']*100:.1f}% → Cuota justa **{f_cuota(metricas['px2'])}**\n"
                                f"- **Over 2.5 Goles:** {metricas['over25']*100:.1f}% → Cuota justa **{f_cuota(metricas['over25'])}**\n"
                                f"- **Under 2.5 Goles:** {metricas['under25']*100:.1f}% → Cuota justa **{f_cuota(metricas['under25'])}**\n"
                                f"- **Ambos Marcan (BTTS):** {metricas['btts']*100:.1f}% → Cuota justa **{f_cuota(metricas['btts'])}**"
                            )
                            st.markdown(tabla_ev)
                            st.session_state["mensajes_ia"].append({"role": "assistant", "content": tabla_ev})
                        else:
                            st.error(f"Error al consultar Gemini: {e}")

with tab2:
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

with tab3:
    st.dataframe(h2h, use_container_width=True, hide_index=True)

with tab4:
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
