
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import json
import sqlite3
from datetime import datetime
from io import BytesIO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TriageIA · Sistema Clínico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif; }
.stApp { background: #060810; color: #dde2f0; }
[data-testid="stSidebar"] { background: #0b0f1c !important; border-right: 1px solid #151d35; }

.hero {
    background: linear-gradient(135deg, #0b0f1c 0%, #0f1628 60%, #0b0f1c 100%);
    border: 1px solid #151d35; border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; top: -30%; right: -10%; width: 50%; height: 180%;
    background: radial-gradient(ellipse, rgba(56,189,248,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.2);
    color: #38bdf8; padding: 4px 14px; border-radius: 100px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    color: #fff; line-height: 1; margin: 0; letter-spacing: -1px;
}
.hero-title span { color: #38bdf8; }
.hero-sub { font-size: 0.9rem; color: #4a5780; margin-top: 0.5rem; font-weight: 300; }

.nav-tab {
    display: flex; gap: 4px; background: #0b0f1c;
    border: 1px solid #151d35; border-radius: 12px;
    padding: 4px; margin-bottom: 1.5rem;
}
.card {
    background: #0b0f1c; border: 1px solid #151d35;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.card-label {
    font-size: 0.65rem; color: #2d3a5e; text-transform: uppercase;
    letter-spacing: 2px; font-weight: 600; margin-bottom: 0.3rem;
}
.card-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; color: #dde2f0; line-height: 1.1; }
.card-unit { font-size: 0.78rem; color: #3a4a72; font-weight: 300; }

.triage-box {
    border-radius: 18px; padding: 2.5rem; text-align: center;
    margin: 1rem 0; position: relative; overflow: hidden;
}
.triage-num {
    font-family: 'Syne', sans-serif; font-size: 6rem;
    font-weight: 800; line-height: 0.9; margin: 0;
}
.triage-lbl {
    font-size: 0.85rem; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; margin-top: 0.6rem; opacity: 0.85;
}
.triage-desc { font-size: 0.82rem; opacity: 0.55; margin-top: 0.5rem; }

.alert-critical {
    background: rgba(239,68,68,0.07); border: 1px solid rgba(239,68,68,0.25);
    border-left: 4px solid #ef4444; border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem; margin: 0.8rem 0;
}
.alert-info {
    background: rgba(56,189,248,0.07); border: 1px solid rgba(56,189,248,0.2);
    border-left: 4px solid #38bdf8; border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem; margin: 0.8rem 0;
}

.prob-row { margin: 5px 0; }
.prob-track { background: #0f1628; border-radius: 100px; height: 7px; margin-top: 3px; }
.prob-fill { height: 7px; border-radius: 100px; }

.section-title {
    font-family: 'Syne', sans-serif; font-size: 0.65rem; color: #2d3a5e;
    text-transform: uppercase; letter-spacing: 3px; font-weight: 600;
    padding-bottom: 0.6rem; border-bottom: 1px solid #151d35; margin-bottom: 1rem;
}

.stat-big {
    font-family: 'Syne', sans-serif; font-size: 2.8rem;
    font-weight: 800; line-height: 1; color: #dde2f0;
}
.stat-label { font-size: 0.72rem; color: #2d3a5e; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

.altitude-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2);
    color: #fbbf24; padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 500;
}

.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Instrument Sans', sans-serif !important; font-size: 0.9rem !important;
    font-weight: 600 !important; padding: 0.65rem 1.5rem !important;
    width: 100% !important; letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    box-shadow: 0 4px 20px rgba(29,78,216,0.35) !important;
}

label { color: #4a5780 !important; font-size: 0.82rem !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
TRIAGE_CFG = {
    1: {"color":"#ef4444","bg":"rgba(239,68,68,0.1)","border":"rgba(239,68,68,0.35)",
        "emoji":"🔴","label":"ATENCIÓN INMEDIATA","desc":"Riesgo de muerte · intervención inmediata","tiempo":"Inmediato"},
    2: {"color":"#f97316","bg":"rgba(249,115,22,0.1)","border":"rgba(249,115,22,0.35)",
        "emoji":"🟠","label":"MUY URGENTE","desc":"Situación grave · < 10 min","tiempo":"< 10 min"},
    3: {"color":"#eab308","bg":"rgba(234,179,8,0.1)","border":"rgba(234,179,8,0.35)",
        "emoji":"🟡","label":"URGENTE","desc":"Requiere atención pronta · < 30 min","tiempo":"< 30 min"},
    4: {"color":"#22c55e","bg":"rgba(34,197,94,0.1)","border":"rgba(34,197,94,0.35)",
        "emoji":"🟢","label":"MENOS URGENTE","desc":"Puede esperar · < 60 min","tiempo":"< 60 min"},
    5: {"color":"#3b82f6","bg":"rgba(59,130,246,0.1)","border":"rgba(59,130,246,0.35)",
        "emoji":"🔵","label":"NO URGENTE","desc":"Consulta general · < 120 min","tiempo":"< 120 min"},
}

# Ajuste de saturación por altitud (msnm)
ALTITUD_ZONAS = {
    "Costa / Litoral (0–500 msnm)":       {"min": 0,    "max": 500,   "sat_normal": (95,100), "ajuste": 0},
    "Piedemonte (500–1500 msnm)":          {"min": 500,  "max": 1500,  "sat_normal": (94,99),  "ajuste": -1},
    "Zona Media (1500–2500 msnm)":         {"min": 1500, "max": 2500,  "sat_normal": (93,98),  "ajuste": -2},
    "Zona Alta — Cauca/Nariño (2500–3200 msnm)": {"min": 2500, "max": 3200, "sat_normal": (91,96), "ajuste": -3},
    "Alta Montaña (>3200 msnm)":           {"min": 3200, "max": 5000,  "sat_normal": (88,94),  "ajuste": -5},
}

# ── DB local (sqlite) para historial de pacientes ──────────────────────────────
def init_db():
    conn = sqlite3.connect("triage_local.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS registros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, nombre TEXT, edad INTEGER,
            temperatura REAL, fc INTEGER, fr INTEGER,
            saturacion INTEGER, presion INTEGER,
            confusion INTEGER, oxigeno INTEGER,
            perfil TEXT, zona TEXT, news2 INTEGER,
            triage INTEGER, fuente TEXT
        )
    """)
    conn.commit()
    conn.close()

def guardar_registro(data):
    conn = sqlite3.connect("triage_local.db")
    c = conn.cursor()
    c.execute("""INSERT INTO registros
        (timestamp,nombre,edad,temperatura,fc,fr,saturacion,presion,
         confusion,oxigeno,perfil,zona,news2,triage,fuente)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", data)
    conn.commit()
    conn.close()

def cargar_historial():
    conn = sqlite3.connect("triage_local.db")
    df = pd.read_sql_query("SELECT * FROM registros ORDER BY id DESC LIMIT 100", conn)
    conn.close()
    return df

init_db()

# ── Funciones NEWS2 ────────────────────────────────────────────────────────────
def score_fr(fr):
    if fr<=8: return 3
    if fr<=11: return 1
    if fr<=20: return 0
    if fr<=24: return 2
    return 3

def score_sat(s, ajuste=0):
    umbral = [91+ajuste, 93+ajuste, 95+ajuste]
    if s<=umbral[0]: return 3
    if s<=umbral[1]: return 2
    if s<=umbral[2]: return 1
    return 0

def score_temp(t):
    if t<=35.0: return 3
    if t<=36.0: return 1
    if t<=38.0: return 0
    if t<=39.0: return 1
    return 2

def score_pas(p):
    if p<=90: return 3
    if p<=100: return 2
    if p<=110: return 1
    if p<=219: return 0
    return 3

def score_fc(f):
    if f<=40: return 3
    if f<=50: return 1
    if f<=90: return 0
    if f<=110: return 1
    if f<=130: return 2
    return 3

def calcular_news2(fr, sat, temp, presion, fc, confusion, oxigeno, ajuste_sat=0):
    return (score_fr(fr) + score_sat(sat, ajuste_sat) + score_temp(temp) +
            score_pas(presion) + score_fc(fc) +
            (3 if confusion else 0) + (2 if oxigeno else 0))

def reglas_criticas(sat, presion, fr, fc, temp, confusion, ajuste=0):
    sat_adj = sat - ajuste
    if sat_adj < 85:                                           return 1, "Hipoxia crítica (SatO₂ ajustada < 85%)"
    if presion < 80:                                           return 1, "Shock circulatorio (PA < 80 mmHg)"
    if fr > 35:                                                return 1, "Fallo respiratorio (FR > 35 resp/min)"
    if fc > 150:                                               return 1, "Taquicardia severa (FC > 150 lat/min)"
    if temp>39 and fc>120 and presion<90 and confusion:        return 1, "Sospecha de sepsis"
    if fc>130 and presion<85:                                  return 1, "Posible shock"
    if presion > 180:                                          return 2, "Crisis hipertensiva"
    if temp>38 and (sat-ajuste)<93:                            return 2, "Fiebre con hipoxia"
    if (sat-ajuste)<95 and fr>22:                              return 2, "Disnea con hipoxemia"
    if confusion:                                              return 2, "Alteración del estado mental"
    return None, None

def news2_a_triage(score):
    if score>=9: return 1
    if score>=7: return 2
    if score>=5: return 3
    if score>=3: return 4
    return 5

# ── Cargar / entrenar modelo ───────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    if os.path.exists("modelo_triage.pkl") and os.path.exists("columnas_modelo.pkl"):
        return joblib.load("modelo_triage.pkl"), joblib.load("columnas_modelo.pkl")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    np.random.seed(42); N=40000
    PERFILES=["estable","infeccion","sepsis","resp_leve","resp_grave","anciano","critico"]
    PROPS=[0.35,0.20,0.10,0.15,0.08,0.07,0.05]
    tipos=np.random.choice(PERFILES,size=N,p=PROPS)
    params={
        "estable":   dict(edad=(18,65),temp=(36.5,0.4),fc=(60,100), fr=(12,20), sat=(95,100),pres=(100,130),p_c=0.0),
        "infeccion": dict(edad=(18,70),temp=(38.2,0.6),fc=(90,120), fr=(16,24), sat=(92,98), pres=(90,130), p_c=0.05),
        "sepsis":    dict(edad=(18,75),temp=(39.2,0.6),fc=(110,150),fr=(20,32), sat=(85,93), pres=(75,100), p_c=0.20),
        "resp_leve": dict(edad=(18,70),temp=(37.2,0.5),fc=(80,115), fr=(20,26), sat=(90,95), pres=(90,135), p_c=0.0),
        "resp_grave":dict(edad=(18,80),temp=(38.6,0.5),fc=(105,145),fr=(26,38), sat=(75,88), pres=(80,120), p_c=0.12),
        "anciano":   dict(edad=(70,95),temp=(36.4,0.5),fc=(55,95),  fr=(14,22), sat=(92,99), pres=(110,160),p_c=0.22),
        "critico":   dict(edad=(18,85),temp=(39.6,0.6),fc=(125,165),fr=(30,42), sat=(68,84), pres=(55,85),  p_c=0.35),
    }
    edad=np.zeros(N);temp=np.zeros(N);fc=np.zeros(N);fr=np.zeros(N)
    sat=np.zeros(N);pres=np.zeros(N);conf=np.zeros(N,dtype=int)
    for i,t in enumerate(tipos):
        p=params[t]
        edad[i]=np.random.randint(*p["edad"]); temp[i]=np.random.normal(p["temp"][0],p["temp"][1])
        fc[i]=np.random.randint(*p["fc"]);     fr[i]=np.random.randint(*p["fr"])
        sat[i]=np.random.randint(*p["sat"]);   pres[i]=np.random.randint(*p["pres"])
        conf[i]=np.random.binomial(1,p["p_c"])
    temp=np.clip(temp,34,41.5); fc=np.clip(fc,35,180); fr=np.clip(fr,6,45)
    sat=np.clip(sat,65,100);    pres=np.clip(pres,55,210)
    prob_o2=np.clip((94-sat)/12,0,1); oxigeno=np.random.binomial(1,prob_o2).astype(int)
    oxigeno[sat>93]=0
    data=pd.DataFrame({"edad":edad,"temperatura":temp,"frecuencia_cardiaca":fc,
                        "frecuencia_respiratoria":fr,"saturacion":sat,"presion_sistolica":pres,
                        "confusion":conf,"oxigeno_suplementario":oxigeno,"perfil_clinico":tipos})
    def _t(r):
        n2=calcular_news2(r.frecuencia_respiratoria,r.saturacion,r.temperatura,
                          r.presion_sistolica,r.frecuencia_cardiaca,r.confusion,r.oxigeno_suplementario)
        rv,_=reglas_criticas(r.saturacion,r.presion_sistolica,r.frecuencia_respiratoria,
                              r.frecuencia_cardiaca,r.temperatura,r.confusion)
        return rv if rv else news2_a_triage(n2)
    data["triage"]=data.apply(_t,axis=1)
    dm=pd.get_dummies(data,columns=["perfil_clinico"])
    X=dm.drop(columns=["triage"]); y=dm["triage"]
    Xtr,_,ytr,_=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    m=RandomForestClassifier(n_estimators=300,max_depth=14,class_weight="balanced",random_state=42,n_jobs=-1)
    m.fit(Xtr,ytr)
    return m, list(X.columns)

with st.spinner("⏳ Iniciando TriageIA..."):
    model, COLUMNAS = cargar_modelo()

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">🏥 Manchester Triage System · NEWS2 · v2.0</div>
  <h1 class="hero-title">Triage<span>IA</span></h1>
  <p class="hero-sub">Infraestructura de admisión y priorización clínica · Suroccidente colombiano</p>
</div>
""", unsafe_allow_html=True)

# ── NAVEGACIÓN ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡ Evaluación de Triage", "📊 Dashboard Clínico", "📋 Historial de Pacientes"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.markdown('<div class="section-title">👤 Datos del Paciente</div>', unsafe_allow_html=True)
        nombre = st.text_input("Nombre / ID del paciente", placeholder="Ej: Juan Pérez o #0042")

        perfil = st.selectbox("Perfil clínico", [
            "estable","infeccion","sepsis","resp_leve","resp_grave","anciano","critico"
        ], format_func=lambda x: {
            "estable":"🟢 Paciente estable","infeccion":"🟡 Infección leve/moderada",
            "sepsis":"🟠 Sepsis moderada","resp_leve":"🟡 Compromiso resp. leve",
            "resp_grave":"🔴 Insuficiencia respiratoria","anciano":"🔵 Adulto mayor",
            "critico":"🔴 Paciente crítico"
        }[x])

        zona = st.selectbox("📍 Zona geográfica / Altitud", list(ALTITUD_ZONAS.keys()))
        ajuste_sat = ALTITUD_ZONAS[zona]["ajuste"]
        sat_rango = ALTITUD_ZONAS[zona]["sat_normal"]

        st.markdown(f"""
        <div class="altitude-badge">
            ⛰️ Ajuste altitud: {ajuste_sat:+d} pts saturación
            · Normal local: {sat_rango[0]}–{sat_rango[1]}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1.2rem;">🫀 Signos Vitales</div>', unsafe_allow_html=True)
        edad     = st.slider("Edad (años)",                    1,  99, 45)
        temp     = st.slider("Temperatura (°C)",             34.0,41.5,36.8,step=0.1)
        fc_val   = st.slider("Frecuencia cardíaca (lat/min)", 30, 180, 80)
        fr_val   = st.slider("Frecuencia resp. (resp/min)",   6,  45,  16)
        sat_val  = st.slider("Saturación O₂ (%)",            65, 100, 97)
        pres_val = st.slider("Presión sistólica (mmHg)",      55, 220, 120)

        st.markdown('<div class="section-title" style="margin-top:1rem;">⚠️ Síntomas</div>', unsafe_allow_html=True)
        confusion_val = st.checkbox("Confusión / alteración mental")
        oxigeno_val   = st.checkbox("Oxígeno suplementario")

        evaluar = st.button("⚡ Evaluar y Registrar Triage")

    if not evaluar:
        c1, c2, c3 = st.columns(3)
        for col, (ico, tit, desc) in zip([c1,c2,c3],[
            ("⚡","Ingresa los datos","Signos vitales + zona geográfica en el panel izquierdo"),
            ("🧠","Evalúa el triage","El modelo ajusta los rangos según la altitud de la zona"),
            ("📋","Registro automático","Cada evaluación queda guardada en el historial local"),
        ]):
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:2rem 1.5rem;">
                    <div style="font-size:2rem;margin-bottom:.8rem;">{ico}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                                color:#dde2f0;margin-bottom:.4rem;">{tit}</div>
                    <div style="font-size:.82rem;color:#3a4a72;line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-info">
            <div style="font-weight:600;color:#38bdf8;font-size:.82rem;margin-bottom:.3rem;">
                🗺️ Nuevo: Ajuste por geolocalización y altitud
            </div>
            <div style="color:#7dd3fc;font-size:.8rem;">
                El sistema recalibra los rangos de saturación de oxígeno según la altitud de la zona,
                evitando sobrediagnóstico en comunidades de alta montaña como el Cauca y Nariño.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Cálculos
        news2_score = calcular_news2(fr_val, sat_val, temp, pres_val, fc_val,
                                     int(confusion_val), int(oxigeno_val), ajuste_sat)
        regla_nivel, regla_desc = reglas_criticas(sat_val, pres_val, fr_val, fc_val,
                                                   temp, confusion_val, ajuste_sat)
        fila = {"edad":edad,"temperatura":temp,"frecuencia_cardiaca":fc_val,
                "frecuencia_respiratoria":fr_val,"saturacion":sat_val,"presion_sistolica":pres_val,
                "confusion":int(confusion_val),"oxigeno_suplementario":int(oxigeno_val),
                f"perfil_clinico_{perfil}":1}
        df_pred = pd.DataFrame([fila]).reindex(columns=COLUMNAS, fill_value=0)
        triage_ml   = int(model.predict(df_pred)[0])
        probabilidades = model.predict_proba(df_pred)[0]
        triage_final = regla_nivel if regla_nivel else triage_ml
        cfg = TRIAGE_CFG[triage_final]

        # Guardar en DB
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fuente = "Regla clínica" if regla_nivel else "Modelo ML"
        guardar_registro((ts, nombre or "Anónimo", edad, temp, fc_val, fr_val,
                          sat_val, pres_val, int(confusion_val), int(oxigeno_val),
                          perfil, zona, news2_score, triage_final, fuente))

        col_r, col_d = st.columns([1, 1.6], gap="large")

        with col_r:
            st.markdown(f"""
            <div class="triage-box" style="background:{cfg['bg']};border:2px solid {cfg['border']};">
                <div style="font-size:.65rem;color:{cfg['color']};text-transform:uppercase;
                            letter-spacing:3px;font-weight:700;margin-bottom:.4rem;">Nivel de Triage</div>
                <div class="triage-num" style="color:{cfg['color']};">{cfg['emoji']} {triage_final}</div>
                <div class="triage-lbl" style="color:{cfg['color']};">{cfg['label']}</div>
                <div class="triage-desc">{cfg['desc']}</div>
                <div style="margin-top:1rem;font-size:.72rem;color:#2d3a5e;">
                    {'⚠️ Regla clínica crítica' if regla_nivel else '🤖 Predicción del modelo ML'}
                    &nbsp;·&nbsp; {ts}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if regla_nivel:
                st.markdown(f"""
                <div class="alert-critical">
                    <div style="font-weight:700;color:#ef4444;font-size:.82rem;margin-bottom:.3rem;">
                        ⚠️ Regla crítica activada
                    </div>
                    <div style="color:#fca5a5;font-size:.8rem;">{regla_desc}</div>
                </div>
                """, unsafe_allow_html=True)

            # Zona geográfica
            st.markdown(f"""
            <div class="alert-info">
                <div style="font-weight:600;color:#38bdf8;font-size:.78rem;margin-bottom:.2rem;">
                    ⛰️ Ajuste altitudinal aplicado
                </div>
                <div style="color:#7dd3fc;font-size:.76rem;">
                    {zona}<br>
                    Saturación ajustada: {sat_val} % → score corregido {ajuste_sat:+d} pts
                </div>
            </div>
            """, unsafe_allow_html=True)

            # NEWS2
            news2_color = "#ef4444" if news2_score>=7 else "#f97316" if news2_score>=5 else "#eab308" if news2_score>=3 else "#22c55e"
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="card-label">Score NEWS2</div>
                <div style="font-family:'Syne',sans-serif;font-size:3.5rem;font-weight:800;
                            color:{news2_color};line-height:1;">{news2_score}</div>
                <div style="font-size:.72rem;color:#2d3a5e;margin-top:.3rem;">
                    {'Crítico ≥9' if news2_score>=9 else 'Muy urgente 7-8' if news2_score>=7
                     else 'Urgente 5-6' if news2_score>=5 else 'Vigilancia 3-4' if news2_score>=3
                     else 'Bajo riesgo 0-2'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="section-title">📊 Signos Vitales Registrados</div>', unsafe_allow_html=True)
            v1,v2 = st.columns(2)
            vitales = [("Edad",f"{edad}","años"),("Temperatura",f"{temp:.1f}","°C"),
                       ("Frec. cardíaca",f"{fc_val}","lat/min"),("Frec. respiratoria",f"{fr_val}","resp/min"),
                       ("Saturación O₂",f"{sat_val}","% (zona)"),("Presión sistólica",f"{pres_val}","mmHg")]
            for i,(lbl,val,unit) in enumerate(vitales):
                with (v1 if i%2==0 else v2):
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-label">{lbl}</div>
                        <div class="card-value">{val} <span class="card-unit">{unit}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="margin-top:.5rem;">📈 Probabilidades por nivel</div>',
                        unsafe_allow_html=True)
            for clase, prob in sorted(zip(model.classes_, probabilidades)):
                c = TRIAGE_CFG[clase]
                pct = prob*100
                st.markdown(f"""
                <div class="prob-row">
                    <div style="display:flex;justify-content:space-between;font-size:.76rem;color:#4a5780;">
                        <span>{c['emoji']} T{clase} — {c['label']}</span>
                        <span style="color:#dde2f0;font-weight:600;">{pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{max(pct,1.5)}%;background:{c['color']};opacity:.8;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Desglose NEWS2
        st.markdown("---")
        st.markdown('<div class="section-title">🔬 Desglose Score NEWS2</div>', unsafe_allow_html=True)
        scores = {
            "FR":       (f"{fr_val} resp/min",  score_fr(fr_val)),
            "SatO₂":    (f"{sat_val}%",          score_sat(sat_val, ajuste_sat)),
            "Temp":     (f"{temp:.1f}°C",        score_temp(temp)),
            "PAS":      (f"{pres_val} mmHg",     score_pas(pres_val)),
            "FC":       (f"{fc_val} lat/min",    score_fc(fc_val)),
            "Mental":   ("Confusión" if confusion_val else "Normal", 3 if confusion_val else 0),
            "O₂ Supl.": ("Sí" if oxigeno_val else "No", 2 if oxigeno_val else 0),
        }
        cols = st.columns(7)
        for col,(var,(valor,score)) in zip(cols, scores.items()):
            sc = "#ef4444" if score>=3 else "#f97316" if score==2 else "#eab308" if score==1 else "#22c55e"
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center;padding:1rem .8rem;">
                    <div class="card-label" style="font-size:.6rem;">{var}</div>
                    <div style="font-size:.78rem;color:#3a4a72;margin:.2rem 0;">{valor}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:2rem;
                                font-weight:800;color:{sc};line-height:1;">{score}</div>
                    <div style="font-size:.6rem;color:#2d3a5e;">pts</div>
                </div>
                """, unsafe_allow_html=True)

        # Exportar PDF básico
        st.markdown("---")
        export_data = {
            "timestamp": ts, "paciente": nombre or "Anónimo",
            "triage": triage_final, "nivel": cfg['label'],
            "news2": news2_score, "zona": zona,
            "signos_vitales": {
                "edad": edad, "temperatura": temp, "fc": fc_val,
                "fr": fr_val, "saturacion": sat_val, "presion": pres_val,
                "confusion": confusion_val, "oxigeno": oxigeno_val
            },
            "fuente_decision": fuente
        }
        st.download_button(
            label="📄 Exportar registro del paciente (JSON)",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"triage_{nombre or 'paciente'}_{ts[:10]}.json",
            mime="application/json"
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Dashboard Clínico en Tiempo Real</div>', unsafe_allow_html=True)
    df_hist = cargar_historial()

    if df_hist.empty:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">📊</div>
            <div style="font-family:'Syne',sans-serif;font-size:1rem;color:#dde2f0;">
                Sin datos aún
            </div>
            <div style="font-size:.82rem;color:#2d3a5e;margin-top:.4rem;">
                Evalúa tu primer paciente para ver el dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(df_hist)
        criticos = len(df_hist[df_hist["triage"]<=2])
        pct_criticos = (criticos/total*100) if total>0 else 0
        news2_prom = df_hist["news2"].mean()

        c1,c2,c3,c4 = st.columns(4)
        for col,(val,lbl,color) in zip([c1,c2,c3,c4],[
            (str(total),       "Total pacientes",    "#dde2f0"),
            (str(criticos),    "Críticos (T1+T2)",   "#ef4444"),
            (f"{pct_criticos:.1f}%","% Críticos",    "#f97316"),
            (f"{news2_prom:.1f}","NEWS2 promedio",   "#38bdf8"),
        ]):
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <div class="stat-big" style="color:{color};">{val}</div>
                    <div class="stat-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        c_graf1, c_graf2 = st.columns(2)

        with c_graf1:
            st.markdown('<div class="section-title">Distribución por nivel de triage</div>',
                        unsafe_allow_html=True)
            dist = df_hist["triage"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor("#0b0f1c")
            ax.set_facecolor("#0b0f1c")
            colors_bar = [TRIAGE_CFG[i]["color"] for i in dist.index]
            bars = ax.bar(dist.index, dist.values, color=colors_bar,
                          edgecolor="#060810", linewidth=2, width=0.6)
            for bar,val in zip(bars,dist.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        str(val), ha="center", va="bottom",
                        color="#dde2f0", fontsize=10, fontweight="bold")
            ax.set_xlabel("Nivel de Triage", color="#4a5780")
            ax.set_ylabel("Pacientes", color="#4a5780")
            ax.tick_params(colors="#4a5780")
            for spine in ax.spines.values(): spine.set_color("#151d35")
            ax.grid(True, alpha=0.1, color="#151d35", axis="y")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with c_graf2:
            st.markdown('<div class="section-title">Distribución por zona geográfica</div>',
                        unsafe_allow_html=True)
            zona_dist = df_hist["zona"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6,4))
            fig2.patch.set_facecolor("#0b0f1c")
            ax2.set_facecolor("#0b0f1c")
            colores_zona = ["#38bdf8","#818cf8","#34d399","#fbbf24","#f87171"]
            wedges, texts, autotexts = ax2.pie(
                zona_dist.values, labels=None,
                autopct="%1.0f%%", colors=colores_zona[:len(zona_dist)],
                pctdistance=0.75, startangle=90,
                wedgeprops=dict(edgecolor="#060810", linewidth=2)
            )
            for at in autotexts:
                at.set_color("#dde2f0"); at.set_fontsize(9)
            ax2.legend(zona_dist.index, loc="lower center",
                       bbox_to_anchor=(0.5,-0.15), ncol=2,
                       fontsize=7, labelcolor="#4a5780",
                       facecolor="#0b0f1c", edgecolor="#151d35")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📋 Historial de Pacientes (almacenamiento local)</div>',
                unsafe_allow_html=True)
    df_hist2 = cargar_historial()
    if df_hist2.empty:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;">
            <div style="font-size:2rem;margin-bottom:1rem;">📋</div>
            <div style="color:#3a4a72;font-size:.9rem;">No hay registros aún</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cols_show = ["timestamp","nombre","edad","triage","news2","zona","fuente"]
        df_show = df_hist2[cols_show].copy()
        df_show.columns = ["Fecha/Hora","Paciente","Edad","Triage","NEWS2","Zona","Fuente decisión"]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.download_button(
            label="📥 Exportar historial completo (CSV)",
            data=df_hist2.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"historial_triage_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        st.markdown("""
        <div class="alert-info" style="margin-top:1rem;">
            <div style="font-weight:600;color:#38bdf8;font-size:.78rem;margin-bottom:.2rem;">
                🔒 Almacenamiento local
            </div>
            <div style="color:#7dd3fc;font-size:.76rem;">
                Los datos se guardan localmente en el dispositivo (SQLite).
                Para integración con HIS hospitalario, exporta el CSV o el JSON individual por paciente.
                Próxima versión: exportación en formato FHIR R4.
            </div>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div style="margin-top:2rem;padding:1rem;text-align:center;border-top:1px solid #151d35;">
    <p style="color:#1e293b;font-size:.72rem;margin:0;">
        TriageIA v2.0 · Manchester Triage System + NEWS2 · Ajuste altitudinal Colombia ·
        <strong style="color:#2d3a5e;">Sistema académico — No reemplaza el juicio médico profesional</strong>
    </p>
</div>
""", unsafe_allow_html=True)
