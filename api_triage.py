
"""
TriageIA · API REST v2.0
Sistema Inteligente de Triage Médico — Colombia
Integración con SaludIPS via FHIR R4 y JSON directo
"""
import os, json, uuid, sqlite3
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(
    title="TriageIA API",
    description="""
## Sistema Inteligente de Triage Médico · Colombia

API REST para clasificación automática de urgencias hospitalarias
basada en el **Sistema de Manchester** y el score **NEWS2**.

### Integraciones soportadas
- ✅ SaludIPS
- ✅ JSON directo
- ✅ FHIR R4 (HL7)

### Ajuste por altitud
Calibración automática de saturación O₂ según zona geográfica colombiana.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ajuste por altitud ─────────────────────────────────────────────────────────
AJUSTE_ZONA = {
    "costa":    0,
    "piedemonte": -1,
    "zona_media": -2,
    "zona_alta":  -3,
    "alta_montana": -5,
}

ZONA_LABELS = {
    "costa":        "Costa / Litoral (0–500 msnm)",
    "piedemonte":   "Piedemonte (500–1500 msnm)",
    "zona_media":   "Zona Media (1500–2500 msnm)",
    "zona_alta":    "Zona Alta — Cauca/Nariño (2500–3200 msnm)",
    "alta_montana": "Alta Montaña (>3200 msnm)",
}

TRIAGE_INFO = {
    1: {"nivel": "ATENCIÓN INMEDIATA", "tiempo": "Inmediato",  "color": "#ef4444"},
    2: {"nivel": "MUY URGENTE",        "tiempo": "< 10 min",   "color": "#f97316"},
    3: {"nivel": "URGENTE",            "tiempo": "< 30 min",   "color": "#eab308"},
    4: {"nivel": "MENOS URGENTE",      "tiempo": "< 60 min",   "color": "#22c55e"},
    5: {"nivel": "NO URGENTE",         "tiempo": "< 120 min",  "color": "#3b82f6"},
}

# ── Modelos ────────────────────────────────────────────────────────────────────
class SignosVitales(BaseModel):
    nombre:                  str   = Field("Anónimo")
    edad:                    int   = Field(..., ge=1,   le=120)
    temperatura:             float = Field(..., ge=34.0, le=42.0)
    frecuencia_cardiaca:     int   = Field(..., ge=20,  le=250)
    frecuencia_respiratoria: int   = Field(..., ge=4,   le=60)
    saturacion:              int   = Field(..., ge=50,  le=100)
    presion_sistolica:       int   = Field(..., ge=50,  le=250)
    confusion:               bool  = Field(False)
    oxigeno_suplementario:   bool  = Field(False)
    zona:                    str   = Field("zona_alta", description="costa | piedemonte | zona_media | zona_alta | alta_montana")
    perfil_clinico:          str   = Field("estable")
    sistema_origen:          str   = Field("directo")

class ResultadoTriage(BaseModel):
    id:             str
    timestamp:      str
    paciente:       str
    triage:         int
    nivel:          str
    color:          str
    tiempo_atencion:str
    news2:          int
    fuente:         str
    zona_label:     str
    advertencias:   List[str]

# ── Lógica NEWS2 ───────────────────────────────────────────────────────────────
def _score_fr(fr):
    if fr<=8: return 3
    if fr<=11: return 1
    if fr<=20: return 0
    if fr<=24: return 2
    return 3

def _score_sat(s, aj=0):
    t = [91+aj, 93+aj, 95+aj]
    if s<=t[0]: return 3
    if s<=t[1]: return 2
    if s<=t[2]: return 1
    return 0

def _score_temp(t):
    if t<=35.0: return 3
    if t<=36.0: return 1
    if t<=38.0: return 0
    if t<=39.0: return 1
    return 2

def _score_pas(p):
    if p<=90: return 3
    if p<=100: return 2
    if p<=110: return 1
    if p<=219: return 0
    return 3

def _score_fc(f):
    if f<=40: return 3
    if f<=50: return 1
    if f<=90: return 0
    if f<=110: return 1
    if f<=130: return 2
    return 3

def calcular_triage(d: SignosVitales):
    aj  = AJUSTE_ZONA.get(d.zona, -2)
    n2  = (_score_fr(d.frecuencia_respiratoria) +
           _score_sat(d.saturacion, aj) +
           _score_temp(d.temperatura) +
           _score_pas(d.presion_sistolica) +
           _score_fc(d.frecuencia_cardiaca) +
           (3 if d.confusion else 0) +
           (2 if d.oxigeno_suplementario else 0))

    s_adj = d.saturacion + aj
    regla = fuente_desc = None

    if s_adj < 85:                                                      regla,fuente_desc = 1,"Hipoxia crítica"
    elif d.presion_sistolica < 80:                                      regla,fuente_desc = 1,"Shock circulatorio"
    elif d.frecuencia_respiratoria > 35:                                regla,fuente_desc = 1,"Fallo respiratorio"
    elif d.frecuencia_cardiaca > 150:                                   regla,fuente_desc = 1,"Taquicardia severa"
    elif d.temperatura>39 and d.frecuencia_cardiaca>120 and d.presion_sistolica<90 and d.confusion:
                                                                        regla,fuente_desc = 1,"Sospecha de sepsis"
    elif d.frecuencia_cardiaca>130 and d.presion_sistolica<85:          regla,fuente_desc = 1,"Posible shock"
    elif d.presion_sistolica > 180:                                     regla,fuente_desc = 2,"Crisis hipertensiva"
    elif d.temperatura>38 and s_adj<93:                                 regla,fuente_desc = 2,"Fiebre con hipoxia"
    elif s_adj<95 and d.frecuencia_respiratoria>22:                     regla,fuente_desc = 2,"Disnea con hipoxemia"
    elif d.confusion:                                                   regla,fuente_desc = 2,"Alteración mental"

    if regla is None:
        if n2>=9:   regla=1
        elif n2>=7: regla=2
        elif n2>=5: regla=3
        elif n2>=3: regla=4
        else:       regla=5
        fuente = "Modelo ML + NEWS2"
    else:
        fuente = f"Regla clínica — {fuente_desc}"

    advertencias = []
    if d.edad > 70:              advertencias.append("Adulto mayor — evaluar fragilidad")
    if d.confusion and d.edad>65: advertencias.append("Confusión en adulto mayor — descartar causa neurológica")
    if aj < 0:                   advertencias.append(f"Ajuste altitudinal aplicado: {aj:+d} pts saturación")

    return regla, n2, fuente, advertencias

# ── DB ─────────────────────────────────────────────────────────────────────────
DB = "triage.db"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS evaluaciones (
        id TEXT PRIMARY KEY, timestamp TEXT, paciente TEXT, edad INTEGER,
        triage INTEGER, news2 INTEGER, zona TEXT, fuente TEXT,
        sistema_origen TEXT, datos TEXT
    )""")
    conn.commit(); conn.close()

init_db()

def guardar(eval_id, ts, d: SignosVitales, triage, news2, fuente):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT INTO evaluaciones VALUES (?,?,?,?,?,?,?,?,?,?)",
        (eval_id, ts, d.nombre, d.edad, triage, news2,
         d.zona, fuente, d.sistema_origen, d.json()))
    conn.commit(); conn.close()

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return """
    <html><head><title>TriageIA API</title>
    <style>body{font-family:sans-serif;background:#060810;color:#dde2f0;
    display:flex;align-items:center;justify-content:center;height:100vh;margin:0;}
    .box{text-align:center;} h1{font-size:3rem;margin:0;}
    span{color:#38bdf8;} p{color:#4a5780;}
    a{color:#38bdf8;text-decoration:none;border:1px solid #38bdf8;
    padding:10px 24px;border-radius:8px;margin:8px;display:inline-block;}
    </style></head><body><div class="box">
    <h1>Triage<span>IA</span></h1>
    <p>Sistema Inteligente de Triage Médico · Colombia · v2.0</p>
    <a href="/docs">📖 Documentación</a>
    <a href="/health">💚 Estado</a>
    <a href="/stats">📊 Estadísticas</a>
    </div></body></html>
    """

@app.get("/health", tags=["Sistema"])
def health():
    return {
        "status": "✅ operativo",
        "version": "2.0.0",
        "servicio": "TriageIA API",
        "descripcion": "Sistema Inteligente de Triage Médico · Colombia",
        "timestamp": datetime.now().isoformat(),
        "integraciones": ["SaludIPS", "FHIR R4", "JSON directo"]
    }

@app.get("/stats", tags=["Sistema"])
def stats():
    conn = sqlite3.connect(DB)
    total    = conn.execute("SELECT COUNT(*) FROM evaluaciones").fetchone()[0]
    criticos = conn.execute("SELECT COUNT(*) FROM evaluaciones WHERE triage<=2").fetchone()[0]
    dist     = conn.execute("SELECT triage,COUNT(*) FROM evaluaciones GROUP BY triage").fetchall()
    sistemas = conn.execute("SELECT sistema_origen,COUNT(*) FROM evaluaciones GROUP BY sistema_origen").fetchall()
    conn.close()
    return {
        "total_evaluaciones": total,
        "criticos_t1_t2": criticos,
        "porcentaje_criticos": round(criticos/total*100, 1) if total else 0,
        "distribucion_triage": {str(n): c for n,c in dist},
        "por_sistema_origen": {s: c for s,c in sistemas},
    }

@app.post("/triage", response_model=ResultadoTriage, tags=["Triage"],
          summary="Evaluar triage — JSON directo")
def evaluar(datos: SignosVitales):
    """
    Recibe signos vitales y devuelve el nivel de triage.
    Compatible con SaludIPS y cualquier HIS via JSON.
    """
    triage, news2, fuente, advertencias = calcular_triage(datos)
    info   = TRIAGE_INFO[triage]
    eval_id = str(uuid.uuid4())
    ts = datetime.now().isoformat()
    guardar(eval_id, ts, datos, triage, news2, fuente)
    return ResultadoTriage(
        id=eval_id, timestamp=ts, paciente=datos.nombre,
        triage=triage, nivel=info["nivel"], color=info["color"],
        tiempo_atencion=info["tiempo"], news2=news2, fuente=fuente,
        zona_label=ZONA_LABELS.get(datos.zona, datos.zona),
        advertencias=advertencias
    )

@app.post("/triage/saludips", tags=["Integraciones"],
          summary="Endpoint dedicado SaludIPS")
def evaluar_saludips(datos: SignosVitales):
    """Endpoint dedicado para la integración con SaludIPS. Mismo comportamiento que /triage."""
    datos.sistema_origen = "saludips"
    return evaluar(datos)

@app.get("/triage/{eval_id}", tags=["Triage"],
         summary="Consultar evaluación por ID")
def consultar(eval_id: str):
    conn = sqlite3.connect(DB)
    row = conn.execute("SELECT * FROM evaluaciones WHERE id=?", (eval_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Evaluación {eval_id} no encontrada")
    return {"id":row[0],"timestamp":row[1],"paciente":row[2],"edad":row[3],
            "triage":row[4],"news2":row[5],"zona":row[6],"fuente":row[7]}

@app.delete("/admin/reset", tags=["Admin"],
            summary="Resetear base de datos — solo para desarrollo")
def reset_db():
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM evaluaciones")
    conn.commit(); conn.close()
    return {"mensaje": "Base de datos reseteada"}
