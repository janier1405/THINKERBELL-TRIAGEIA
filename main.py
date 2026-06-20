
"""
TriageIA API v2.0 — Punto de entrada principal
Desplegado en Railway · https://triageia.up.railway.app
"""
import os
from api_triage import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
