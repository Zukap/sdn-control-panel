"""
proxy.py  -  Proxy Flask que resuelve CORS y enruta peticiones
==============================================================
Rutas:
  /stats/*          -> Ryu ofctl_rest :8080
  /route            -> dijkstra_router :5002
  /topology         -> dijkstra_router :5002
  /bandwidth/*      -> dijkstra_router :5002
  /health           -> dijkstra_router :5002
  /multicast/block  -> dijkstra_router :5002
  /predict          -> collector_xgb   :5001
"""

from flask import Flask, request, Response
import requests

app = Flask(__name__)

RYU    = "http://localhost:8080"
ROUTER = "http://localhost:5002"
XGB    = "http://localhost:5001"

CORS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, ngrok-skip-browser-warning',
}

def _fwd(url, method='GET', data=None):
    try:
        r = requests.request(
            method=method, url=url,
            headers={'Content-Type': 'application/json'},
            data=data, timeout=8,
        )
        return Response(r.content, status=r.status_code,
                        headers={'Content-Type': 'application/json', **CORS})
    except Exception as e:
        return Response(f'{{"error":"no disponible: {e}"}}', status=503,
                        headers={'Content-Type': 'application/json', **CORS})

@app.after_request
def add_cors(response):
    for k, v in CORS.items():
        response.headers[k] = v
    return response

@app.route('/',               defaults={'path': ''}, methods=['GET','POST','DELETE','OPTIONS'])
@app.route('/<path:path>',                           methods=['GET','POST','DELETE','OPTIONS'])
def proxy(path=''):
    if request.method == 'OPTIONS':
        return Response(status=200, headers=CORS)

    # ── Ryu ofctl_rest ────────────────────────────────────────────────────────
    if path.startswith('stats/'):
        return _fwd(f"{RYU}/{path}", request.method, request.get_data())

    # ── dijkstra_router ───────────────────────────────────────────────────────
    if path == 'route':
        return _fwd(f"{ROUTER}/route", 'POST', request.get_data())

    if path == 'topology':
        return _fwd(f"{ROUTER}/topology")

    if path.startswith('bandwidth/'):
        return _fwd(f"{ROUTER}/{path}")

    if path == 'health':
        return _fwd(f"{ROUTER}/health")

    if path == 'multicast/block':
        return _fwd(f"{ROUTER}/multicast/block", 'POST', request.get_data())

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if path == 'predict':
        return _fwd(f"{XGB}/predict")

    # ── Not found ─────────────────────────────────────────────────────────────
    return Response(
        f'{{"error":"ruta no encontrada: /{path}"}}',
        status=404,
        headers={'Content-Type': 'application/json', **CORS},
    )

if __name__ == '__main__':
    print('=' * 50)
    print('  Proxy Flask')
    print('=' * 50)
    print(f'  Ryu    -> {RYU}')
    print(f'  Router -> {ROUTER}')
    print(f'  XGB    -> {XGB}')
    print('=' * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
