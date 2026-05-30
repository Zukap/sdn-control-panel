"""
collector_xgb.py  -  Recolector de metricas y modelo XGBoost
=============================================================
Topologia: tree,depth=2,fanout=3
  s1 (root) -> s2, s3, s4
  s2 -> h1, h2, h3
  s3 -> h4, h5, h6
  s4 -> h7, h8, h9

Modos:
  collect --label 1   trafico normal  (ping entre hosts)
  collect --label 2   trafico alto    (iperf TCP)
  collect --label 3   trafico critico (multiples flujos)
  train               entrenar modelo XGBoost
  serve               servir /predict en puerto 5001

Ejecutar en VM Ryu (mientras Mininet genera trafico):
  python3 collector_xgb.py --mode collect --label 1
  python3 collector_xgb.py --mode train
  python3 collector_xgb.py --mode serve
"""

import argparse
import csv
import json
import os
import time
import pickle
import requests
import signal
signal.signal(signal.SIGTERM, lambda s,f: exit(0))
import numpy as np
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS

RYU_URL    = 'http://localhost:8080'
DPIDS      = [1, 2, 3, 4]
CSV_FILE   = 'dataset.csv'
MODEL_FILE = 'xgb_model.pkl'
INTERVAL   = 5

CSV_HEADERS = [
    'timestamp', 'dpid',
    'bytes_per_sec', 'packets_per_sec',
    'duration_avg', 'flow_count',
    'byte_total', 'packet_total',
    'label'
]

def get_flows(dpid):
    try:
        r = requests.get(f'{RYU_URL}/stats/flow/{dpid}', timeout=5)
        return r.json().get(str(dpid), [])
    except:
        return []

def extract_features(flows, prev_flows, elapsed):
    if not flows:
        return dict(bytes_per_sec=0, packets_per_sec=0, duration_avg=0,
                    flow_count=0, byte_total=0, packet_total=0)
    prev = {json.dumps(f.get('match',{}), sort_keys=True): f for f in prev_flows}
    byte_total   = sum(f.get('byte_count',   0) for f in flows)
    packet_total = sum(f.get('packet_count', 0) for f in flows)
    dur_avg      = sum(f.get('duration_sec', 0) for f in flows) / len(flows)
    delta_b = delta_p = 0
    for f in flows:
        key = json.dumps(f.get('match', {}), sort_keys=True)
        pb  = prev[key].get('byte_count',   0) if key in prev else 0
        pp  = prev[key].get('packet_count', 0) if key in prev else 0
        delta_b += f.get('byte_count',   0) - pb
        delta_p += f.get('packet_count', 0) - pp
    t = elapsed if elapsed > 0 else 1
    return dict(
        bytes_per_sec   = max(0, delta_b / t),
        packets_per_sec = max(0, delta_p / t),
        duration_avg    = dur_avg,
        flow_count      = len(flows),
        byte_total      = byte_total,
        packet_total    = packet_total,
    )

def collect(label):
    exists = os.path.exists(CSV_FILE)
    f = open(CSV_FILE, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
    if not exists:
        writer.writeheader()
    print(f'Recolectando trafico tipo {label}. Ctrl+C para detener.')
    prev = {d: [] for d in DPIDS}
    n = 0
    try:
        while True:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for dpid in DPIDS:
                flows    = get_flows(dpid)
                features = extract_features(flows, prev[dpid], INTERVAL)
                prev[dpid] = flows
                writer.writerow({'timestamp': ts, 'dpid': dpid,
                                 'label': label, **features})
            n += 1
            f.flush()
            print(f'  Muestra {n} | {ts} | label={label}')
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print(f'\nDetenido. {n} muestras guardadas en {CSV_FILE}')
        f.close()

def train():
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
    except ImportError as e:
        print(f'Error: {e}')
        return
    if not os.path.exists(CSV_FILE):
        print(f'No existe {CSV_FILE}. Ejecuta --mode collect primero.')
        return
    rows = list(csv.DictReader(open(CSV_FILE)))
    print(f'Dataset: {len(rows)} muestras')
    if len(rows) < 30:
        print('Dataset muy pequeno. Recolecta al menos 30 muestras por tipo.')
        return
    FEATURES = ['bytes_per_sec','packets_per_sec','duration_avg',
                'flow_count','byte_total','packet_total']
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    y = np.array([int(r['label']) for r in rows]) - 1
    for cls in np.unique(y):
        print(f'  Tipo {cls+1}: {(y==cls).sum()} muestras')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    print('Entrenando XGBoost...')
    clf = xgb.XGBClassifier(
        n_estimators  = 100,
        max_depth     = 6,
        learning_rate = 0.1,
        eval_metric   = 'mlogloss',
        random_state  = 42,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = clf.predict(X_te)
    print('\nResultados:')
    print(classification_report(y_te, y_pred,
          target_names=['Tipo1-Normal','Tipo2-Alto','Tipo3-Critico']))
    print('Importancia de features:')
    for feat, imp in sorted(zip(FEATURES, clf.feature_importances_),
                            key=lambda x: x[1], reverse=True):
        bar = '█' * int(imp * 30)
        print(f'  {feat:<20} {bar} {imp:.3f}')
    pickle.dump({'model': clf, 'features': FEATURES}, open(MODEL_FILE, 'wb'))
    print(f'\nModelo guardado en {MODEL_FILE}')

flask_app = Flask(__name__)
CORS(flask_app)
_cache = None

def load_model():
    global _cache
    if _cache is None and os.path.exists(MODEL_FILE):
        _cache = pickle.load(open(MODEL_FILE, 'rb'))
    return (_cache['model'], _cache['features']) if _cache else (None, None)

def predict_now():
    clf, features = load_model()
    if clf is None:
        return None
    agg = {f: 0.0 for f in features}
    for dpid in DPIDS:
        flows = get_flows(dpid)
        feat  = extract_features(flows, [], INTERVAL)
        for k in features:
            agg[k] += feat.get(k, 0)
    agg['duration_avg'] /= len(DPIDS)
    agg['flow_count']   /= len(DPIDS)
    X    = np.array([[agg[f] for f in features]])
    pred = int(clf.predict(X)[0]) + 1
    prob = clf.predict_proba(X)[0]
    pesos = {1: 10, 2: 40, 3: 80}
    return {
        'traffic_type':    pred,
        'dijkstra_weight': pesos[pred],
        'confidence':      float(max(prob)),
        'probabilities':   {str(i+1): float(p) for i, p in enumerate(prob)},
        'features':        agg,
        'description': {
            1: 'Normal - trafico bajo',
            2: 'Alto   - transferencias masivas',
            3: 'Critico - congestion detectada'
        }[pred]
    }

@flask_app.get('/predict')
def api_predict():
    result = predict_now()
    if result is None:
        return jsonify({'error': 'Modelo no cargado. Ejecuta --mode train'}), 503
    return jsonify(result)

@flask_app.get('/health')
def api_health():
    clf, _ = load_model()
    return jsonify({
        'model_loaded':   clf is not None,
        'dataset_exists': os.path.exists(CSV_FILE),
        'switches':       DPIDS,
    })

def serve():
    clf, _ = load_model()
    print(f'Servidor XGBoost en http://0.0.0.0:5001')
    print(f'Modelo: {"OK" if clf else "SIN MODELO - entrena primero"}')
    flask_app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',  choices=['collect','train','serve'], required=True)
    parser.add_argument('--label', type=int, choices=[1,2,3])
    parser.add_argument('--ryu',   default='http://localhost:8080')
    args = parser.parse_args()
    RYU_URL = args.ryu
    if args.mode == 'collect':
        if not args.label:
            parser.error('--mode collect requiere --label')
        collect(args.label)
    elif args.mode == 'train':
        train()
    elif args.mode == 'serve':
        serve()
