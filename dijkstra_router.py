"""
dijkstra_router.py  -  Enrutador Dijkstra con pesos dinamicos via XGBoost
=========================================================================
Topologia: tree,depth=2,fanout=3
  s1 (root) -> s2, s3, s4
  s2 -> h1(p2), h2(p3), h3(p4)
  s3 -> h4(p2), h5(p3), h6(p4)
  s4 -> h7(p2), h8(p3), h9(p4)
"""

import heapq
import time
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

RYU_URL     = 'http://localhost:8080'
XGB_URL     = 'http://localhost:5001'
ROUTER_PORT = 5002

GRAPH = {
    1: {2:{'bw':100,'delay':5}, 3:{'bw':100,'delay':5}, 4:{'bw':100,'delay':5}},
    2: {1:{'bw':100,'delay':5}},
    3: {1:{'bw':100,'delay':5}},
    4: {1:{'bw':100,'delay':5}},
}

HOST_TO_SWITCH = {
    'h1':2,'h2':2,'h3':2,
    'h4':3,'h5':3,'h6':3,
    'h7':4,'h8':4,'h9':4,
}

HOST_IPS = {
    'h1':'10.0.0.1','h2':'10.0.0.2','h3':'10.0.0.3',
    'h4':'10.0.0.4','h5':'10.0.0.5','h6':'10.0.0.6',
    'h7':'10.0.0.7','h8':'10.0.0.8','h9':'10.0.0.9',
}

NODE_NAMES = {1:'Root-S1', 2:'Leaf-S2', 3:'Leaf-S3', 4:'Leaf-S4'}

PORT_TABLE = {
    (1,2):1,(1,3):2,(1,4):3,
    (2,1):1,(3,1):1,(4,1):1,
}

HOST_PORT = {
    'h1':(2,2),'h2':(2,3),'h3':(2,4),
    'h4':(3,2),'h5':(3,3),'h6':(3,4),
    'h7':(4,2),'h8':(4,3),'h9':(4,4),
}

_last_weight    = 10
_last_traf_type = 1


# ─────────────────────────────────────────────────────────────────────────────
# Algoritmo Dijkstra
# ─────────────────────────────────────────────────────────────────────────────

def dijkstra(src, dst, weight_factor):
    dist = {n: float('inf') for n in GRAPH}
    prev = {n: None for n in GRAPH}
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        cost, u = heapq.heappop(heap)
        if cost > dist[u]:
            continue
        if u == dst:
            break
        for v, edge in GRAPH.get(u, {}).items():
            nc = dist[u] + edge['delay'] * weight_factor
            if nc < dist[v]:
                dist[v] = nc
                prev[v] = u
                heapq.heappush(heap, (nc, v))
    path, node = [], dst
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return (path, dist[dst]) if path and path[0] == src else ([], float('inf'))


def path_bandwidth(path):
    if len(path) < 2:
        return 0
    bw = float('inf')
    for i in range(len(path)-1):
        link_bw = GRAPH.get(path[i],{}).get(path[i+1],{}).get('bw',0)
        bw = min(bw, link_bw)
    return bw if bw != float('inf') else 0


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def get_traffic_type():
    global _last_weight, _last_traf_type
    try:
        r = requests.get(f'{XGB_URL}/predict', timeout=3)
        d = r.json()
        _last_traf_type = d.get('traffic_type', 1)
        _last_weight    = d.get('dijkstra_weight', 10)
    except:
        pass
    return _last_traf_type, _last_weight


# ─────────────────────────────────────────────────────────────────────────────
# Instalacion de flows
# ─────────────────────────────────────────────────────────────────────────────

def install_flow(dpid, priority, match, actions):
    """
    Instala un flow via ofctl_rest.
    Convierte campos OF1.3 a OF1.0 (dl_type/nw_src/nw_dst) que OVS acepta.
    Instala ademas un flow ICMP especifico para que el ping use la ruta.
    """
    m = {}
    for k, v in match.items():
        if k == 'eth_type':   m['dl_type'] = v
        elif k == 'ipv4_src': m['nw_src']  = v
        elif k == 'ipv4_dst': m['nw_dst']  = v
        else: m[k] = v

    ok = False
    try:
        r = requests.post(f'{RYU_URL}/stats/flowentry/add',
                          json={'dpid':dpid,'priority':priority,
                                'match':m,'actions':actions}, timeout=5)
        ok = r.ok

        # Flow ICMP especifico (prioridad +1 para que gane sobre el generico)
        icmp_m = dict(m)
        icmp_m['nw_proto'] = 1
        requests.post(f'{RYU_URL}/stats/flowentry/add',
                      json={'dpid':dpid,'priority':priority+1,
                            'match':icmp_m,'actions':actions}, timeout=5)
    except Exception as e:
        print(f'  Error flow dpid={dpid}: {e}')
    return ok


def install_path(path, src_host, dst_host):
    """
    Instala flows bidireccionales en cada switch del camino.
    """
    src_ip = HOST_IPS[src_host]
    dst_ip = HOST_IPS[dst_host]
    src_sw = HOST_TO_SWITCH[src_host]
    dst_sw = HOST_TO_SWITCH[dst_host]
    installed = []

    # ── Caso especial: mismo switch ──────────────────────────────────────────
    if src_sw == dst_sw:
        sw       = src_sw
        src_port = HOST_PORT[src_host][1]
        dst_port = HOST_PORT[dst_host][1]

        ok = install_flow(sw, 50,
            match={'eth_type':2048,'ipv4_src':src_ip,'ipv4_dst':dst_ip},
            actions=[{'type':'OUTPUT','port':dst_port}])
        install_flow(sw, 50,
            match={'eth_type':2048,'ipv4_src':dst_ip,'ipv4_dst':src_ip},
            actions=[{'type':'OUTPUT','port':src_port}])

        if ok:
            installed.append({'switch':f's{sw}','name':NODE_NAMES[sw],'port':dst_port})
        return installed

    # ── Caso normal: switches distintos ──────────────────────────────────────
    for i, sw in enumerate(path):
        if i == len(path) - 1:
            fwd_port = HOST_PORT[dst_host][1]
        else:
            fwd_port = PORT_TABLE.get((sw, path[i+1]))

        if i == 0:
            rev_port = HOST_PORT[src_host][1]
        else:
            rev_port = PORT_TABLE.get((sw, path[i-1]))

        if fwd_port is None:
            continue

        ok = install_flow(sw, 50,
            match={'eth_type':2048,'ipv4_src':src_ip,'ipv4_dst':dst_ip},
            actions=[{'type':'OUTPUT','port':fwd_port}])

        if rev_port:
            install_flow(sw, 50,
                match={'eth_type':2048,'ipv4_src':dst_ip,'ipv4_dst':src_ip},
                actions=[{'type':'OUTPUT','port':rev_port}])

        if ok:
            installed.append({'switch':f's{sw}','name':NODE_NAMES[sw],'port':fwd_port})

    return installed


def install_flood_base():
    for dpid in GRAPH:
        try:
            requests.post(f'{RYU_URL}/stats/flowentry/add',
                json={'dpid':dpid,'priority':0,'match':{},
                      'actions':[{'type':'OUTPUT','port':'FLOOD'}]},
                timeout=5)
        except:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)


@app.post('/route')
def api_route():
    body = request.get_json()
    src  = body.get('src')
    dst  = body.get('dst')

    if src not in HOST_TO_SWITCH or dst not in HOST_TO_SWITCH:
        return jsonify({'error':f'Hosts validos: {list(HOST_TO_SWITCH.keys())}'}), 400

    if src == dst:
        return jsonify({'error':'Origen y destino deben ser distintos'}), 400

    src_sw = HOST_TO_SWITCH[src]
    dst_sw = HOST_TO_SWITCH[dst]

    traf_type, weight = get_traffic_type()

    if src_sw == dst_sw:
        path = [src_sw]
        cost = 0
        bw   = 100
    else:
        path, cost = dijkstra(src_sw, dst_sw, weight_factor=weight)
        if not path:
            return jsonify({'error':f'Sin ruta entre {src} y {dst}'}), 404
        bw = path_bandwidth(path)

    flows = install_path(path, src, dst)

    return jsonify({
        'src':             src,
        'dst':             dst,
        'path':            [{'switch':f's{sw}','name':NODE_NAMES[sw]} for sw in path],
        'flows_installed': flows,
        'traffic_type':    traf_type,
        'traffic_desc':    {1:'Normal',2:'Alto',3:'Critico'}[traf_type],
        'dijkstra_weight': weight,
        'bandwidth_mbps':  bw,
        'cost':            cost,
        'hops':            len(path)-1,
        'same_switch':     src_sw == dst_sw,
    })


@app.get('/topology')
def api_topology():
    traf_type, weight = get_traffic_type()
    edges, seen = [], set()
    for u, neighbors in GRAPH.items():
        for v, data in neighbors.items():
            key = tuple(sorted([u,v]))
            if key not in seen:
                seen.add(key)
                edges.append({'from':u,'to':v,
                    'from_name':NODE_NAMES[u],'to_name':NODE_NAMES[v],
                    'bw_mbps':data['bw'],'delay_ms':data['delay'],
                    'weight':data['delay']*weight})
    nodes = [{'id':k,'name':v,
              'hosts':[h for h,sw in HOST_TO_SWITCH.items() if sw==k]}
             for k,v in NODE_NAMES.items()]
    return jsonify({'nodes':nodes,'edges':edges,
                    'traffic_type':traf_type,'current_weight':weight})


@app.get('/bandwidth/<src>/<dst>')
def api_bandwidth(src, dst):
    if src not in HOST_TO_SWITCH or dst not in HOST_TO_SWITCH:
        return jsonify({'error':'Host invalido'}), 400
    traf_type, weight = get_traffic_type()
    src_sw = HOST_TO_SWITCH[src]
    dst_sw = HOST_TO_SWITCH[dst]
    if src_sw == dst_sw:
        return jsonify({'src':src,'dst':dst,'bandwidth_mbps':100,
                        'path':[NODE_NAMES[src_sw]],'hops':0,'cost':0,
                        'note':'Mismo switch - enlace directo'})
    path, cost = dijkstra(src_sw, dst_sw, weight)
    return jsonify({'src':src,'dst':dst,
                    'bandwidth_mbps':path_bandwidth(path),
                    'path':[NODE_NAMES[sw] for sw in path],
                    'hops':len(path)-1,'cost':cost,'weight_used':weight})


@app.post('/multicast/block')
def multicast_block():
    """
    Controla que puertos del switch leaf reciben el stream multicast 239.1.1.1.

    Match simplificado: solo dl_type=2048 + nw_dst=239.1.1.1
    (sin nw_proto ni in_port para que OVS lo capture con certeza)

    Logica:
      1. Borra TODAS las reglas priority=200 en el switch (delete_strict=False).
      2. Espera 1s para que OVS limpie la fast-path cache.
      3. Si hay puertos activos: instala regla que reenvía solo a esos puertos.
         Si no hay puertos: no instala nada -> el FLOOD (pri=0) sigue activo
         pero la ausencia de regla multicast especifica NO bloquea por si sola.
         Por eso cuando active_ports esta vacio instalamos una regla DROP.
    """
    body = request.get_json()
    if not body:
        return jsonify({'ok': False, 'error': 'Body JSON requerido'}), 400

    try:
        dpid         = int(body['dpid'])
        active_ports = [int(p) for p in body.get('active_ports', [])]
    except (KeyError, ValueError) as e:
        return jsonify({'ok': False, 'error': f'Parametro invalido: {e}'}), 400

    MATCH = {'dl_type': 2048, 'nw_dst': '239.1.1.1'}

    # ── Paso 1: borrar regla multicast existente en este switch ───────────────
    try:
        requests.post(
            f'{RYU_URL}/stats/flowentry/delete',
            json={'dpid': dpid, 'priority': 200, 'match': MATCH},
            timeout=5,
        )
        print(f'  [multicast] Regla priority=200 eliminada en DPID {dpid}')
    except Exception as e:
        print(f'  [multicast] Advertencia al borrar en DPID {dpid}: {e}')

    # ── Paso 2: esperar a que OVS invalide fast-path cache ────────────────────
    time.sleep(1.0)

    # ── Paso 3: instalar nueva regla ──────────────────────────────────────────
    if active_ports:
        # Reenviar solo a los puertos activos
        actions = [{'type': 'OUTPUT', 'port': p} for p in active_ports]
        log_msg = f'puertos activos {active_ports}'
    else:
        # Sin receptores: DROP (lista de acciones vacia = drop en OpenFlow)
        actions = []
        log_msg = 'DROP (sin receptores activos)'

    try:
        r = requests.post(
            f'{RYU_URL}/stats/flowentry/add',
            json={'dpid': dpid, 'priority': 200, 'match': MATCH, 'actions': actions},
            timeout=5,
        )
        if not r.ok:
            return jsonify({'ok': False, 'error': f'Ryu rechazo el flow: HTTP {r.status_code}'}), 500
        print(f'  [multicast] DPID {dpid} -> {log_msg}')
    except Exception as e:
        print(f'  [multicast] Error al instalar en DPID {dpid}: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500

    return jsonify({'ok': True, 'dpid': dpid, 'active_ports': active_ports})

@app.get('/health')
def api_health():
    try:    xgb_ok = requests.get(f'{XGB_URL}/health', timeout=2).ok
    except: xgb_ok = False
    try:    ryu_ok = requests.get(f'{RYU_URL}/stats/switches', timeout=2).ok
    except: ryu_ok = False
    return jsonify({'dijkstra_router':'ok',
                    'xgb_model':'ok' if xgb_ok else 'unreachable',
                    'ryu_ofctl':'ok' if ryu_ok else 'unreachable'})


if __name__ == '__main__':
    print('='*50)
    print('  Dijkstra Router - tree,depth=2,fanout=3')
    print('='*50)
    print(f'  ofctl_rest: {RYU_URL}')
    print(f'  XGBoost:    {XGB_URL}')
    print(f'  Puerto:     {ROUTER_PORT}')
    print('='*50)
    print('\nInstalando flows FLOOD base...')
    install_flood_base()
    print('Listo.\n')
    app.run(host='0.0.0.0', port=ROUTER_PORT, debug=False)
