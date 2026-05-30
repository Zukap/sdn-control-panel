#!/bin/bash
# recolectar.sh  -  Automatiza la recoleccion de datos para XGBoost
# =================================================================
# Ejecutar en VM Ryu: chmod +x recolectar.sh && ./recolectar.sh

COLLECTOR="python3 collector_xgb.py"
DURACION=90

echo "========================================================"
echo "  RECOLECCION DE DATOS PARA MODELO XGBoost"
echo "========================================================"
echo ""

recolectar_tipo() {
    local label=$1
    local nombre=$2
    local instrucciones=$3

    echo "[ TIPO $label - $nombre ]"
    echo ""
    echo "  Ve a la VM Mininet y ejecuta:"
    echo ""
    echo "$instrucciones"
    echo ""
    echo "  Tienes 60 segundos para hacerlo..."
    echo ""
    for i in $(seq 60 -1 1); do
        printf "\r  Iniciando en %2d segundos..." $i
        sleep 1
    done
    echo ""
    echo ""
    echo "  Recolectando tipo $label durante ${DURACION}s..."

    # Lanzar colector en background y guardar PID
    $COLLECTOR --mode collect --label $label &
    COLLECTOR_PID=$!

    # Esperar DURACION segundos y luego matar el proceso limpiamente
    sleep $DURACION
    kill -TERM $COLLECTOR_PID 2>/dev/null
    wait $COLLECTOR_PID 2>/dev/null

    echo ""
    echo "  Tipo $label completado."
    echo ""
}

# ── TIPO 1 ────────────────────────────────────────────────────────
recolectar_tipo 1 "TRAFICO NORMAL" \
"    mininet> h1 ping -c 1000 10.0.0.4 &
    mininet> h2 ping -c 1000 10.0.0.7 &
    mininet> h3 ping -c 1000 10.0.0.9 &"

# ── TIPO 2 ────────────────────────────────────────────────────────
recolectar_tipo 2 "TRAFICO ALTO" \
"    mininet> h4 iperf -s &
    mininet> h7 iperf -s &
    mininet> h1 iperf -c 10.0.0.4 -t 200 &
    mininet> h2 iperf -c 10.0.0.7 -t 200 &"

# ── TIPO 3 ────────────────────────────────────────────────────────
recolectar_tipo 3 "TRAFICO CRITICO" \
"    mininet> h4 iperf -s &
    mininet> h7 iperf -s &
    mininet> h9 iperf -s &
    mininet> h1 iperf -c 10.0.0.4 -t 200 -P 4 &
    mininet> h2 iperf -c 10.0.0.7 -t 200 -P 4 &
    mininet> h3 iperf -c 10.0.0.9 -t 200 -P 4 &"

# ── Entrenar ──────────────────────────────────────────────────────
echo "========================================================"
echo "  ENTRENANDO MODELO XGBoost..."
echo "========================================================"
echo ""
$COLLECTOR --mode train
echo ""

# ── Verificar ─────────────────────────────────────────────────────
echo "========================================================"
if [ -f "xgb_model.pkl" ]; then
    echo "  LISTO - xgb_model.pkl creado correctamente."
    echo ""
    echo "  Siguiente paso:"
    echo "    Terminal 1: python3 collector_xgb.py --mode serve"
    echo "    Terminal 2: python3 dijkstra_router.py"
    echo "    Terminal 3: python3 proxy.py"
else
    echo "  ERROR: no se genero xgb_model.pkl"
    echo "  Verifica el dataset: wc -l dataset.csv"
fi
echo "========================================================"
