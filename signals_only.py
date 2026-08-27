"""
=============================================================================
 SIGNALS ONLY - Feed de señales puro (corre EN PARALELO al paper trader)
=============================================================================
 NO simula cuenta, NO abre posiciones, NO tiene topes de cartera. Solo
 detecta y avisa TODAS las señales que la estrategia genera (3/4 y 4/4),
 incluidas las que el paper trader ignora por falta de cupo de riesgo.

 Cada corrida:
   1. Analiza el S&P 500 (regimen + su propio score) -> el termometro.
   2. Escanea los 22 tickers con la MISMA regla del screener.
   3. Avisa por Telegram lo que CAMBIO (señales nuevas y desaparecidas)
      + la lista vigente completa como referencia.

 Usa un state propio y pequeño (signals_state.json) solo para recordar
 que señales habia ayer y no repetirlas. Independiente del state.json
 del paper trader. La regla de entrada es la misma de screener.py.
=============================================================================
"""

import os
import json
from datetime import datetime, timezone

import yfinance as yf
from screener import enrich, evaluate, CONFIG


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

SIG_CONFIG = {
    **CONFIG,
    "regime_ticker": "^GSPC",
    "lookback": "1y",
    # True  = avisa solo lo que cambio (nuevas/desaparecidas) + lista vigente.
    # False = manda todas las señales vigentes cada dia (mas ruidoso).
    "only_new": True,
}

STATE_FILE = "signals_state.json"


# =============================================================================
# ESTADO (minimo: solo recuerda las señales de ayer)
# =============================================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "active_signals": {}}   # ticker -> score


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# =============================================================================
# ANÁLISIS DEL S&P 500 (el termometro)
# =============================================================================

def analyze_sp500(cfg, df):
    """Regimen + distancia a EMA200 + score propio del indice."""
    if df is None or df.empty:
        return None
    d = enrich(df.copy(), cfg)
    last = d.iloc[-1]
    ema200 = float(last["ema_trend"])
    price = float(last["Close"])
    dist_pct = (price - ema200) / ema200 * 100
    res = evaluate(d, cfg)
    return {
        "price": round(price, 2),
        "ema200": round(ema200, 2),
        "dist_pct": round(dist_pct, 1),
        "bullish": price > ema200,
        "score": res["score"] if res else 0,
        "is_signal": bool(res and res["is_signal"]),
    }


# =============================================================================
# ESCANEO DE SEÑALES
# =============================================================================

def scan(cfg, cache):
    """Devuelve {ticker: {score, entry, sl, tp}} de todo lo que da señal."""
    signals = {}
    for t in cfg["tickers"]:
        df = cache.get(t)
        if df is None:
            continue
        d = enrich(df.copy(), cfg)
        res = evaluate(d, cfg)
        if res and res["is_signal"]:
            signals[t] = {
                "score": res["score"],
                "entry": res["entry"], "sl": res["sl"], "tp": res["tp"],
            }
    return signals


def download_all(cfg):
    cache = {}
    tickers = list(dict.fromkeys(cfg["tickers"] + [cfg["regime_ticker"]]))
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=cfg["lookback"], interval=cfg["interval"])
            cache[t] = df if not df.empty else None
        except Exception:
            cache[t] = None
    return cache


# =============================================================================
# CICLO PRINCIPAL
# =============================================================================

def run(cfg=SIG_CONFIG):
    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_run") == today:
        print(f"Ya se corrio hoy ({today}). Saltando.")
        return state

    cache = download_all(cfg)
    sp = analyze_sp500(cfg, cache.get(cfg["regime_ticker"]))
    current = scan(cfg, cache)                 # {ticker: {...}}
    previous = state.get("active_signals", {})  # {ticker: score}

    # Cambios respecto a ayer
    new = {t: v for t, v in current.items() if t not in previous}
    dropped = [t for t in previous if t not in current]
    upgraded = {t: v for t, v in current.items()
                if t in previous and v["score"] > previous.get(t, 0)}

    summary = build_summary(cfg, today, sp, current, new, dropped, upgraded)
    print(summary)
    send_telegram(summary)

    # Guarda el estado nuevo (solo ticker -> score)
    state["active_signals"] = {t: v["score"] for t, v in current.items()}
    state["last_run"] = today
    save_state(state)
    return state


# =============================================================================
# RESUMEN
# =============================================================================

def fmt_sig(t, v):
    return f"  {t}: {v['score']}/4 | entrada {v['entry']} | SL {v['sl']} | TP {v['tp']}"


def build_summary(cfg, today, sp, current, new, dropped, upgraded):
    lines = [f"SEÑALES (paralelo) | {today}"]

    # --- Termometro S&P 500 ---
    if sp:
        estado = "ALCISTA ✓" if sp["bullish"] else "BAJISTA ✗"
        lines.append(
            f"S&P500: {sp['price']} | {estado} | "
            f"{'+' if sp['dist_pct']>=0 else ''}{sp['dist_pct']}% vs EMA200 | "
            f"score {sp['score']}/4")
        if not sp["bullish"]:
            lines.append("  (Regimen bajista: la estrategia larga estaria dormida)")
    else:
        lines.append("S&P500: sin datos hoy")

    n4 = sum(1 for v in current.values() if v["score"] == 4)
    n3 = sum(1 for v in current.values() if v["score"] == 3)
    lines.append(f"Señales vigentes: {len(current)} ({n4} de 4/4, {n3} de 3/4)")

    # --- Cambios ---
    if new:
        lines.append("--- NUEVAS hoy ---")
        for t, v in sorted(new.items(), key=lambda x: -x[1]["score"]):
            lines.append(fmt_sig(t, v))
    if upgraded:
        lines.append("--- SUBIERON a 4/4 ---")
        for t, v in upgraded.items():
            lines.append(fmt_sig(t, v))
    if dropped:
        lines.append("--- DESAPARECIERON ---")
        lines.append("  " + ", ".join(dropped))
    if not (new or upgraded or dropped):
        lines.append("Sin cambios respecto a ayer.")

    # --- Lista vigente completa (referencia) ---
    show_all = (not cfg["only_new"]) or new or upgraded
    if current and show_all:
        lines.append("--- Todas las vigentes ---")
        for t, v in sorted(current.items(), key=lambda x: -x[1]["score"]):
            lines.append(fmt_sig(t, v))

    return "\n".join(lines)


def send_telegram(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("(Telegram no configurado; omitiendo alerta)")
        return
    try:
        import urllib.request, urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15)
        print("(Alerta enviada a Telegram)")
    except Exception as e:
        print(f"(Fallo al enviar Telegram: {e})")


if __name__ == "__main__":
    run()
