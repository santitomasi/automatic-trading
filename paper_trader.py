"""
=============================================================================
 PAPER TRADER v3 - Riesgo dinamico por conviccion + presupuesto completo
=============================================================================
 Mejoras sobre la v2 (a partir de 3 semanas de datos en vivo):

   1. RIESGO DINAMICO POR SCORE: señal 4/4 -> 2% de riesgo; señal 3/4 -> 1%.
      La "conviccion" es una REGLA (el score de confluencia), no una
      sensacion. El CSV registra el score de cada operacion para poder
      comparar despues si las 4/4 realmente rinden mas que las 3/4.
   2. DIMENSIONADO AL PRESUPUESTO RESTANTE: si el cupo de riesgo libre es
      menor que el riesgo deseado, la posicion se abre mas chica para
      llenar el presupuesto exacto (minimo 0.5%). Se acabo el "5.1% y
      bloqueado": el 6% se usa completo sin excederse nunca.
   3. ENFRIAMIENTO TRAS STOP: despues de un stop loss en un ticker, no se
      reentra en ese ticker por 5 dias habiles (evita la puerta giratoria
      que vimos con AAPL el 31-jul).
   4. REPORTE ENRIQUECIDO: cada posicion muestra precio actual, PnL no
      realizado en $ y progreso en R (+1.0R = a mitad de camino al TP con
      ratio 1:2; -1.0R = tocando el stop).

 Compatible con el state.json existente (migracion automatica).
 La regla de ENTRADA sigue intacta (screener.py). Sin cambios en SL/TP.
=============================================================================
"""

import os
import json
import csv
from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from screener import enrich, evaluate, CONFIG


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

PT_CONFIG = {
    **CONFIG,
    "start_equity": 10000.0,
    "cost_pct": 0.05,
    "lookback": "1y",

    # --- Riesgo dinamico por conviccion (nuevo) ------------------------------
    # El score de confluencia decide cuanto arriesgar. Regla, no sensacion.
    "risk_by_score": {3: 0.01, 4: 0.02},   # 3/4 -> 1% | 4/4 -> 2%
    "min_position_risk": 0.005,            # no abrir posiciones de menos de 0.5%

    # --- Gestion de cartera --------------------------------------------------
    "max_total_risk": 0.06,
    "max_per_sector": 2,
    "regime_ticker": "^GSPC",
    "time_stop_bars": 40,
    "cooldown_bars": 5,        # dias habiles sin reentrar tras un stop loss
}

SECTORS = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "AMZN": "tech",
    "JPM": "financiero", "BAC": "financiero", "V": "financiero",
    "JNJ": "salud", "UNH": "salud", "PFE": "salud",
    "XOM": "energia", "CVX": "energia",
    "PG": "consumo", "KO": "consumo", "WMT": "consumo",
    "CAT": "industrial", "BA": "industrial",
    "^GSPC": "indices", "^NDX": "indices", "^DJI": "indices", "^RUT": "indices",
}

STATE_FILE = "state.json"
TRADES_CSV = "trades_paper.csv"
CSV_FIELDS = ["ticker", "entry_date", "exit_date", "entry", "exit",
              "shares", "outcome", "pnl", "equity_after", "score", "risk_pct"]


# =============================================================================
# ESTADO PERSISTENTE (migracion automatica v1/v2 -> v3)
# =============================================================================

def load_state(cfg):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        state.setdefault("equity_history", [])
        state.setdefault("cooldowns", {})       # ticker -> fecha del ultimo SL
        return state
    return {
        "equity": cfg["start_equity"],
        "start_equity": cfg["start_equity"],
        "peak_equity": cfg["start_equity"],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "last_run": None,
        "open_positions": {},
        "closed_trades": [],
        "equity_history": [],
        "cooldowns": {},
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_csv(trade):
    exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(trade)


# =============================================================================
# GESTIÓN DE RIESGO
# =============================================================================

def current_total_risk(state):
    total = sum(p.get("risk_amount", 0) for p in state["open_positions"].values())
    return total / state["equity"] if state["equity"] > 0 else 1.0


def sector_count(state, sector):
    return sum(1 for t in state["open_positions"]
               if SECTORS.get(t, "otros") == sector)


def regime_is_bullish(cfg, cache):
    df = cache.get(cfg["regime_ticker"])
    if df is None:
        return True
    ema200 = df["Close"].ewm(span=cfg["ema_trend"], adjust=False).mean()
    return bool(df["Close"].iloc[-1] > ema200.iloc[-1])


def bars_held(entry_date_str, today_str):
    try:
        return int(np.busday_count(entry_date_str, today_str))
    except Exception:
        return 0


def in_cooldown(state, ticker, today, cfg):
    """True si el ticker tuvo un stop loss hace menos de cooldown_bars dias."""
    last_sl = state.get("cooldowns", {}).get(ticker)
    if not last_sl:
        return False
    return bars_held(last_sl, today) < cfg["cooldown_bars"]


def desired_risk_fraction(score, cfg):
    """Riesgo segun conviccion (score). Scores no mapeados usan el minimo."""
    return cfg["risk_by_score"].get(score, min(cfg["risk_by_score"].values()))


# =============================================================================
# CUENTA SIMULADA
# =============================================================================

def open_position(state, ticker, sig, cfg, today):
    """
    Abre posicion con riesgo dinamico por score, recortado al presupuesto
    restante del tope global. Si el hueco libre es < min_position_risk,
    no abre (devuelve None).
    """
    equity = state["equity"]
    wanted = desired_risk_fraction(sig["score"], cfg)
    headroom = cfg["max_total_risk"] - current_total_risk(state)
    risk_frac = min(wanted, headroom)
    if risk_frac < cfg["min_position_risk"]:
        return None

    risk_amount = equity * risk_frac
    per_share_risk = sig["entry"] - sig["sl"]
    if per_share_risk <= 0:
        return None
    shares = risk_amount / per_share_risk

    state["open_positions"][ticker] = {
        "entry": sig["entry"], "sl": sig["sl"], "tp": sig["tp"],
        "shares": round(shares, 4), "risk_amount": round(risk_amount, 2),
        "risk_pct": round(risk_frac * 100, 2),
        "score": sig["score"],
        "entry_date": today,
    }
    return state["open_positions"][ticker]


def close_position(state, ticker, exit_price, outcome, cfg, today):
    pos = state["open_positions"].pop(ticker)
    cost = cfg["cost_pct"] / 100.0
    eff_entry = pos["entry"] * (1 + cost)
    eff_exit  = exit_price * (1 - cost)
    pnl = pos["shares"] * (eff_exit - eff_entry)
    state["equity"] += pnl
    state["peak_equity"] = max(state["peak_equity"], state["equity"])

    if outcome.startswith("SL"):
        state.setdefault("cooldowns", {})[ticker] = today   # activa enfriamiento

    trade = {
        "ticker": ticker,
        "entry_date": pos["entry_date"], "exit_date": today,
        "entry": round(pos["entry"], 2), "exit": round(exit_price, 2),
        "shares": pos["shares"], "outcome": outcome,
        "pnl": round(pnl, 2), "equity_after": round(state["equity"], 2),
        "score": pos.get("score", ""), "risk_pct": pos.get("risk_pct", ""),
    }
    state["closed_trades"].append(trade)
    append_csv(trade)
    return trade


def position_status(pos, current_price):
    """PnL no realizado en $ y progreso en multiplos de R."""
    pnl = pos["shares"] * (current_price - pos["entry"])
    per_share_risk = pos["entry"] - pos["sl"]
    r_mult = (current_price - pos["entry"]) / per_share_risk if per_share_risk > 0 else 0
    return round(pnl, 2), round(r_mult, 2)


def mark_to_market(state, cache):
    mtm = state["equity"]
    for t, p in state["open_positions"].items():
        df = cache.get(t)
        if df is not None and not df.empty:
            mtm += p["shares"] * (float(df["Close"].iloc[-1]) - p["entry"])
    return round(mtm, 2)


# =============================================================================
# CICLO PRINCIPAL
# =============================================================================

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


def run(cfg=PT_CONFIG):
    state = load_state(cfg)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if state.get("last_run") == today:
        print(f"Ya se corrio hoy ({today}). Saltando.")
        return state

    events = []
    cache = download_all(cfg)
    bullish = regime_is_bullish(cfg, cache)

    # --- 1. SALIDAS (siempre) -----------------------------------------------
    for ticker in list(state["open_positions"].keys()):
        df = cache.get(ticker)
        if df is None:
            events.append(f"[!] {ticker}: sin datos hoy, posicion sigue abierta")
            continue
        last = df.iloc[-1]
        pos = state["open_positions"][ticker]

        hit_sl = last["Low"] <= pos["sl"]
        hit_tp = last["High"] >= pos["tp"]
        expired = bars_held(pos["entry_date"], today) >= cfg["time_stop_bars"]

        if hit_sl and hit_tp:
            t = close_position(state, ticker, pos["sl"], "SL (ambiguo)", cfg, today)
            events.append(f"CIERRE {ticker}: SL ambiguo, PnL ${t['pnl']}")
        elif hit_sl:
            t = close_position(state, ticker, pos["sl"], "SL", cfg, today)
            events.append(f"CIERRE {ticker}: STOP LOSS, PnL ${t['pnl']} (enfriamiento {cfg['cooldown_bars']}d)")
        elif hit_tp:
            t = close_position(state, ticker, pos["tp"], "TP", cfg, today)
            events.append(f"CIERRE {ticker}: TAKE PROFIT, PnL ${t['pnl']}")
        elif expired:
            t = close_position(state, ticker, float(last["Close"]), "TIEMPO", cfg, today)
            events.append(f"CIERRE {ticker}: STOP TEMPORAL ({cfg['time_stop_bars']}d), PnL ${t['pnl']}")

    # --- 2. ENTRADAS (solo en regimen alcista) ------------------------------
    if not bullish:
        events.append("REGIMEN BAJISTA (S&P bajo EMA200): no se abren posiciones nuevas.")
    else:
        for ticker in cfg["tickers"]:
            if ticker in state["open_positions"]:
                continue
            if in_cooldown(state, ticker, today, cfg):
                continue
            headroom = cfg["max_total_risk"] - current_total_risk(state)
            if headroom < cfg["min_position_risk"]:
                events.append(f"PRESUPUESTO DE RIESGO agotado "
                              f"({current_total_risk(state)*100:.1f}% de {cfg['max_total_risk']*100:.0f}%).")
                break
            sector = SECTORS.get(ticker, "otros")
            if sector_count(state, sector) >= cfg["max_per_sector"]:
                continue
            df = cache.get(ticker)
            if df is None:
                continue
            df = enrich(df, cfg)
            res = evaluate(df, cfg)
            if res and res["is_signal"]:
                pos = open_position(state, ticker, res, cfg, today)
                if pos:
                    events.append(
                        f"ENTRADA {ticker} ({sector}) @ {res['entry']} | "
                        f"SL {res['sl']} TP {res['tp']} | {res['score']}/4 "
                        f"| riesgo {pos['risk_pct']}%")

    # --- 3. EQUITY A VALOR DE MERCADO ---------------------------------------
    mtm = mark_to_market(state, cache)
    state["equity_history"].append({
        "date": today,
        "cash_equity": round(state["equity"], 2),
        "mtm_equity": mtm,
    })

    state["last_run"] = today
    save_state(state)

    summary = build_summary(state, events, today, mtm, bullish, cfg, cache)
    print(summary)
    send_telegram(summary)
    return state


# =============================================================================
# RESUMEN Y ALERTAS
# =============================================================================

def build_summary(state, events, today, mtm, bullish, cfg, cache):
    eq = state["equity"]
    start = state["start_equity"]
    ret_mtm = (mtm - start) / start * 100
    risk_used = current_total_risk(state) * 100
    regime = "ALCISTA" if bullish else "BAJISTA"

    lines = [
        f"PAPER TRADING v3 | {today} | Regimen: {regime}",
        f"Cuenta (mercado): ${mtm:,.2f} ({ret_mtm:+.1f}%) | Efectivo: ${eq:,.2f}",
        f"Riesgo en uso: {risk_used:.1f}% de {cfg['max_total_risk']*100:.0f}% | "
        f"Abiertas: {len(state['open_positions'])} | Cerradas: {len(state['closed_trades'])}",
    ]
    if events:
        lines.append("--- Hoy ---")
        lines += events
    else:
        lines.append("Sin movimientos hoy.")

    if state["open_positions"]:
        lines.append("--- Abiertas (precio actual | PnL | progreso R) ---")
        for t, p in state["open_positions"].items():
            held = bars_held(p["entry_date"], today)
            df = cache.get(t)
            if df is not None and not df.empty:
                px = float(df["Close"].iloc[-1])
                pnl, r_mult = position_status(p, px)
                sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"  {t}: {px:.2f} | {sign}${pnl} ({sign}{r_mult}R) | "
                    f"SL {p['sl']} TP {p['tp']} | {p.get('risk_pct','?')}% | {held}d")
            else:
                lines.append(f"  {t}: sin precio hoy | SL {p['sl']} TP {p['tp']} | {held}d")
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
