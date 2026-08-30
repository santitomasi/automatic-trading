"""
=============================================================================
 PAPER TRADER TRAILING - Variante con trailing stop 3.5 ATR (EN PARALELO)
=============================================================================
 Segunda variante que compite en vivo contra el paper trader FIJO. Misma
 entrada (regla larga de screener.py), mismos controles de cartera, PERO
 salida por TRAILING STOP en vez de objetivo fijo 1:2.

 DIFERENCIAS con el paper trader fijo:
   - Salida: stop inicial a 3.5*ATR, que PERSIGUE al precio hacia arriba
     (nunca baja). Sin take profit fijo -> deja correr las tendencias.
   - Stop temporal: 120 dias habiles (red de seguridad larga), NO 40.
     El de 40 cortaria justo las corridas largas que dan la ventaja al
     trailing. El trailing ya limpia solo las posiciones estancadas.
   - Enfriamiento: solo tras una salida PERDEDORA (una salida ganadora
     significa que la tendencia maduro; reentrar con nueva señal es OK).

 Estado propio (state_trailing.json) y CSV propio (trades_trailing.csv),
 independientes del sistema fijo. Telegram etiquetado "TRAILING".
 Todo lo demas (regimen, topes, riesgo dinamico, mtm) es identico al fijo
 para que la comparacion sea justa: solo cambia el mecanismo de salida.
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

    "risk_by_score": {3: 0.01, 4: 0.02},
    "min_position_risk": 0.005,

    "max_total_risk": 0.06,
    "max_per_sector": 2,
    "regime_ticker": "^GSPC",
    "cooldown_bars": 5,

    # --- Especifico del trailing --------------------------------------------
    "trail_mult": 3.5,          # distancia del trailing stop, en ATR
    "time_stop_bars": 120,      # red de seguridad larga (NO 40)
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

STATE_FILE = "state_trailing.json"
TRADES_CSV = "trades_trailing.csv"
CSV_FIELDS = ["ticker", "entry_date", "exit_date", "entry", "exit", "shares",
              "outcome", "pnl", "R", "equity_after", "score", "risk_pct", "bars_held"]


# =============================================================================
# ESTADO
# =============================================================================

def load_state(cfg):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        state.setdefault("equity_history", [])
        state.setdefault("cooldowns", {})
        return state
    return {
        "equity": cfg["start_equity"], "start_equity": cfg["start_equity"],
        "peak_equity": cfg["start_equity"],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "last_run": None, "open_positions": {}, "closed_trades": [],
        "equity_history": [], "cooldowns": {},
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
# GESTIÓN DE RIESGO (identica al fijo)
# =============================================================================

def current_total_risk(state):
    total = sum(p.get("risk_amount", 0) for p in state["open_positions"].values())
    return total / state["equity"] if state["equity"] > 0 else 1.0


def sector_count(state, sector):
    return sum(1 for t in state["open_positions"] if SECTORS.get(t, "otros") == sector)


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
    last = state.get("cooldowns", {}).get(ticker)
    if not last:
        return False
    return bars_held(last, today) < cfg["cooldown_bars"]


def desired_risk_fraction(score, cfg):
    return cfg["risk_by_score"].get(score, min(cfg["risk_by_score"].values()))


# =============================================================================
# CUENTA (apertura con trailing, sin TP fijo)
# =============================================================================

def open_position(state, ticker, sig, atr_val, cfg, today):
    equity = state["equity"]
    wanted = desired_risk_fraction(sig["score"], cfg)
    headroom = cfg["max_total_risk"] - current_total_risk(state)
    risk_frac = min(wanted, headroom)
    if risk_frac < cfg["min_position_risk"]:
        return None

    entry = sig["entry"]
    trail_dist = cfg["trail_mult"] * atr_val
    init_stop = entry - trail_dist
    per_share_risk = entry - init_stop
    if per_share_risk <= 0:
        return None
    risk_amount = equity * risk_frac
    shares = risk_amount / per_share_risk

    state["open_positions"][ticker] = {
        "entry": entry, "init_stop": round(init_stop, 2),
        "trail_dist": round(trail_dist, 4), "highest": entry,
        "shares": round(shares, 4), "risk_amount": round(risk_amount, 2),
        "risk_pct": round(risk_frac * 100, 2), "score": sig["score"],
        "entry_date": today,
    }
    return state["open_positions"][ticker]


def close_position(state, ticker, exit_price, outcome, cfg, today):
    pos = state["open_positions"].pop(ticker)
    cost = cfg["cost_pct"] / 100.0
    eff_entry = pos["entry"] * (1 + cost)
    eff_exit = exit_price * (1 - cost)
    pnl = pos["shares"] * (eff_exit - eff_entry)
    risk = pos["entry"] - pos["init_stop"]
    r_mult = (exit_price - pos["entry"]) / risk if risk > 0 else 0

    state["equity"] += pnl
    state["peak_equity"] = max(state["peak_equity"], state["equity"])

    # Enfriamiento SOLO si la salida fue perdedora
    if pnl < 0:
        state.setdefault("cooldowns", {})[ticker] = today

    trade = {
        "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": today,
        "entry": round(pos["entry"], 2), "exit": round(exit_price, 2),
        "shares": pos["shares"], "outcome": outcome,
        "pnl": round(pnl, 2), "R": round(r_mult, 2),
        "equity_after": round(state["equity"], 2),
        "score": pos.get("score", ""), "risk_pct": pos.get("risk_pct", ""),
        "bars_held": bars_held(pos["entry_date"], today),
    }
    state["closed_trades"].append(trade)
    append_csv(trade)
    return trade


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
    for t in list(dict.fromkeys(cfg["tickers"] + [cfg["regime_ticker"]])):
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

    # --- 1. SALIDAS (trailing + red de seguridad temporal) ------------------
    for ticker in list(state["open_positions"].keys()):
        df = cache.get(ticker)
        if df is None:
            events.append(f"[!] {ticker}: sin datos hoy, sigue abierta")
            continue
        last = df.iloc[-1]
        pos = state["open_positions"][ticker]

        # Trailing stop vigente HOY = maximo hasta AYER menos la distancia
        trailing = pos["highest"] - pos["trail_dist"]
        cur_stop = max(pos["init_stop"], trailing)
        expired = bars_held(pos["entry_date"], today) >= cfg["time_stop_bars"]

        if last["Low"] <= cur_stop:
            t = close_position(state, ticker, cur_stop, "TRAIL", cfg, today)
            tag = " (enfriamiento)" if t["pnl"] < 0 else ""
            events.append(f"CIERRE {ticker}: TRAILING @ {round(cur_stop,2)}, "
                          f"PnL ${t['pnl']} ({t['R']}R){tag}")
        elif expired:
            t = close_position(state, ticker, float(last["Close"]), "TIEMPO", cfg, today)
            events.append(f"CIERRE {ticker}: STOP TEMPORAL ({cfg['time_stop_bars']}d), "
                          f"PnL ${t['pnl']} ({t['R']}R)")
        else:
            # No salio: actualizar el maximo con el alto de HOY (para mañana)
            new_high = max(pos["highest"], float(last["High"]))
            pos["highest"] = round(new_high, 2)

    # --- 2. ENTRADAS (solo regimen alcista) ---------------------------------
    if not bullish:
        events.append("REGIMEN BAJISTA (S&P bajo EMA200): no se abren posiciones.")
    else:
        for ticker in cfg["tickers"]:
            if ticker in state["open_positions"] or in_cooldown(state, ticker, today, cfg):
                continue
            headroom = cfg["max_total_risk"] - current_total_risk(state)
            if headroom < cfg["min_position_risk"]:
                events.append(f"PRESUPUESTO DE RIESGO agotado "
                              f"({current_total_risk(state)*100:.1f}%).")
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
                atr_val = float(df.iloc[-1]["atr"])
                pos = open_position(state, ticker, res, atr_val, cfg, today)
                if pos:
                    events.append(
                        f"ENTRADA {ticker} ({sector}) @ {res['entry']} | "
                        f"stop inicial {pos['init_stop']} (trail {cfg['trail_mult']}ATR) "
                        f"| {res['score']}/4 | riesgo {pos['risk_pct']}%")

    # --- 3. EQUITY A VALOR DE MERCADO ---------------------------------------
    mtm = mark_to_market(state, cache)
    state["equity_history"].append({
        "date": today, "cash_equity": round(state["equity"], 2), "mtm_equity": mtm,
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
    eq = state["equity"]; start = state["start_equity"]
    ret_mtm = (mtm - start) / start * 100
    risk_used = current_total_risk(state) * 100
    regime = "ALCISTA" if bullish else "BAJISTA"

    lines = [
        f"PAPER TRADING TRAILING {cfg['trail_mult']}ATR | {today} | Regimen: {regime}",
        f"Cuenta (mercado): ${mtm:,.2f} ({ret_mtm:+.1f}%) | Efectivo: ${eq:,.2f}",
        f"Riesgo en uso: {risk_used:.1f}% de {cfg['max_total_risk']*100:.0f}% | "
        f"Abiertas: {len(state['open_positions'])} | Cerradas: {len(state['closed_trades'])}",
    ]
    if events:
        lines.append("--- Hoy ---"); lines += events
    else:
        lines.append("Sin movimientos hoy.")

    if state["open_positions"]:
        lines.append("--- Abiertas (precio | PnL | R | stop movil) ---")
        for t, p in state["open_positions"].items():
            held = bars_held(p["entry_date"], today)
            df = cache.get(t)
            if df is not None and not df.empty:
                px = float(df["Close"].iloc[-1])
                risk = p["entry"] - p["init_stop"]
                pnl = p["shares"] * (px - p["entry"])
                r_mult = (px - p["entry"]) / risk if risk > 0 else 0
                cur_stop = max(p["init_stop"], p["highest"] - p["trail_dist"])
                sign = "+" if pnl >= 0 else ""
                lines.append(f"  {t}: {px:.2f} | {sign}${round(pnl,2)} "
                             f"({sign}{round(r_mult,2)}R) | stop {round(cur_stop,2)} | {held}d")
            else:
                lines.append(f"  {t}: sin precio hoy | {held}d")
    return "\n".join(lines)


def parse_recipients(raw):
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def send_telegram(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    recipients = parse_recipients(os.environ.get("TELEGRAM_CHAT_ID"))
    if not token or not recipients:
        print("(Telegram no configurado; omitiendo alerta)")
        return
    import urllib.request, urllib.parse
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok, fail = 0, 0
    for chat_id in recipients:
        try:
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"(Fallo al enviar a {chat_id}: {e})")
    print(f"(Enviado a {ok} destinatario(s); {fail} fallaron)")


if __name__ == "__main__":
    run()
