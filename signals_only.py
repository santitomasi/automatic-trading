"""
=============================================================================
 SIGNALS ONLY - Feed de señales publico (corre EN PARALELO al paper trader)
=============================================================================
 Dos modos: "scheduled" (automatico al cierre, solo señales NUEVAS, al CANAL)
 y "manual" (vistazo en vivo con todas las vigentes, a tu chat privado).

 FIX DE RAIZ (v3): clean_ohlc() elimina las filas con precios vacios (la
 fila "placeholder" que yfinance devuelve a veces) APENAS se descargan,
 antes de calcular ningun indicador. Asi todo se evalua sobre el ultimo
 dia real y completo: una señal 4/4 se ve como 4/4 con precios reales,
 en vez de deformarse en "3/4 con NaN" y perderse.
=============================================================================
"""

import os
import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import yfinance as yf
from screener import enrich, evaluate, CONFIG


SIG_CONFIG = {**CONFIG, "regime_ticker": "^GSPC", "lookback": "1y"}
STATE_FILE = "signals_state.json"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DISCLAIMER = ("⚠️ This is NOT financial advice. Educational information only. "
              "Trading involves risk of loss. You are responsible for your "
              "own decisions.")
DIV = "━━━━━━━━━━━━━━━━━━"


# =============================================================================
# LIMPIEZA DE DATOS (el arreglo de raiz)
# =============================================================================

def clean_ohlc(df):
    """Elimina filas con High/Low/Close vacios. Se aplica al descargar,
    ANTES de cualquier indicador. Devuelve None si no queda nada util."""
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["High", "Low", "Close"])
    return d if not d.empty else None


def is_finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


# =============================================================================
# ESTADO / MERCADO
# =============================================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "active_signals": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def market_status():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    is_weekday = now_et.weekday() < 5
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return {"is_open": is_weekday and open_t <= now_et < close_t,
            "et": now_et.strftime("%H:%M ET")}


# =============================================================================
# ANÁLISIS / ESCANEO
# =============================================================================

def analyze_sp500(cfg, df):
    if df is None or df.empty:
        return None
    d = enrich(df.copy(), cfg)
    price = float(d["Close"].iloc[-1]); ema200 = float(d["ema_trend"].iloc[-1])
    if not (is_finite_num(price) and is_finite_num(ema200)) or ema200 == 0:
        return None
    return {"price": round(price, 2),
            "dist_pct": round((price - ema200) / ema200 * 100, 1),
            "bullish": price > ema200}


def scan(cfg, cache):
    signals = {}
    for t in cfg["tickers"]:
        df = cache.get(t)
        if df is None:
            continue
        res = evaluate(enrich(df.copy(), cfg), cfg)
        if not (res and res["is_signal"]):
            continue
        # Red de seguridad (con los datos limpios no deberia dispararse)
        if not all(is_finite_num(res.get(k)) for k in ("entry", "sl", "tp")):
            print(f"  [!] {t}: señal descartada por valores invalidos")
            continue
        signals[t] = {"score": res["score"], "entry": res["entry"],
                      "sl": res["sl"], "tp": res["tp"]}
    return signals


def download_all(cfg):
    cache = {}
    for t in list(dict.fromkeys(cfg["tickers"] + [cfg["regime_ticker"]])):
        try:
            raw = yf.Ticker(t).history(period=cfg["lookback"], interval=cfg["interval"])
            cache[t] = clean_ohlc(raw)          # <- limpieza en la fuente
        except Exception:
            cache[t] = None
    return cache


# =============================================================================
# CICLO PRINCIPAL
# =============================================================================

def run(cfg=SIG_CONFIG):
    mode = os.environ.get("RUN_MODE", "scheduled").strip().lower()
    if mode not in ("scheduled", "manual"):
        mode = "scheduled"
    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if mode == "scheduled" and state.get("last_run") == today:
        print(f"Ya se corrio hoy ({today}). Saltando (modo automatico).")
        return state

    cache = download_all(cfg)
    sp = analyze_sp500(cfg, cache.get(cfg["regime_ticker"]))
    current = scan(cfg, cache)
    mkt = market_status()

    if mode == "scheduled":
        previous = state.get("active_signals", {})
        shown = {t: v for t, v in current.items()
                 if previous.get(t) is None or v["score"] > previous.get(t)}
        raw_ids = os.environ.get("TELEGRAM_CHAT_ID_SIGNALS") or os.environ.get("TELEGRAM_CHAT_ID")
    else:
        shown = current
        raw_ids = os.environ.get("TELEGRAM_CHAT_ID")

    print(f"Señales vigentes: {len(current)} | nuevas hoy: {len(shown)} | tickers con datos: "
          f"{sum(1 for t in cfg['tickers'] if cache.get(t) is not None)}/{len(cfg['tickers'])}")
    summary = build_public_message(today, sp, shown, mkt, mode)
    print(summary)
    send_telegram(summary, raw_ids)

    if mode == "scheduled":
        state["active_signals"] = {t: v["score"] for t, v in current.items()}
        state["last_run"] = today
        save_state(state)
    return state


# =============================================================================
# MENSAJE PÚBLICO
# =============================================================================

def fmt_date(s):
    y, m, d = s.split("-")
    return f"{MONTHS[int(m)-1]} {int(d)}, {y}"


def fmt_setup(t, v):
    return (f"  {t}\n     Entry: ${v['entry']:,.2f}\n"
            f"     🛑 Stop Loss: ${v['sl']:,.2f}\n     🎯 Take Profit: ${v['tp']:,.2f}")


def build_public_message(today, sp, shown, mkt, mode):
    if mode == "manual":
        lines = ["📊 TRADING SIGNALS — LIVE SNAPSHOT", f"📅 {fmt_date(today)} · {mkt['et']}"]
    else:
        lines = ["📊 TRADING SIGNALS", f"📅 {fmt_date(today)}"]
    lines.append("🔵 US market OPEN — values are provisional (still moving)" if mkt["is_open"]
                 else "⚪ US market CLOSED — values are settled")
    lines.append("")
    if sp:
        if sp["bullish"]:
            lines += ["🌎 Market (S&P 500): 🟢 Bullish", f"   +{sp['dist_pct']}% above yearly average"]
        else:
            lines += ["🌎 Market (S&P 500): 🔴 Bearish", f"   {sp['dist_pct']}% below yearly average",
                      "   ⚠️ Caution: broad trend is down"]
    else:
        lines.append("🌎 Market (S&P 500): data unavailable")
    lines.append(DIV)

    strong = {t: v for t, v in shown.items() if v["score"] == 4}
    moderate = {t: v for t, v in shown.items() if v["score"] == 3}
    tag = "" if mode == "manual" else " (new today)"
    if not shown:
        lines += ["", "No new setups today." if mode == "scheduled" else "No active setups right now.", ""]
    else:
        if strong:
            lines += [f"🟢 STRONG SETUPS (4/4){tag}", ""]
            for t, v in sorted(strong.items()):
                lines += [fmt_setup(t, v), ""]
        if moderate:
            lines += [f"🟡 MODERATE SETUPS (3/4){tag}", ""]
            for t, v in sorted(moderate.items()):
                lines += [fmt_setup(t, v), ""]
    if mode == "manual" and mkt["is_open"]:
        lines += ["⏳ Intraday snapshot — final signals are confirmed at market close.", ""]
    lines += [DIV, "", DISCLAIMER]
    return "\n".join(lines)


# =============================================================================
# TELEGRAM
# =============================================================================

def parse_recipients(raw):
    return [r.strip() for r in raw.split(",") if r.strip()] if raw else []


def send_telegram(text, raw_ids):
    token = os.environ.get("TELEGRAM_TOKEN")
    recipients = parse_recipients(raw_ids)
    if not token or not recipients:
        print("(Telegram no configurado; omitiendo alerta)"); return
    import urllib.request, urllib.parse
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok = fail = 0
    for chat_id in recipients:
        try:
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=15)
            ok += 1
        except Exception as e:
            fail += 1; print(f"(Fallo al enviar a {chat_id}: {e})")
    print(f"(Enviado a {ok} destinatario(s); {fail} fallaron)")


if __name__ == "__main__":
    run()
