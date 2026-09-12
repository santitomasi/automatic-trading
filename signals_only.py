"""
=============================================================================
 SIGNALS ONLY - Feed de señales publico (corre EN PARALELO al paper trader)
=============================================================================
 Escanea los tickers con la MISMA regla del screener y publica un mensaje
 visual en ingles. Dos modos:

   - "scheduled" (automatico, al cierre): solo señales NUEVAS (Opcion A),
     actualiza estado, va al CANAL. Valores definitivos.
   - "manual" (lo disparas vos): TODAS las vigentes con precios frescos, NO
     actualiza estado, va a TU CHAT PRIVADO. Vistazo en vivo.

 El modo lo define RUN_MODE (lo setea el workflow). Cada mensaje aclara si
 el mercado de EEUU esta ABIERTO (provisorio) o CERRADO (definitivo).

 FIX (v2): blindaje contra "nan". Si un ticker devuelve datos con la ultima
 vela incompleta, sus entry/sl/tp salen nan. Ahora scan() DESCARTA cualquier
 señal cuyos valores no sean finitos, en vez de mostrarla rota. Un ticker
 con datos malos simplemente no aparece ese dia.

 DESTINATARIOS:
   scheduled -> TELEGRAM_CHAT_ID_SIGNALS (canal) si existe, si no el privado.
   manual    -> TELEGRAM_CHAT_ID (tu chat privado) siempre.
=============================================================================
"""

import os
import json
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import yfinance as yf
from screener import enrich, evaluate, CONFIG


SIG_CONFIG = {
    **CONFIG,
    "regime_ticker": "^GSPC",
    "lookback": "1y",
}

STATE_FILE = "signals_state.json"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

DISCLAIMER = ("⚠️ This is NOT financial advice. Educational information only. "
              "Trading involves risk of loss. You are responsible for your "
              "own decisions.")

DIV = "━━━━━━━━━━━━━━━━━━"


# =============================================================================
# HELPERS
# =============================================================================

def is_finite_num(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


# =============================================================================
# ESTADO
# =============================================================================

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_run": None, "active_signals": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# =============================================================================
# ESTADO DEL MERCADO (EEUU, horario de Nueva York)
# =============================================================================

def market_status():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    is_weekday = now_et.weekday() < 5
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return {"is_open": is_weekday and open_t <= now_et < close_t,
            "et": now_et.strftime("%H:%M ET")}


# =============================================================================
# ANÁLISIS S&P 500 / ESCANEO  (blindados contra nan)
# =============================================================================

def analyze_sp500(cfg, df):
    if df is None or df.empty:
        return None
    d = enrich(df.copy(), cfg)
    close = d["Close"].dropna()
    ema = d["ema_trend"].dropna()
    if close.empty or ema.empty:
        return None
    price = float(close.iloc[-1]); ema200 = float(ema.iloc[-1])
    if not (is_finite_num(price) and is_finite_num(ema200)) or ema200 == 0:
        return None
    return {"price": round(price, 2),
            "dist_pct": round((price - ema200) / ema200 * 100, 1),
            "bullish": price > ema200}


def scan(cfg, cache):
    signals = {}
    for t in cfg["tickers"]:
        df = cache.get(t)
        if df is None or df.empty:
            continue
        d = enrich(df.copy(), cfg)
        res = evaluate(d, cfg)
        if not (res and res["is_signal"]):
            continue
        # BLINDAJE: descartar señales con entry/sl/tp no finitos (datos rotos)
        if not all(is_finite_num(res.get(k)) for k in ("entry", "sl", "tp")):
            print(f"  [!] {t}: señal descartada por valores invalidos (datos incompletos)")
            continue
        signals[t] = {"score": res["score"], "entry": res["entry"],
                      "sl": res["sl"], "tp": res["tp"]}
    return signals


def download_all(cfg):
    cache = {}
    for t in list(dict.fromkeys(cfg["tickers"] + [cfg["regime_ticker"]])):
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

def fmt_date(today_str):
    y, m, d = today_str.split("-")
    return f"{MONTHS[int(m)-1]} {int(d)}, {y}"


def fmt_setup(ticker, v):
    return (f"  {ticker}\n"
            f"     Entry: ${v['entry']:,.2f}\n"
            f"     🛑 Stop Loss: ${v['sl']:,.2f}\n"
            f"     🎯 Take Profit: ${v['tp']:,.2f}")


def build_public_message(today, sp, shown, mkt, mode):
    if mode == "manual":
        lines = ["📊 TRADING SIGNALS — LIVE SNAPSHOT", f"📅 {fmt_date(today)} · {mkt['et']}"]
    else:
        lines = ["📊 TRADING SIGNALS", f"📅 {fmt_date(today)}"]

    if mkt["is_open"]:
        lines.append("🔵 US market OPEN — values are provisional (still moving)")
    else:
        lines.append("⚪ US market CLOSED — values are settled")
    lines.append("")

    if sp:
        if sp["bullish"]:
            lines.append("🌎 Market (S&P 500): 🟢 Bullish")
            lines.append(f"   +{sp['dist_pct']}% above yearly average")
        else:
            lines.append("🌎 Market (S&P 500): 🔴 Bearish")
            lines.append(f"   {sp['dist_pct']}% below yearly average")
            lines.append("   ⚠️ Caution: broad trend is down")
    else:
        lines.append("🌎 Market (S&P 500): data unavailable")

    lines.append(DIV)

    strong = {t: v for t, v in shown.items() if v["score"] == 4}
    moderate = {t: v for t, v in shown.items() if v["score"] == 3}
    header_new = "" if mode == "manual" else " (new today)"

    if not shown:
        lines.append("")
        lines.append("No new setups today." if mode == "scheduled" else "No active setups right now.")
        lines.append("")
    else:
        if strong:
            lines.append(f"🟢 STRONG SETUPS (4/4){header_new}"); lines.append("")
            for t, v in sorted(strong.items()):
                lines.append(fmt_setup(t, v)); lines.append("")
        if moderate:
            lines.append(f"🟡 MODERATE SETUPS (3/4){header_new}"); lines.append("")
            for t, v in sorted(moderate.items()):
                lines.append(fmt_setup(t, v)); lines.append("")

    if mode == "manual" and mkt["is_open"]:
        lines.append("⏳ Intraday snapshot — final signals are confirmed at market close.")
        lines.append("")

    lines.append(DIV); lines.append(""); lines.append(DISCLAIMER)
    return "\n".join(lines)


# =============================================================================
# TELEGRAM
# =============================================================================

def parse_recipients(raw):
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def send_telegram(text, raw_ids):
    token = os.environ.get("TELEGRAM_TOKEN")
    recipients = parse_recipients(raw_ids)
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
