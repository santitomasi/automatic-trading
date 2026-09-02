"""
=============================================================================
 SIGNALS ONLY - Feed de señales publico (corre EN PARALELO al paper trader)
=============================================================================
 NO simula cuenta, NO abre posiciones. Escanea los tickers con la MISMA
 regla del screener y publica un mensaje visual, en ingles, pensado para
 un canal publico.

 FORMATO: muestra solo las señales NUEVAS del dia como "setups
 de hoy" (no re-lista las que ya venian de dias previos), para evitar que
 el publico se confunda con valores recalculados de operaciones ya abiertas.
 Internamente sigue recordando todas las vigentes para detectar cuales son
 nuevas mañana.

 DESTINATARIOS: usa TELEGRAM_CHAT_ID_SIGNALS si existe (el CANAL); si no,
 cae a TELEGRAM_CHAT_ID. Varios IDs por coma. El bot debe ser admin del canal.
=============================================================================
"""

import os
import json
from datetime import datetime, timezone

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
# ANÁLISIS S&P 500
# =============================================================================

def analyze_sp500(cfg, df):
    if df is None or df.empty:
        return None
    d = enrich(df.copy(), cfg)
    last = d.iloc[-1]
    ema200 = float(last["ema_trend"]); price = float(last["Close"])
    return {
        "price": round(price, 2),
        "dist_pct": round((price - ema200) / ema200 * 100, 1),
        "bullish": price > ema200,
    }


# =============================================================================
# ESCANEO
# =============================================================================

def scan(cfg, cache):
    signals = {}
    for t in cfg["tickers"]:
        df = cache.get(t)
        if df is None:
            continue
        d = enrich(df.copy(), cfg)
        res = evaluate(d, cfg)
        if res and res["is_signal"]:
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
    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("last_run") == today:
        print(f"Ya se corrio hoy ({today}). Saltando.")
        return state

    cache = download_all(cfg)
    sp = analyze_sp500(cfg, cache.get(cfg["regime_ticker"]))
    current = scan(cfg, cache)
    previous = state.get("active_signals", {})

    # Opcion A: solo lo NUEVO (no estaba ayer) o que SUBIO a 4/4
    fresh = {}
    for t, v in current.items():
        was = previous.get(t)
        if was is None or v["score"] > was:
            fresh[t] = v

    summary = build_public_message(today, sp, fresh)
    print(summary)
    send_telegram(summary)

    state["active_signals"] = {t: v["score"] for t, v in current.items()}
    state["last_run"] = today
    save_state(state)
    return state


# =============================================================================
# MENSAJE PÚBLICO (ingles, visual, opcion A)
# =============================================================================

def fmt_date(today_str):
    y, m, d = today_str.split("-")
    return f"{MONTHS[int(m)-1]} {int(d)}, {y}"


def fmt_setup(ticker, v):
    return (f"  {ticker}\n"
            f"     Entry: ${v['entry']:,.2f}\n"
            f"     🛑 Stop Loss: ${v['sl']:,.2f}\n"
            f"     🎯 Take Profit: ${v['tp']:,.2f}")


def build_public_message(today, sp, fresh):
    lines = ["📊 TRADING SIGNALS", f"📅 {fmt_date(today)}", ""]

    # Estado del mercado
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

    strong = {t: v for t, v in fresh.items() if v["score"] == 4}
    moderate = {t: v for t, v in fresh.items() if v["score"] == 3}

    if not fresh:
        lines.append("")
        lines.append("No new setups today.")
        lines.append("")
    else:
        if strong:
            lines.append("🟢 STRONG SETUPS (4/4)")
            lines.append("")
            for t, v in sorted(strong.items()):
                lines.append(fmt_setup(t, v)); lines.append("")
        if moderate:
            lines.append("🟡 MODERATE SETUPS (3/4)")
            lines.append("")
            for t, v in sorted(moderate.items()):
                lines.append(fmt_setup(t, v)); lines.append("")

    lines.append(DIV)
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# =============================================================================
# TELEGRAM (canal para señales, fallback al chat privado)
# =============================================================================

def parse_recipients(raw):
    if not raw:
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def send_telegram(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    raw = os.environ.get("TELEGRAM_CHAT_ID_SIGNALS") or os.environ.get("TELEGRAM_CHAT_ID")
    recipients = parse_recipients(raw)
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
