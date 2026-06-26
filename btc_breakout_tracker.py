"""
多幣種突破確認 + 出場訊號 + AI 每日建議追蹤器 (BTC / ETH / XRP)
============================================================
每次執行流程：
  1. 自動從 Farside Investors 抓取 BTC/ETH ETF 資金流（XRP 手動更新）
  2. 從 Binance 抓取日K線，計算 RSI / ATR 等技術指標
  3. 檢查四項突破條件，輸出 0–4 分評分與出場計畫
  4. 呼叫 Claude API，根據技術數據自動產生「今日操作建議」
  5. 每個幣種各推播一則 Telegram 訊息

環境變數（請設定於 GitHub Secrets）：
  TELEGRAM_BOT_TOKEN   Telegram Bot Token
  TELEGRAM_CHAT_ID     Telegram 聊天室 ID
  ANTHROPIC_API_KEY    Anthropic API 金鑰（用於 Claude AI 建議）

排程建議：GitHub Actions cron '0 */4 * * *'（每 4 小時，台灣時間整點）
"""

import os
import sys
import io
import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# ========== 追蹤幣種設定 ==========
ASSETS = {
    "BTC": {
        "symbol": "BTCUSDT",
        "decimals": 0,
        "resistance_levels": {
            "第一壓力": 66000,
            "第二壓力": 68400,
        },
        "support_levels": {
            "近期支撐": 58000,
            "深層支撐": 52000,
        },
    },
    "ETH": {
        "symbol": "ETHUSDT",
        "decimals": 0,
        "resistance_levels": {
            "第一壓力": 1730,
            "第二壓力": 1800,
        },
        "support_levels": {
            "近期支撐": 1500,
            "深層支撐": 1400,
        },
    },
    "XRP": {
        "symbol": "XRPUSDT",
        "decimals": 4,
        "resistance_levels": {
            "第一壓力": 1.19,
            "第二壓力": 1.29,
        },
        "support_levels": {
            "近期支撐": 1.00,
            "深層支撐": 0.85,
        },
    },
}

# ========== 通用參數 ==========
INTERVAL = "1d"
LOOKBACK_DAYS = 120
VOLUME_MULTIPLIER = 1.5
RETEST_WINDOW = 5
RETEST_TOLERANCE = 0.01
ATR_PERIOD = 14
STOP_ATR_MULT = 1.5
TP2_ATR_MULT = 3
TP3_ATR_MULT = 4
TRAIL_ATR_MULT = 1.5

# ETF 資金流訊號（BTC/ETH 自動抓取，XRP 手動）
ETF_FLOW_POSITIVE = {
    "BTC": False,
    "ETH": False,
    "XRP": False,
}

FARSIDE_URLS = {
    "BTC": "https://farside.co.uk/btc/",
    "ETH": "https://farside.co.uk/eth/",
}

# Claude AI 建議設定
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 800


# ========== 工具函式 ==========
def fmt(value, decimals=0):
    return f"${value:,.{decimals}f}"


def fetch_ohlcv_bybit(symbol, limit=LOOKBACK_DAYS):
    """Bybit 公開 API（無需 API Key，GitHub Actions 可用）"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "spot", "symbol": symbol, "interval": "D", "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise Exception(f"Bybit API 錯誤：{data.get('retMsg')}")
    # Bybit 回傳格式：[startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    # 由新到舊排列，需要反轉
    rows = data["result"]["list"][::-1]
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv_okx(symbol, limit=LOOKBACK_DAYS):
    """OKX 公開 API（備援，無需 API Key）"""
    # OKX symbol 格式：BTC-USDT
    okx_symbol = symbol.replace("USDT", "-USDT")
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": okx_symbol, "bar": "1D", "limit": min(limit, 300)}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise Exception(f"OKX API 錯誤：{data.get('msg')}")
    # OKX 回傳格式：[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]，由新到舊
    rows = data["data"][::-1]
    df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def fetch_ohlcv(symbol, interval=INTERVAL, limit=LOOKBACK_DAYS):
    """嘗試多個交易所 API，第一個成功就用，全部失敗才報錯"""
    sources = [
        ("Bybit",  fetch_ohlcv_bybit),
        ("OKX",    fetch_ohlcv_okx),
    ]
    last_error = None
    for name, func in sources:
        try:
            df = func(symbol, limit)
            print(f"  ✅ 資料來源：{name}")
            return df
        except Exception as e:
            print(f"  ⚠️ {name} 失敗：{e}")
            last_error = e
    raise Exception(f"所有資料來源均失敗，最後錯誤：{last_error}")
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open_time", "open", "high", "low", "close", "volume"]]


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=ATR_PERIOD):
    high, low = df["high"], df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ========== ETF 資金流自動抓取 ==========
def parse_money(text):
    text = text.replace(",", "").strip()
    if text in ("", "-", "—"):
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def fetch_etf_flow_farside(url):
    date_pattern = re.compile(r"^\d{1,2}\s+[A-Za-z]{3}\s+\d{4}$")
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        latest_date, latest_total = None, None
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            first_cell = cells[0].get_text(strip=True)
            if date_pattern.match(first_cell):
                value = parse_money(cells[-1].get_text(strip=True))
                if value is not None:
                    latest_date, latest_total = first_cell, value
        return latest_date, latest_total
    except Exception:
        return None, None


def update_etf_flow_auto():
    etf_summary = {}
    for asset_name, url in FARSIDE_URLS.items():
        date_str, total = fetch_etf_flow_farside(url)
        if total is not None:
            ETF_FLOW_POSITIVE[asset_name] = total > 0
            sign = "淨流入" if total > 0 else "淨流出"
            msg = f"📡 {asset_name} ETF：{date_str} {sign} ${abs(total):,.1f}M"
            etf_summary[asset_name] = {"date": date_str, "total": total, "positive": total > 0}
        else:
            msg = f"⚠️ {asset_name} ETF 自動抓取失敗，沿用手動設定值"
            etf_summary[asset_name] = {"date": None, "total": None, "positive": ETF_FLOW_POSITIVE.get(asset_name)}
        print(msg)
    etf_summary["XRP"] = {"date": None, "total": None, "positive": ETF_FLOW_POSITIVE.get("XRP")}
    return etf_summary


# ========== 四項突破條件 ==========
def check_breakout(df, resistance):
    latest_close = df["close"].iloc[-1]
    return latest_close > resistance, latest_close


def check_volume_confirm(df, multiplier=VOLUME_MULTIPLIER):
    # CoinGecko OHLC API 不提供成交量，自動回傳中性結果
    if df["volume"].sum() == 0:
        return None, None  # None 表示無資料，評分時跳過
    avg_vol_20 = df["volume"].iloc[-21:-1].mean()
    if avg_vol_20 == 0:
        return None, None
    latest_vol = df["volume"].iloc[-1]
    ratio = latest_vol / avg_vol_20
    return ratio >= multiplier, ratio


def check_retest_hold(df, resistance, window=RETEST_WINDOW, tolerance=RETEST_TOLERANCE):
    recent = df.tail(window + 1)
    broke_above = (recent["close"] > resistance).any()
    if not broke_above:
        return False, None
    pulled_back = recent["low"].min() <= resistance * (1 + tolerance)
    held = recent["close"].iloc[-1] > resistance * (1 - tolerance)
    return pulled_back, held


def evaluate_breakout_status(df, asset_name, resistance_name, resistance_level, decimals):
    print(f"\n----- {resistance_name}：{fmt(resistance_level, decimals)} -----")
    is_above, latest_close = check_breakout(df, resistance_level)
    vol_confirmed, vol_ratio = check_volume_confirm(df)
    tested, held = check_retest_hold(df, resistance_level)
    score = 0
    print(f"現價：{fmt(latest_close, decimals)}")
    if is_above:
        print("✅ 條件1：收盤站穩壓力之上")
        score += 1
    else:
        gap_pct = ((resistance_level / latest_close) - 1) * 100
        print(f"❌ 條件1：尚未站穩（距離壓力 {gap_pct:.1f}%）")
    if vol_confirmed is None:
        print("➖ 條件2：成交量資料不可用（CoinGecko OHLC 不提供量能）")
    elif vol_confirmed:
        print(f"✅ 條件2：放量確認（量比 {vol_ratio:.2f}x）")
        score += 1
    else:
        print(f"❌ 條件2：量能不足（量比 {vol_ratio:.2f}x，需 ≥ {VOLUME_MULTIPLIER}x）")
    if tested and held:
        print("✅ 條件3：拉回測試守住")
        score += 1
    elif tested and not held:
        print("⚠️ 條件3：拉回測試未守住（疑似假突破）")
    else:
        print("➖ 條件3：尚未發生拉回測試")
    if ETF_FLOW_POSITIVE.get(asset_name, False):
        print("✅ 條件4：ETF/籌碼面轉強")
        score += 1
    else:
        print("❌ 條件4：ETF/籌碼面尚未轉強")
    print(f"突破確認分數：{score}/4", end="  ")
    if score >= 3:
        print("🟢 高信心突破訊號")
    elif score == 2:
        print("🟡 初步訊號，建議觀察")
    else:
        print("🔴 尚未突破，建議觀望")
    return score


def suggest_exit_plan(df, resistance_level, decimals, next_resistance=None):
    atr = df["ATR"].iloc[-1]
    stop_loss = resistance_level - STOP_ATR_MULT * atr
    tp1 = next_resistance if next_resistance else resistance_level + 2 * atr
    tp2 = resistance_level + TP2_ATR_MULT * atr
    tp3 = resistance_level + TP3_ATR_MULT * atr
    risk = resistance_level - stop_loss
    rr1 = (tp1 - resistance_level) / risk if risk > 0 else float("nan")
    rr2 = (tp2 - resistance_level) / risk if risk > 0 else float("nan")
    rr3 = (tp3 - resistance_level) / risk if risk > 0 else float("nan")
    print(f"出場計畫（ATR={fmt(atr, decimals)}）：")
    print(f"  停損 {fmt(stop_loss, decimals)}（關卡下方 {STOP_ATR_MULT}xATR）")
    tag1 = "下一壓力位" if next_resistance else "+2xATR"
    print(f"  TP1 {fmt(tp1, decimals)}（{tag1}，風報比 {rr1:.1f}R）→ 出1/3部位，停損移成本價")
    print(f"  TP2 {fmt(tp2, decimals)}（+{TP2_ATR_MULT}xATR，風報比 {rr2:.1f}R）→ 再出1/3，停損移TP1")
    print(f"  TP3 {fmt(tp3, decimals)}（+{TP3_ATR_MULT}xATR，風報比 {rr3:.1f}R）→ 剩餘改移動停利")
    print(f"  移動停利：回落 {TRAIL_ATR_MULT}xATR({fmt(TRAIL_ATR_MULT*atr, decimals)}) 即出場")


# ========== Claude AI 每日建議 ==========
def generate_ai_recommendation(asset_name, config, df, breakout_scores, etf_info):
    """呼叫 Claude API，根據技術數據產生繁體中文操作建議"""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️ 未設定 ANTHROPIC_API_KEY，略過 AI 建議"

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change_pct = (latest["close"] - prev["close"]) / prev["close"] * 100
    rsi = df["RSI"].iloc[-1]
    atr = df["ATR"].iloc[-1]
    decimals = config["decimals"]

    resistance_list = "\n".join(
        [f"  - {k}：{fmt(v, decimals)}" for k, v in config["resistance_levels"].items()]
    )
    support_list = "\n".join(
        [f"  - {k}：{fmt(v, decimals)}" for k, v in config["support_levels"].items()]
    )
    score_list = "\n".join(
        [f"  - {k}：{v}/4 分" for k, v in breakout_scores.items()]
    )
    etf_status = etf_info.get(asset_name, {})
    if etf_status.get("total") is not None:
        etf_text = f"最新一日{'淨流入' if etf_status['positive'] else '淨流出'} ${abs(etf_status['total']):,.1f}M（{etf_status['date']}）"
    else:
        etf_text = "資料暫無法取得"

    today_str = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""你是一位專業的加密貨幣技術分析師，請根據以下 {asset_name} 的即時技術數據，
用繁體中文提供今日操作建議。請保持客觀，不要過度樂觀或悲觀。
今天日期是：{today_str}

【技術數據】
- 現價：{fmt(latest['close'], decimals)}
- 24小時漲跌：{price_change_pct:+.2f}%
- RSI(14)：{rsi:.1f}
- ATR(14)：{fmt(atr, decimals)}
- 成交量（今日/20日均量比）：{latest['volume'] / df['volume'].iloc[-21:-1].mean():.2f}x

【關鍵價位】
壓力區：
{resistance_list}
支撐區：
{support_list}

【突破確認分數（0–4分）】
{score_list}

【ETF 資金流向】
{etf_text}

請依照以下格式回覆（直接輸出，不要有任何前言）：

📊 {asset_name} 今日分析（{today_str}）

🔍 現況判讀
（2–3句，描述目前趨勢強弱、RSI 位置、量能狀況）

🎯 操作建議
（明確說明：現在適不適合進場？為什麼？如果不適合，等什麼條件？）

📌 關鍵價位
• 進場參考：
• 停損設定：
• 短期停利：

⚠️ 主要風險
（1–2句，列出最需要注意的下行風險）

---
以上為技術分析參考，非投資建議。"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": CLAUDE_MAX_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )
        # 日期已在 prompt 中直接傳入，無需替換
        return text.strip()
    except Exception as e:
        return f"⚠️ Claude API 呼叫失敗：{e}"


# ========== 單一幣種完整報告 ==========
def generate_asset_report(asset_name, config, etf_summary):
    symbol = config["symbol"]
    decimals = config["decimals"]
    resistance_levels = config["resistance_levels"]

    print(f"\n{'='*55}")
    print(f"  {asset_name} 突破確認追蹤 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    df = fetch_ohlcv(symbol)
    df["RSI"] = compute_rsi(df["close"])
    df["ATR"] = compute_atr(df)
    print(f"現價：{fmt(df['close'].iloc[-1], decimals)}　RSI(14)：{df['RSI'].iloc[-1]:.1f}　ATR(14)：{fmt(df['ATR'].iloc[-1], decimals)}")

    levels = list(resistance_levels.items())
    breakout_scores = {}
    for i, (name, level) in enumerate(levels):
        score = evaluate_breakout_status(df, asset_name, name, level, decimals)
        breakout_scores[name] = score
        next_level = levels[i + 1][1] if i + 1 < len(levels) else None
        suggest_exit_plan(df, level, decimals, next_resistance=next_level)

    print(f"\n{'─'*40}")
    print("🤖 Claude AI 今日建議")
    print(f"{'─'*40}")
    ai_text = generate_ai_recommendation(asset_name, config, df, breakout_scores, etf_summary)
    print(ai_text)

    return df, ai_text


# ========== Telegram 推播 ==========
def send_telegram_message(text):
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("⚠️ 未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，略過推播")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:4000]}
    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        print("✅ 已推播至 Telegram")
    except Exception as e:
        print(f"⚠️ Telegram 推播失敗：{e}")


# ========== 主程式 ==========
def main():
    print("===== 步驟1：自動更新 ETF 資金流訊號 =====")
    etf_summary = update_etf_flow_auto()

    print("\n===== 步驟2：逐幣種產生報告並推播 =====")
    for asset_name, config in ASSETS.items():
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        ai_text = ""
        try:
            _, ai_text = generate_asset_report(asset_name, config, etf_summary)
        except Exception as e:
            print(f"⚠️ {asset_name} 執行失敗：{e}")
        finally:
            sys.stdout = original_stdout

        report_text = buffer.getvalue()
        print(report_text)

        # 推播兩則：技術指標報告 + AI 建議各一則，確保不超過 Telegram 字數限制
        send_telegram_message(report_text)
        if ai_text:
            send_telegram_message(ai_text)


if __name__ == "__main__":
    main()
