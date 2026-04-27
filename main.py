#!/data/data/com.termux/files/usr/bin/env python3
"""
Crypto TA Analyzer + Smart Money (Production Version)
+ OOP Refactored
+ LTF Confirmation
+ Order Block Mitigation & FVG
+ Entry / SL / TP / RR (Dynamic Swings)
+ Vectorized Operations (Termux Friendly)
"""

import time
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt
from tabulate import tabulate
import warnings

warnings.filterwarnings('ignore')

# =========================
# CONFIG
# =========================
MTF_MAP = {
    "5m":  ("1h", "5m", "1m"),
    "15m": ("4h", "15m", "5m"),
    "1h":  ("1d", "1h", "15m"),
    "4h":  ("1w", "4h", "1h"),
}

RISK_REWARD = 2.0

SCORE_WEIGHTS = {
    "ema_cross": 15,
    "ema50_trend": 10,
    "rsi_momentum": 8,
    "rsi_fast": 4,
    "macd_hist": 10,
    "stoch_cross": 10,
    "bb_mid_bias": 5,
    "vol_spike": 6,
    "vol_flat": 2, 
    "obv_trend": 6,
    "htf_ema200": 15,
    "ltf_align": 10  # [POIN 2] Bobot skor untuk konfirmasi LTF
}

# =========================
# INDICATORS (Stateless)
# =========================
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    gain = up.ewm(alpha=1/n).mean()
    loss = dn.ewm(alpha=1/n).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(s):
    e1, e2 = ema(s, 12), ema(s, 26)
    m = e1 - e2
    sig = ema(m, 9)
    return m, sig, m - sig

def atr(df, n=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n).mean()

def bollinger_bands(s, period=20, std_dev=2):
    mid = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    upper = mid + (std_dev * sd)
    lower = mid - (std_dev * sd)
    width = (upper - lower) / (mid + 1e-9)
    return mid, upper, lower, width

def stochastic_oscillator(df, k_period=14, d_period=3, smooth_k=3):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()

    raw_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    stoch_k = raw_k.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(d_period).mean()

    return stoch_k, stoch_d

def obv(df):
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()

def compute_indicators(df):
    df = df.copy()
    close = df['close']

    df["ema9"] = ema(close, 9)
    df["ema21"] = ema(close, 21)
    df["ema50"] = ema(close, 50)
    df["ema200"] = ema(close, 200)

    df["rsi14"] = rsi(close, 14)
    df["rsi7"] = rsi(close, 7)

    df["macd"], df["macd_sig"], df["macd_hist"] = macd(close)

    df["bb_mid"], df["bb_upper"], df["bb_lower"], df["bb_width"] = bollinger_bands(close, 20, 2)
    df["stoch_k"], df["stoch_d"] = stochastic_oscillator(df, 14, 3, 3)

    df["obv"] = obv(df)
    df["obv_ma"] = df["obv"].rolling(20).mean()

    df["atr"] = atr(df)
    df["vol_ma"] = df["volume"].rolling(20).mean()

    return df

# =========================
# SMC LOGIC (HANYA BAGIAN YANG DIPERBAIKI)
# =========================
def swings(df, left=5, right=5):
    is_swing_high = pd.Series(False, index=df.index)
    is_swing_low = pd.Series(False, index=df.index)
    
    for i in range(left, len(df) - right):
        window_high = df['high'].iloc[i-left:i+right+1]
        window_low  = df['low'].iloc[i-left:i+right+1]
        
        if df['high'].iloc[i] == window_high.max():
            is_swing_high.iloc[i] = True
        if df['low'].iloc[i] == window_low.min():
            is_swing_low.iloc[i] = True

    sh = df.index[is_swing_high].tolist()
    sl = df.index[is_swing_low].tolist()
    
    return sh, sl

def structure(df):
    sh, sl = swings(df, left=5, right=5)
    if not sh or not sl:
        return "RANGE"

    last_close = df['close'].iloc[-1]
    hh = df['high'].iloc[sh[-1]]
    ll = df['low'].iloc[sl[-1]]

    if last_close > hh:
        return "BOS UP"
    if last_close < ll:
        return "BOS DOWN"

    if len(sh) > 1 and len(sl) > 1:
        prev_hh = df['high'].iloc[sh[-2]]
        prev_ll = df['low'].iloc[sl[-2]]
        
        if hh < prev_hh and last_close < prev_ll:
            return "CHOCH DOWN"
        if ll > prev_ll and last_close > prev_hh:
            return "CHOCH UP"

    return "RANGE"

def order_block_fvg(df, lookback=30):
    """[IMPROVED] 
    - Strength filter (displacement > ATR × 1.3)
    - Mitigation: Untouched / Partial / Full Mitigated (berdasarkan wick vs close)"""
    recent = df.tail(lookback + 10).reset_index(drop=True)
    
    atr_avg = recent['atr'].mean() if not recent['atr'].empty else (recent['high'] - recent['low']).mean()
    if atr_avg <= 0:
        atr_avg = 1e-8
    
    # FVG tetap sama
    bull_fvg = (recent['low'] > recent['high'].shift(2)) & \
               ((recent['low'] - recent['high'].shift(2)) > atr_avg * 0.5)
    bear_fvg = (recent['high'] < recent['low'].shift(2)) & \
               ((recent['low'].shift(2) - recent['high']) > atr_avg * 0.5)

    # OB + strength filter (displacement candle)
    bullish_ob = (recent['close'].shift(1) < recent['open'].shift(1)) & \
                 (recent['close'] > recent['open']) & \
                 (recent['close'] > recent['high'].shift(1)) & \
                 (abs(recent['close'] - recent['open']) > atr_avg * 1.3)
                 
    bearish_ob = (recent['close'].shift(1) > recent['open'].shift(1)) & \
                 (recent['close'] < recent['open']) & \
                 (recent['close'] < recent['low'].shift(1)) & \
                 (abs(recent['close'] - recent['open']) > atr_avg * 1.3)

    bull_idx = recent[bullish_ob].index
    bear_idx = recent[bearish_ob].index

    if not bull_idx.empty and (bear_idx.empty or bull_idx[-1] > bear_idx[-1]):
        idx = bull_idx[-1]
        ob_candle_idx = max(0, idx - 1)
        ob_high = recent['high'].iloc[ob_candle_idx]
        ob_low  = recent['low'].iloc[ob_candle_idx]
        
        # Refined mitigation (Partial / Full)
        mitigation_status = "Untouched"
        if idx + 1 < len(recent):
            post = recent.iloc[idx+1:]
            touched = (post['low'] <= ob_high).any()
            fully_mitigated = (post['close'] <= ob_low).any()
            if fully_mitigated:
                mitigation_status = "Full Mitigated"
            elif touched:
                mitigation_status = "Partial"
        
        has_fvg = False
        if idx + 2 < len(recent):
            has_fvg = bull_fvg.iloc[idx+1] or bull_fvg.iloc[idx+2]
            
        return f"Bullish OB{'+FVG' if has_fvg else ''} ({mitigation_status})"

    elif not bear_idx.empty:
        idx = bear_idx[-1]
        ob_candle_idx = max(0, idx - 1)
        ob_high = recent['high'].iloc[ob_candle_idx]
        ob_low  = recent['low'].iloc[ob_candle_idx]
        
        mitigation_status = "Untouched"
        if idx + 1 < len(recent):
            post = recent.iloc[idx+1:]
            touched = (post['high'] >= ob_low).any()
            fully_mitigated = (post['close'] >= ob_high).any()
            if fully_mitigated:
                mitigation_status = "Full Mitigated"
            elif touched:
                mitigation_status = "Partial"
        
        has_fvg = False
        if idx + 2 < len(recent):
            has_fvg = bear_fvg.iloc[idx+1] or bear_fvg.iloc[idx+2]

        return f"Bearish OB{'+FVG' if has_fvg else ''} ({mitigation_status})"

    return "None"

# =========================
# OOP SYSTEM
# =========================
class CryptoAnalyzer:
    def __init__(self, symbol, tf):
        self.symbol = self._normalize_symbol(symbol)
        self.tf = tf
        if tf not in MTF_MAP:
            sys.exit(f"[ERROR] TF tidak valid. Pilih: {', '.join(MTF_MAP.keys())}")
            
        self.htf, self.btf, self.ltf = MTF_MAP[tf]
        self.exchange = ccxt.binance({"enableRateLimit": True})
        
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.data = {}

    def _normalize_symbol(self, sym):
        sym = sym.upper().replace(" ", "")
        if "/" in sym: return sym
        if sym.endswith("USDT"): return sym[:-4] + "/USDT"
        return sym + "/USDT"

    def fetch_ohlcv(self, timeframe, limit=500):
        """[IMPROVED] Caching dengan TTL (60-1800 detik tergantung TF)"""
        cache_file = os.path.join(self.cache_dir, f"{self.symbol.replace('/', '_')}_{timeframe}_{limit}.pkl")
        
        if os.path.exists(cache_file):
            try:
                mtime = os.path.getmtime(cache_file)
                age = time.time() - mtime
                
                ttl_map = {
                    "1m": 60, "5m": 60, "15m": 120,
                    "1h": 180, "4h": 300, "1d": 600, "1w": 1800
                }
                ttl = ttl_map.get(timeframe, 120)
                
                if age < ttl:
                    df = pd.read_pickle(cache_file)
                    print(f"[CACHE] Loaded {timeframe} from cache")
                    return df
                else:
                    print(f"[CACHE] Cache expired for {timeframe}, fetching fresh...")
            except Exception as e:
                print(f"[WARNING] Cache corrupted ({timeframe}), fetching fresh...")

        for attempt in range(3):
            try:
                raw = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
                if not raw: raise ValueError("Data kosong dari exchange")

                df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
                
                df.to_pickle(cache_file)
                print(f"[CACHE] Saved {timeframe} to cache")
                return df
            except Exception as e:
                print(f"[WARNING] Error fetching {timeframe}: {e} | retry ({attempt+1}/3)")
                time.sleep(2)
        sys.exit(f"\n[ERROR] Gagal fetch data {timeframe}. Cek koneksi.")

    def load_data(self):
        print(f"\n[INFO] Menggunakan symbol: {self.symbol}")
        print("[INFO] Fetching MTF data...\n")
        self.data["df_high"] = compute_indicators(self.fetch_ohlcv(self.htf, 200))
        self.data["df_base"] = compute_indicators(self.fetch_ohlcv(self.btf, 500))
        self.data["df_low"]  = compute_indicators(self.fetch_ohlcv(self.ltf, 200))

    def analyze(self):
        df = self.data["df_base"]
        htf = self.data["df_high"]
        ltf = self.data["df_low"]

        struct = structure(df)
        ltf_struct = structure(ltf)
        ob = order_block_fvg(df)

        sc = self.calculate_score(df, htf, ltf_struct)
        setup = self.trade_setup(df, struct)
        act = self.decision(sc, setup["valid"])
        
        self.render(df, htf, struct, ltf_struct, ob, setup, sc, act)

    def trade_setup(self, df, struct):
        last = df.iloc[-1]
        atr_val = last['atr']
        entry = last['close']
        reason = []

        if pd.isna(atr_val) or atr_val <= 0:
            return {"valid": False, "entry": entry, "sl": entry, "tp": entry, "rr": 0.0, "reason": ["ATR tidak valid"]}

        sh_idx, sl_idx = swings(df, left=5, right=5)

        recent_swing_high = df['high'].iloc[sh_idx[-1]] if sh_idx else entry + (atr_val * 1.5)
        recent_swing_low = df['low'].iloc[sl_idx[-1]] if sl_idx else entry - (atr_val * 1.5)

        if "UP" in struct:
            sl = recent_swing_low - (atr_val * 0.2)
            if sl >= entry: sl = entry - atr_val 
            risk = entry - sl
            tp = entry + (risk * RISK_REWARD)
            reason.append("Trend UP (SL @ Recent Swing Low)")

        elif "DOWN" in struct:
            sl = recent_swing_high + (atr_val * 0.2)
            if sl <= entry: sl = entry + atr_val 
            risk = sl - entry
            tp = entry - (risk * RISK_REWARD)
            reason.append("Trend DOWN (SL @ Recent Swing High)")
        else:
            reason.append("Market RANGE (ATR Base)")
            return {"valid": False, "entry": entry, "sl": entry - atr_val, "tp": entry + atr_val, "rr": 1.0, "reason": reason}

        rr = abs((tp - entry) / (entry - sl)) if (entry - sl) != 0 else 0.0
        return {"valid": True, "entry": entry, "sl": sl, "tp": tp, "rr": rr, "reason": reason}

    def calculate_score(self, df_base, df_htf, ltf_struct):
        b = df_base.iloc[-1]
        prev = df_base.iloc[-2]
        h = df_htf.iloc[-1]
        sc = 0
        w = SCORE_WEIGHTS

        if b['close'] > b['ema9'] > b['ema21']: sc += w["ema_cross"]
        elif b['close'] < b['ema9'] < b['ema21']: sc -= w["ema_cross"]

        if b['close'] > b['ema50']: sc += w["ema50_trend"]
        elif b['close'] < b['ema50']: sc -= w["ema50_trend"]

        if b['close'] > b['ema50'] and "UP" in ltf_struct: sc += w["ltf_align"]
        elif b['close'] < b['ema50'] and "DOWN" in ltf_struct: sc -= w["ltf_align"]

        if b['rsi14'] > 55: sc += w["rsi_momentum"]
        elif b['rsi14'] < 45: sc -= w["rsi_momentum"]

        if b['macd_hist'] > 0: sc += w["macd_hist"]
        else: sc -= w["macd_hist"]

        if prev['stoch_k'] < prev['stoch_d'] and b['stoch_k'] > b['stoch_d'] and b['stoch_k'] < 20: sc += w["stoch_cross"]
        elif prev['stoch_k'] > prev['stoch_d'] and b['stoch_k'] < b['stoch_d'] and b['stoch_k'] > 80: sc -= w["stoch_cross"]

        if pd.notna(b['vol_ma']) and b['volume'] > b['vol_ma'] * 1.2: sc += w["vol_spike"]
        else: sc -= w["vol_flat"]

        if h['close'] > h['ema200']: sc += w["htf_ema200"]
        elif h['close'] < h['ema200']: sc -= w["htf_ema200"]

        return sc

    def decision(self, sc, valid):
        if not valid: return "WAIT"
        if sc >= 40: return "BUY"
        if sc <= -30: return "SELL"
        return "WAIT"

    def render(self, df, htf, struct, ltf_struct, ob, setup, sc, act):
        os.system('clear' if os.name != 'nt' else 'cls')
        last = df.iloc[-1]
        
        vol_sig = "HIGH VOLUME" if (pd.notna(last['vol_ma']) and last['volume'] > last['vol_ma'] * 1.5) else "NORMAL"
        
        reasons_extra = []
        if ob != "None": reasons_extra.append(f"Order Block Status: {ob}")
        if vol_sig == "HIGH VOLUME": reasons_extra.append("Volume Spike terdeteksi")
        if act != "WAIT": reasons_extra.append(f"LTF Validation: {ltf_struct}")

        print("\n" + "=" * 60)
        print(f"{self.symbol} | {self.btf} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        market_table = [
            ["Close", f"{last['close']:.4f}"],
            ["HTF Bias", "Uptrend" if htf.iloc[-1]['close'] > htf.iloc[-1]['ema200'] else "Downtrend"],
            ["LTF Bias", ltf_struct],
            ["Structure", struct],
            ["Order Block", ob],
            ["Volume", vol_sig],
        ]

        print("\n[ MARKET SNAPSHOT ]")
        print(tabulate(market_table, headers=["Item", "Value"], tablefmt="github"))

        indicator_table = [
            ["EMA50 / EMA200", f"{last['ema50']:.2f} / {last['ema200']:.2f}"],
            ["RSI14 / RSI7", f"{last['rsi14']:.2f} / {last['rsi7']:.2f}"],
            ["MACD Hist", f"{last['macd_hist']:.4f}"],
            ["Stoch K / D", f"{last['stoch_k']:.2f} / {last['stoch_d']:.2f}"],
            ["ATR", f"{last['atr']:.4f}"],
        ]

        print("\n[ INDICATORS ]")
        print(tabulate(indicator_table, headers=["Indikator", "Nilai"], tablefmt="github"))

        print("\n[ TRADE SETUP ]")
        setup_table = [
            ["Entry", f"{setup['entry']:.4f}"],
            ["StopLoss", f"{setup['sl']:.4f}"],
            ["TakeProfit", f"{setup['tp']:.4f}"],
            ["RR", f"{setup['rr']:.2f}"],
            ["Valid", "YES" if setup["valid"] else "NO"],
        ]
        print(tabulate(setup_table, headers=["Field", "Value"], tablefmt="github"))

        print("\nReason:")
        for r in setup["reason"] + reasons_extra:
            print(f"- {r}")

        print("\nScore :", sc)
        print("Action:", act)
        print("=" * 60)

# =========================
# MAIN
# =========================
def main():
    print("=" * 50)
    print("CRYPTO MTF + SMC ANALYZER (PRO)")
    print("=" * 50)
    print("Contoh input: BTC, BTCUSDT, ETH/USDT\n")

    symbol = input("Symbol: ")
    tf = input("TF (5m/15m/1h/4h): ").lower()

    analyzer = CryptoAnalyzer(symbol, tf)
    analyzer.load_data()
    analyzer.analyze()

if __name__ == "__main__":
    main()
