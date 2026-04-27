# 📈 Crypto MTF + SMC Analyzer (Pro)

Sebuah *command-line tool* berbasis Python untuk menganalisis market Cryptocurrency menggunakan kombinasi indikator Technical Analysis (TA) klasik dan Smart Money Concepts (SMC). 

Dirancang secara khusus agar **sangat ringan dan 100% kompatibel dengan environment mobile (Termux)** tanpa mengorbankan kedalaman analisis.

## ✨ Fitur Unggulan

- **Advanced SMC Logic**: 
  - Deteksi **Order Block (OB)** dan **Fair Value Gap (FVG)** yang difilter menggunakan *displacement strength* (ATR × 1.3) untuk menghindari sinyal palsu.
  - Sistem *Mitigation Tracking* mendetail (Untouched, Partial, Full Mitigated) berdasarkan interaksi harga dengan *wick* dan *close* candle.
- **Robust Market Structure**: Menggunakan deteksi *swing* dengan rasio 5/5 (kiri/kanan) untuk memetakan BOS (Break of Structure) dan CHOCH (Change of Character) dengan standar profesional, meminimalisir *noise* dari *Lower Timeframe*.
- **Multi-Timeframe (MTF) Confluence**: Validasi otomatis pergerakan harga lintas *timeframe* (HTF bias, Base TF structure, LTF alignment).
- **Termux-Optimized (Zero Heavy Libs)**: Kalkulasi indikator dibangun murni menggunakan vektorisasi **Pandas & NumPy**. Tidak memerlukan *dependencies* berbasis C yang rawan *crash* di Termux seperti `TA-Lib` atau `pandas_ta`.
- **Smart TTL Caching**: Dilengkapi dengan sistem *caching* `.pkl` dengan *Time-To-Live* (TTL) dinamis per *timeframe*. Menghemat kuota, mempercepat eksekusi, dan mencegah terkena *rate-limit* dari API Exchange.
- **Dynamic Risk Management**: Otomatis mengkalkulasi Entry, Stop Loss (berbasis ATR & Recent Swings), Take Profit, dan rasio Risk/Reward.

## ⚙️ Persyaratan Sistem

- Python 3.8+
- Koneksi internet aktif

Install *library* yang dibutuhkan:

```bash
pip install pandas numpy ccxt tabulate

Tribute to ChatGPT
