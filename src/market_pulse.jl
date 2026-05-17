# ─── MarketPulse — Market data decoding for time-series input ────────────────
#
# Provides domain-specific input parsing for feeding external signals into
# the reservoir. This module is optional: SparseBrain and EnsembleBrain accept
# any CuVector{Float32} input — MarketPulse is a convenience for financial
# time-series applications.

using CUDA

# ── MarketPulse packed struct layout (120 bytes from Rust) ────────────────────
# Canonical channel order: DNX(0), Quai(1), Qubic(2), Kaspa(3), Monero(4), Ocean(5), Verus(6)
# [0..8]    timestamp_ns: UInt64
# [8..16]   dnx:    (price_norm f32, volatility f32)  Ch 0
# [16..24]  quai:   (price_norm f32, volatility f32)  Ch 1
# [24..32]  qubic:  (price_norm f32, volatility f32)  Ch 2
# [32..40]  kaspa:  (price_norm f32, volatility f32)  Ch 3
# [40..48]  monero: (price_norm f32, volatility f32)  Ch 4
# [48..56]  ocean:  (price_norm f32, volatility f32)  Ch 5
# [56..64]  verus:  (price_norm f32, volatility f32)  Ch 6
# [64..68]  confidence_signal: f32
# [68..72]  coinglass_funding_rate: f32
# [72..76]  coinglass_liquidation_volume: f32
# [76..80]  dex_liquidity_delta: f32
# [80..84]  l3_order_imbalance: f32
# [84..88]  gpu_temp_c: f32
# [88..92]  gpu_power_w: f32
# [92..96]  gpu_util_pct: f32
# [96..100]  basys_uart_buffer_load: f32
# [100..104] dydx_oi_delta: f32
# [104..108] dydx_funding_rate: f32
# [108..112] qubic_tick_trace: f32
# [112..116] qubic_tick_rate: f32
# [116..120] qubic_epoch_progress: f32

struct MarketPulse
    timestamp_ns::UInt64
    # ── Asset ticks (Ch 0-6) ────────────────────────────────────────────
    dnx_price::Float32
    dnx_vol::Float32
    quai_price::Float32
    quai_vol::Float32
    qubic_price::Float32
    qubic_vol::Float32
    kaspa_price::Float32
    kaspa_vol::Float32
    monero_price::Float32
    monero_vol::Float32
    ocean_price::Float32
    ocean_vol::Float32
    verus_price::Float32
    verus_vol::Float32
    # ── Auxiliary / confidence signal ────────────────────────────────────
    confidence_signal::Float32
    # ── Institutional sensor slots (zeroed) ──────────────────────────────
    funding_rate::Float32           # zeroed (mining coins lack perp markets)
    liquidation_vol::Float32        # stress proxy
    liquidity_delta::Float32        # zeroed
    l3_order_imbalance::Float32     # zeroed
    # ── Hardware Proprioception ───────────────────────────────────────────
    gpu_temp_c::Float32
    gpu_power_w::Float32
    gpu_util_pct::Float32
    basys_buffer_load::Float32
    # ── dYdX v4 Key-Free Signals ─────────────────────────────────────────
    dydx_oi_delta::Float32          # dYdX BTC-USD OI normalised delta
    dydx_funding_rate::Float32      # dYdX BTC-USD next funding rate
end

"""
    decode_market_pulse(buf::Vector{UInt8}) -> MarketPulse

Zero-copy decode of the 120-byte packed struct from Rust.
Uses reinterpret to cast raw bytes directly to typed values.
Bytes [108..120] carry Qubic Global Computing Pulse fields; decoded but not
forwarded to the reservoir (available for future lobe integration).
"""
function decode_market_pulse(buf::Vector{UInt8})
    @assert length(buf) == 120 "Expected 120 bytes, got $(length(buf))"

    ts = reinterpret(UInt64, buf[1:8])[1]
    f = reinterpret(Float32, buf[9:108])  # 25 Float32 values (base market fields)

    MarketPulse(
        ts,
        f[1],  f[2],   # dnx   (Ch 0)
        f[3],  f[4],   # quai  (Ch 1)
        f[5],  f[6],   # qubic (Ch 2)
        f[7],  f[8],   # kaspa (Ch 3)
        f[9],  f[10],  # monero (Ch 4)
        f[11], f[12],  # ocean (Ch 5)
        f[13], f[14],  # verus (Ch 6)
        f[15],         # confidence_signal
        f[16],         # funding_rate (zeroed)
        f[17],         # liquidation_vol (stress proxy)
        f[18],         # liquidity_delta (zeroed)
        f[19],         # l3_order_imbalance (zeroed)
        f[20],         # gpu_temp_c
        f[21],         # gpu_power_w
        f[22],         # gpu_util_pct
        f[23],         # basys_buffer_load
        f[24],         # dydx_oi_delta
        f[25]          # dydx_funding_rate
    )
end

"""
    pulse_to_input(pulse::MarketPulse) -> Vector{Float32}

Convert a MarketPulse into a 14-element input vector for the reservoir.
Layout: [dnx_price, dnx_vol, quai_price, quai_vol, qubic_price, qubic_vol,
         kaspa_price, kaspa_vol, monero_price, monero_vol,
         ocean_price, ocean_vol, verus_price, verus_vol]
Canonical channel order: DNX(0), Quai(1), Qubic(2), Kaspa(3), Monero(4), Ocean(5), Verus(6)
"""
function pulse_to_input(pulse::MarketPulse)
    Float32[
        pulse.dnx_price,    pulse.dnx_vol,
        pulse.quai_price,   pulse.quai_vol,
        pulse.qubic_price,  pulse.qubic_vol,
        pulse.kaspa_price,  pulse.kaspa_vol,
        pulse.monero_price, pulse.monero_vol,
        pulse.ocean_price,  pulse.ocean_vol,
        pulse.verus_price,  pulse.verus_vol,
    ]
end