"""Synthesize a vowel-SWITCH + pitch-glide fixture matching the app's real use
case: hold /u/ (gliding pitch), switch to /a/, hold. A good tracker must stay
smooth within each vowel yet FOLLOW the abrupt formant switch (not over-smooth).
"""
import numpy as np, soundfile as sf, os
SR = 48000
OUT = os.path.join(os.path.dirname(__file__), "..", "audio", "fixtures")

def resonant(f0_start, f0_end, formants, seconds):
    n = int(seconds * SR)
    f0 = np.linspace(f0_start, f0_end, n)
    phase = np.cumsum(f0) / SR
    # impulse train at gliding f0
    src = np.zeros(n)
    last = -1
    for i in range(n):
        k = int(phase[i])
        if k != last:
            src[i] = 1.0; last = k
    y = src
    for ff, bw in formants:
        r = np.exp(-np.pi * bw / SR); th = 2*np.pi*ff/SR
        a1 = 2*r*np.cos(th); a2 = -(r*r)
        out = np.zeros(n)
        for i in range(n):
            out[i] = y[i] + (a1*out[i-1] if i>=1 else 0) + (a2*out[i-2] if i>=2 else 0)
        y = out
    return y / (np.max(np.abs(y))+1e-9)

u = resonant(140, 170, [(350,60),(800,90),(2400,150)], 2.2)
a = resonant(170, 150, [(730,80),(1090,90),(2440,150)], 2.2)
sig = np.concatenate([u, a]).astype(np.float32) * 0.25
sf.write(os.path.join(OUT, "rec_switch.wav"), np.clip(sig,-1,1), SR, subtype="PCM_16")
print("wrote rec_switch.wav", len(sig)/SR, "s")
