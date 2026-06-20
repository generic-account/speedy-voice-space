# speedy-voice-space
Real-time pitch and formant/resonance tracking, with praat, noise reduction, normalization, and 2D plots

[website here](https://generic-account.github.io/speedy-voice-space)

<img width="1726" height="994" alt="Screenshot 2026-06-20 at 6 41 03 PM" src="https://github.com/user-attachments/assets/30ce17a1-f3ac-4884-9ded-d94a2982d191" />

## Notes:
- Fairly sensitive to noise. Use in a quiet, private environment.
- If your space is noisy, you can turn on noise reduction, increase median ranges, or reduce exp decay params.
- The noise reduction can take a moment to start up sometimes.
- Tune RMS cutoff based on your room's noise level. Turning on noise reduction can lead to RMS changes, so you'll probably have to tune your cutoff
- Speak close to the microphone
- Higher window lengths will be less responsive but more accurate.

## Python version (old)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### Run

```bash
python3 ui.py
```
