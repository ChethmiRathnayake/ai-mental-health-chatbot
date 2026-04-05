import numpy as np
import re
from typing import List, Dict, Any

_WORD = re.compile(r"[A-Za-z0-9']+")

def text_features(text: str) -> Dict[str, float]:
    tokens = _WORD.findall(text.lower())
    total = len(tokens)
    uniq = len(set(tokens))
    ttr = (uniq / total) if total else 0.0
    lex = ttr

    sents = re.split(r"[.!?]+", text.strip())
    sent_tokens = [_WORD.findall(s) for s in sents if s.strip()]
    avg_len = (sum(len(st) for st in sent_tokens) / len(sent_tokens)) if sent_tokens else 0.0
    synt = float(np.tanh(avg_len / 20.0))
    return {"ttr": ttr, "lexical_diversity": lex, "syntactic_complexity": synt}


def typing_features(keystrokes: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Accepts keystrokes with schema like:
      {"key": "a", "ts_ms": 123, "type": "..."}
    Supports 'type' values:
      - "char", "backspace" (your original)
      - "keydown", "keyup"  (common from web / RN)
    Uses key="Backspace" as backup signal.
    """
    if not keystrokes:
        return {"typing_speed": 0.0, "pause_count": 0.0, "error_rate": 0.0, "mean_iki_ms": 0.0}

    # timestamps
    ts = np.array([k.get("ts_ms", 0) for k in keystrokes], dtype=np.float64)
    ts = ts[ts >= 0]
    if ts.size < 2:
        return {"typing_speed": 0.0, "pause_count": 0.0, "error_rate": 0.0, "mean_iki_ms": 0.0}

    ts.sort()
    dt = np.diff(ts)

    mean_iki = float(np.mean(dt)) if dt.size else 0.0
    pauses = float(np.sum(dt > 1000.0)) if dt.size else 0.0

    # character count heuristic:
    # - if you emit explicit "char" events, use that
    # - otherwise, treat keydown events with single-char keys as chars
    chars = 0
    backs = 0
    for k in keystrokes:
        ktype = (k.get("type") or "").lower()
        key = k.get("key")

        # backspace detection
        if ktype == "backspace" or (isinstance(key, str) and key.lower() in ("backspace", "bksp")):
            backs += 1
            continue

        if ktype == "char":
            chars += 1
            continue

        # common event stream from web/RN: "keydown"
        if ktype in ("keydown", "keyup", "input"):
            # count single printable chars as "char"
            if isinstance(key, str) and len(key) == 1 and key.isprintable():
                chars += 1

    dur_min = (float(ts[-1] - ts[0]) / 60000.0) if ts[-1] > ts[0] else 1e-6
    wpm = (chars / 5.0) / dur_min if dur_min > 0 else 0.0

    err = float(backs / len(keystrokes)) if keystrokes else 0.0

    return {"typing_speed": float(wpm), "pause_count": pauses, "error_rate": err, "mean_iki_ms": mean_iki}


def build_x10(typing: Dict[str, float], text: Dict[str, float], voice: Dict[str, float]) -> np.ndarray:
    return np.array([
        typing["typing_speed"], typing["pause_count"], typing["error_rate"], typing["mean_iki_ms"],
        text["ttr"], text["lexical_diversity"], text["syntactic_complexity"],
        voice["pitch_variance"], voice["volume_fluctuation"], voice["tone_variability"],
    ], dtype=np.float32)