import numpy as np
import torch
from sqlalchemy.orm import Session
from ..config import settings
from .. import models

ALPHA = 0.05

def compute_baseline_delta(x10: np.ndarray, baseline_mean10: np.ndarray | None) -> float:
    if baseline_mean10 is None:
        return 0.0
    return float(np.linalg.norm(x10 - baseline_mean10))

def update_baseline(db: Session, user_id: int, x10: np.ndarray):
    ub = db.query(models.UserBaseline).filter(models.UserBaseline.user_id == user_id).first()
    if not ub:
        ub = models.UserBaseline(user_id=user_id, is_ready=False, mean10_json={}, n_samples=0)
        db.add(ub)
        db.commit()
        db.refresh(ub)

    mean10 = np.array([ub.mean10_json.get(str(i), 0.0) for i in range(10)], dtype=np.float32) if ub.n_samples else None

    if mean10 is None:
        mean10 = x10.copy()
    else:
        mean10 = (1 - ALPHA) * mean10 + ALPHA * x10

    ub.n_samples += 1
    ub.is_ready = ub.n_samples >= 15  # baseline after 15 samples (tune)
    ub.mean10_json = {str(i): float(mean10[i]) for i in range(10)}
    db.commit()

def get_baseline_mean10(db: Session, user_id: int):
    ub = db.query(models.UserBaseline).filter(models.UserBaseline.user_id == user_id).first()
    if not ub or ub.n_samples == 0:
        return None, False, 0
    mean10 = np.array([ub.mean10_json.get(str(i), 0.0) for i in range(10)], dtype=np.float32)
    return mean10, ub.is_ready, ub.n_samples

def last_n_timesteps(db: Session, user_id: int, n: int):
    q = (db.query(models.InteractionTimestep)
         .filter(models.InteractionTimestep.user_id == user_id)
         .order_by(models.InteractionTimestep.created_at.desc())
         .limit(n))
    rows = list(reversed(q.all()))
    return rows

def softmax(logits: np.ndarray):
    e = np.exp(logits - np.max(logits))
    return e / (np.sum(e) + 1e-12)

def predict_if_ready(db: Session, user_id: int, model, scaler, inv_label_map, device: str):
    rows = last_n_timesteps(db, user_id, settings.SEQ_LEN)
    if len(rows) < settings.SEQ_LEN:
        return False, len(rows), None, None

    seq = np.array([[
        r.typing_speed, r.pause_count, r.error_rate, r.mean_iki_ms,
        r.ttr, r.lexical_diversity, r.syntactic_complexity,
        r.pitch_variance, r.volume_fluctuation, r.tone_variability, r.baseline_delta
    ] for r in rows], dtype=np.float32)

    seq_scaled = scaler.transform(seq)
    x = torch.tensor(seq_scaled).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()[0]
    p = softmax(logits)
    pred_id = int(np.argmax(p))
    pred_label = inv_label_map[pred_id]
    probs = {inv_label_map[i]: float(p[i]) for i in range(len(p))}
    return True, settings.SEQ_LEN, pred_label, probs