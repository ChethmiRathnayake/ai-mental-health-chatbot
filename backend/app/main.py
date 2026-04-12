from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
import torch
from fastapi.security import OAuth2PasswordRequestForm

from .db import get_db, Base, engine
from .config import settings
from . import models, schemas
from .auth import hash_password, verify_password, create_access_token, get_current_user
from .ml.model import load_artifact
from .ml.feature_engineering import typing_features, text_features, build_x10
from .ml.inference import get_baseline_mean10, compute_baseline_delta, update_baseline, predict_if_ready
from .llm_client import llm_chat, LLMError, simple_fallback_reply

from fastapi.middleware.wsgi import WSGIMiddleware
from Api import app as flask_app

app = FastAPI(title="Cognitive Load + Chat Backend")
app.mount("/AITMS", WSGIMiddleware(flask_app))

Base.metadata.create_all(bind=engine)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ml_model, scaler, feature_order, label_map, inv_label_map = load_artifact(settings.MODEL_PATH, device=DEVICE)


# -------------------------
# Auth
# -------------------------
@app.post("/auth/signup", response_model=schemas.TokenOut)
def signup(req: schemas.SignUp, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == req.email).first():
        raise HTTPException(400, "Email already registered")

    u = models.User(email=req.email, password_hash=hash_password(req.password))
    db.add(u)
    db.commit()
    db.refresh(u)

    b = models.UserBaseline(user_id=u.id, is_ready=False, mean10_json={}, n_samples=0)
    db.add(b)
    db.commit()

    return schemas.TokenOut(access_token=create_access_token(u.id))


@app.post("/auth/signin", response_model=schemas.TokenOut)
def signin(req: schemas.SignIn, db: Session = Depends(get_db)):
    u = db.query(models.User).filter(models.User.email == req.email).first()
    if not u or not verify_password(req.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return schemas.TokenOut(access_token=create_access_token(u.id))



@app.post("/auth/token", response_model=schemas.TokenOut)
def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Swagger sends "username" — you are using email as username
    u = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not u or not verify_password(form_data.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return schemas.TokenOut(access_token=create_access_token(u.id))


# -------------------------
# Baseline
# -------------------------
@app.get("/baseline/status", response_model=schemas.BaselineStatus)
def baseline_status(user=Depends(get_current_user), db: Session = Depends(get_db)):
    _, is_ready, n = get_baseline_mean10(db, user.id)
    return schemas.BaselineStatus(is_ready=is_ready, n_samples=n)


# -------------------------
# Ingest (kept)
# -------------------------
@app.post("/ingest", response_model=schemas.PredictOut)
def ingest(req: schemas.IngestRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    klist = [k.model_dump() for k in (req.keystrokes or [])]
    tfeat = typing_features(klist)
    xfeat = text_features(req.text)

    vfeat = (req.voice_features.model_dump() if req.voice_features else {
        "pitch_variance": 0.0,
        "volume_fluctuation": 0.0,
        "tone_variability": 0.0
    })

    # enforce: at least one modality present
    if (not klist) and (req.voice_features is None):
        raise HTTPException(422, "Provide at least one: keystrokes or voice_features.")

    x10 = build_x10(tfeat, xfeat, vfeat)

    mean10, baseline_ready, _ = get_baseline_mean10(db, user.id)
    bdelta = compute_baseline_delta(x10, mean10)

    row = models.InteractionTimestep(
        user_id=user.id,
        typing_speed=float(x10[0]),
        pause_count=float(x10[1]),
        error_rate=float(x10[2]),
        mean_iki_ms=float(x10[3]),
        ttr=float(x10[4]),
        lexical_diversity=float(x10[5]),
        syntactic_complexity=float(x10[6]),
        pitch_variance=float(x10[7]),
        volume_fluctuation=float(x10[8]),
        tone_variability=float(x10[9]),
        baseline_delta=float(bdelta),
    )
    db.add(row)
    db.commit()

    update_baseline(db, user.id, x10)

    ready, win, pred_label, probs = predict_if_ready(db, user.id, ml_model, scaler, inv_label_map, DEVICE)

    if ready and pred_label and probs:
        db.add(models.Prediction(user_id=user.id, predicted_label=pred_label, probs_json=probs))
        db.commit()

    _, baseline_ready, _ = get_baseline_mean10(db, user.id)

    return schemas.PredictOut(
        ready=ready,
        baseline_ready=baseline_ready,
        window_size=win,
        predicted_label=pred_label,
        probs=probs
    )


# -------------------------
# Chat helpers
# -------------------------
def _system_prompt(pred_label: str | None, baseline_ready: bool) -> str:
    base = (
        "You are a supportive wellbeing chatbot for university students. "
        "You are not a clinician. Do not claim diagnosis or treatment. "
        "If the user expresses self-harm intent or imminent danger, advise seeking immediate local emergency help or professional support. "
        "Keep responses practical, emotionally supportive, and avoid overconfident medical claims."
    )

    if not baseline_ready:
        return base + " Personalization baseline is still being learned; ask clarifying questions and avoid strong personalization."

    if pred_label == "High":
        return base + " Cognitive load is HIGH: keep it short, calming, one step at a time, avoid lists and complex options."
    if pred_label == "Medium":
        return base + " Cognitive load is MEDIUM: be structured, concise, ask 1-2 focused questions."
    if pred_label == "Low":
        return base + " Cognitive load is LOW: normal supportive conversation."
    return base


def _fetch_recent_chat(db: Session, session_id: int, limit: int):
    rows = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


# -------------------------
# Chat endpoints
# -------------------------
@app.post("/chat/start", response_model=schemas.ChatStartOut)
def chat_start(user=Depends(get_current_user), db: Session = Depends(get_db)):
    s = models.ChatSession(user_id=user.id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return schemas.ChatStartOut(session_id=s.id)


@app.post("/chat/message", response_model=schemas.ChatMessageOut)
def chat_message(session_id: int, req: schemas.ChatMessageIn, user=Depends(get_current_user), db: Session = Depends(get_db)):
    session = (
        db.query(models.ChatSession)
        .filter(models.ChatSession.id == session_id, models.ChatSession.user_id == user.id)
        .first()
    )
    if not session:
        raise HTTPException(404, "Session not found")

    # store user message
    db.add(models.ChatMessage(session_id=session_id, user_id=user.id, role="user", content=req.text))
    db.commit()

    # --- ingest features for this message ---
    klist = [k.model_dump() for k in (req.keystrokes or [])]
    tfeat = typing_features(klist)
    xfeat = text_features(req.text)

    vfeat = (req.voice_features.model_dump() if req.voice_features else {
        "pitch_variance": 0.0,
        "volume_fluctuation": 0.0,
        "tone_variability": 0.0
    })

    if (not klist) and (req.voice_features is None):
        raise HTTPException(422, "Provide at least one: keystrokes or voice_features.")

    x10 = build_x10(tfeat, xfeat, vfeat)

    mean10, baseline_ready, _ = get_baseline_mean10(db, user.id)
    bdelta = compute_baseline_delta(x10, mean10)

    db.add(models.InteractionTimestep(
        user_id=user.id,
        typing_speed=float(x10[0]),
        pause_count=float(x10[1]),
        error_rate=float(x10[2]),
        mean_iki_ms=float(x10[3]),
        ttr=float(x10[4]),
        lexical_diversity=float(x10[5]),
        syntactic_complexity=float(x10[6]),
        pitch_variance=float(x10[7]),
        volume_fluctuation=float(x10[8]),
        tone_variability=float(x10[9]),
        baseline_delta=float(bdelta),
    ))
    db.commit()

    update_baseline(db, user.id, x10)
    _, baseline_ready, _ = get_baseline_mean10(db, user.id)

    ready, win, pred_label, probs = predict_if_ready(db, user.id, ml_model, scaler, inv_label_map, DEVICE)

    if ready and pred_label and probs:
        db.add(models.Prediction(user_id=user.id, predicted_label=pred_label, probs_json=probs))
        db.commit()

    # --- build LLM messages from recent chat ---
    sys = _system_prompt(pred_label if ready else None, baseline_ready)
    history = _fetch_recent_chat(db, session_id, settings.CHAT_MAX_TURNS)

    messages = [{"role": "system", "content": sys}]
    for m in history:
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})

    # --- call OpenAI ---
    try:
        reply = llm_chat(messages)
    except LLMError:
        # don't break your pipeline if Ollama isn't running
        reply = simple_fallback_reply(pred_label if ready else None, baseline_ready)

    # store assistant message
    db.add(models.ChatMessage(session_id=session_id, user_id=user.id, role="assistant", content=reply))
    db.commit()

    return schemas.ChatMessageOut(
        reply=reply,
        session_id=session_id,
        cognitive_load_ready=ready,
        baseline_ready=baseline_ready,
        window_size=win,
        predicted_label=(pred_label if ready else None),
        probs=(probs if ready else None),
    )
