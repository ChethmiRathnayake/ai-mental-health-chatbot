from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class SignUp(BaseModel):
    email: str
    password: str

class SignIn(BaseModel):
    email: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class Keystroke(BaseModel):
    key: str
    ts_ms: int
    type: str


class VoiceFeatures(BaseModel):
    pitch_variance: float
    volume_fluctuation: float
    tone_variability: float


class ChatMessageIn(BaseModel):
    text: str
    keystrokes: Optional[List[Keystroke]] = None
    voice_features: Optional[VoiceFeatures] = None

class IngestRequest(BaseModel):
    text: str
    keystrokes: Optional[List[Keystroke]] = None
    voice_features: Optional[VoiceFeatures] = None

class PredictOut(BaseModel):
    ready: bool
    baseline_ready: bool
    window_size: int
    predicted_label: Optional[str] = None
    probs: Optional[Dict[str, float]] = None

class BaselineStatus(BaseModel):
    is_ready: bool
    n_samples: int

class ChatStartOut(BaseModel):
    session_id: int

class ChatMessageOut(BaseModel):
    reply: str
    session_id: int
    cognitive_load_ready: bool
    baseline_ready: bool
    window_size: int
    predicted_label: Optional[str] = None
    probs: Optional[Dict[str, float]] = None