from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Boolean, JSON, UniqueConstraint,Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .db import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    baseline: Mapped["UserBaseline"] = relationship(back_populates="user", uselist=False)

class UserBaseline(Base):
    __tablename__ = "user_baselines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    is_ready: Mapped[bool] = mapped_column(Boolean, default=False)

    # baseline mean over first 10 features (exclude baseline_delta)
    mean10_json: Mapped[dict] = mapped_column(JSON, default=dict)
    n_samples: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(back_populates="baseline")

class InteractionTimestep(Base):
    __tablename__ = "interaction_timesteps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # 11 features
    typing_speed: Mapped[float] = mapped_column(Float)
    pause_count: Mapped[float] = mapped_column(Float)
    error_rate: Mapped[float] = mapped_column(Float)
    mean_iki_ms: Mapped[float] = mapped_column(Float)
    ttr: Mapped[float] = mapped_column(Float)
    lexical_diversity: Mapped[float] = mapped_column(Float)
    syntactic_complexity: Mapped[float] = mapped_column(Float)
    pitch_variance: Mapped[float] = mapped_column(Float)
    volume_fluctuation: Mapped[float] = mapped_column(Float)
    tone_variability: Mapped[float] = mapped_column(Float)
    baseline_delta: Mapped[float] = mapped_column(Float)

class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predicted_label: Mapped[str] = mapped_column(String(16))
    probs_json: Mapped[dict] = mapped_column(JSON, default=dict)

    from sqlalchemy import Text
from datetime import datetime

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)