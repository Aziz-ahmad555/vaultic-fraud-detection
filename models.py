# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    is_frozen = db.Column(db.Boolean, default=False, nullable=False)

class Transaction(db.Model):
    __tablename__ = "transactions"
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    scored = db.Column(db.Boolean, default=False, nullable=False)
    scored_at = db.Column(db.DateTime, nullable=True)
    fraud_score = db.Column(db.Float, nullable=True)
    tx_type = db.Column(db.String(50), nullable=False)  # deposit / withdraw
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    device_id = db.Column(db.String(128), nullable=True)
    meta = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "amount": self.amount,
            "tx_type": self.tx_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "device_id": self.device_id,
            "meta": self.meta
        }

class Alert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    alert_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    # Nullable: daily-frequency anomaly alerts (see ml_model.analyze_daily_behavior)
    # are not tied to a single transaction, so this must allow NULL.
    transaction_id = db.Column(db.String(64), nullable=True)
    score = db.Column(db.Float, nullable=True)
    severity = db.Column(db.String(32), nullable=True)
    status = db.Column(db.String(32), default="Pending")
    message = db.Column(db.String(512), nullable=True)
    generated_at = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "transaction_id": self.transaction_id,
            "score": self.score,
            "severity": self.severity,
            "status": self.status,
            "message": self.message,
            "generated_at": self.generated_at.isoformat()
        }

class Label(db.Model):
    __tablename__ = "labels"
    id = db.Column(db.Integer, primary_key=True)
    label_id = db.Column(db.String(64), unique=True, nullable=False)
    alert_id = db.Column(db.String(64), db.ForeignKey('alerts.alert_id'), nullable=False)
    is_fraud = db.Column(db.Boolean, nullable=False)
    labeled_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    labeled_at = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc))
    notes = db.Column(db.Text, nullable=True)

class Dispute(db.Model):
    __tablename__ = "disputes"
    id = db.Column(db.Integer, primary_key=True)
    dispute_id = db.Column(db.String(64), unique=True, nullable=False)
    alert_id = db.Column(db.String(64), db.ForeignKey('alerts.alert_id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reason = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(32), default="Open")  # Open / Under Review / Resolved
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            "dispute_id": self.dispute_id,
            "alert_id": self.alert_id,
            "user_id": self.user_id,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }