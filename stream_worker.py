# stream_worker.py
import queue
import threading
import time
import uuid
from datetime import datetime, timezone

_task_queue = queue.Queue()
_worker_started = False
_lock = threading.Lock()

def enqueue_transaction(transaction_pk):
    """Called from the request thread — just pushes the primary key, returns instantly."""
    _task_queue.put(transaction_pk)

def _worker_loop(app):
    from models import db, Transaction, Alert, User
    import ml_model
    from utils import send_email_alert, send_sms_alert
    from crypto_utils import decrypt_device_id

    while True:
        transaction_pk = _task_queue.get()
        start = time.time()
        with app.app_context():
            txn = Transaction.query.get(transaction_pk)
            if txn is None:
                _task_queue.task_done()
                continue
            try:
                user_id = txn.user_id
                amount = txn.amount
                tx_type = txn.tx_type
                device_id = decrypt_device_id(txn.device_id)
                timestamp = txn.timestamp
                txn_id = txn.transaction_id

                tx_count = Transaction.query.filter_by(user_id=user_id).count()
                if tx_count >= 10:
                    ml_model.train_user_model(user_id)

                txn_payload = {
                    "amount": amount,
                    "timestamp": timestamp,
                    "device_id": device_id,
                    "transaction_id": txn_id,
                }
                score_res = ml_model.score_transaction(user_id, txn_payload)

                if score_res.get("score") is not None:
                    score = float(score_res.get("score", 0.0))
                    is_anom = bool(score_res.get("is_anomaly", False))
                    severity = ml_model.severity_band(score)

                    txn.fraud_score = score

                    if is_anom and severity in ("Medium", "High"):
                        rule_reason = score_res.get("rule_reason")
                        base_message = f"Suspicious transaction: Pkr.{amount:.2f} (score={score:.2f})"
                        message = f"{base_message} — {rule_reason}" if rule_reason else base_message

                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            user_id=user_id,
                            transaction_id=txn_id,
                            score=score,
                            severity=severity,
                            status="Pending",
                            message=message
                        )
                        db.session.add(alert)

                        user_obj = User.query.get(user_id)
                        sent, detail = send_email_alert(
                            to_email=user_obj.email if user_obj else None,
                            subject=f"[Vaultic] {severity} severity fraud alert on your account",
                            body=(
                                f"A {severity.lower()}-severity anomaly was detected on your account.\n"
                                f"Amount: Pkr.{amount:.2f}\nType: {tx_type}\nScore: {score:.2f}\n\n"
                                f"Log in to Vaultic to review and resolve this alert."
                            ),
                        )
                        if not sent:
                            app.logger.warning(f"Email alert failed for user {user_id}: {detail}")

                        if severity == "High":
                            sms_sent, sms_detail = send_sms_alert(
                                body=(
                                    f"[Vaultic] HIGH severity fraud alert: Pkr.{amount:.2f} "
                                    f"({tx_type}, score={score:.2f}). Log in to review."
                                )
                            )
                            if not sms_sent:
                                app.logger.warning(f"SMS alert failed for user {user_id}: {sms_detail}")

                day_check = ml_model.analyze_daily_behavior(user_id)
                if day_check.get("status") == "anomaly":
                    print(f"[stream_worker] daily anomaly for user {user_id}: "
                          f"{day_check.get('count')} txns, z={day_check.get('z_score'):.2f}")

                txn.scored = True
                txn.scored_at = datetime.now(timezone.utc)
                db.session.commit()

                latency_ms = (time.time() - start) * 1000
                with open('scoring_latency.log', 'a') as f:
                    f.write(f"{datetime.now(timezone.utc).isoformat()},{transaction_pk},{latency_ms:.2f}\n")

            except Exception as e:
                print(f"[stream_worker] error scoring transaction {transaction_pk}: {e}")
                db.session.rollback()
        _task_queue.task_done()

def start_worker(app):
    global _worker_started
    with _lock:
        if _worker_started:
            return
        t = threading.Thread(target=_worker_loop, args=(app,), daemon=True)
        t.start()
        _worker_started = True
