# app.py
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()  # loads FLASK_SECRET, SMTP_*, MAIL_DEV_MODE, FERNET_KEY, JWT_SECRET_KEY from a local .env file
# NOTE: this must run BEFORE crypto_utils is imported below, since crypto_utils
# reads FERNET_KEY from the environment at import time.

from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from flask_wtf import CSRFProtect
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from xhtml2pdf import pisa

from models import db, User, Transaction, Alert, Label, Dispute
import ml_model
from utils import send_email_alert, send_sms_alert
from crypto_utils import encrypt_device_id, decrypt_device_id

# --------------------------
# Flask app config
# --------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET")
if not app.secret_key:
    # Fail loudly in prod-like runs instead of silently using a guessable default.
    if os.environ.get("FLASK_ENV") == "production":
        raise RuntimeError("FLASK_SECRET must be set in the environment for production.")
    app.secret_key = "dev-only-insecure-key"

# Tests set DATABASE_URL to a temporary file; ignoring it would make the test
# suite's drop_all() wipe the real instance/fraud_detection.db.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", 'sqlite:///fraud_detection.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get("JWT_SECRET_KEY")
db.init_app(app)
csrf = CSRFProtect(app)
jwt = JWTManager(app)

# Make decrypt_device_id available inside Jinja templates as a filter, e.g.
# {{ txn.device_id | decrypt_device_id }} -- so any template that displays
# device_id can show the plaintext without app.py needing to touch templates.
app.jinja_env.filters['decrypt_device_id'] = decrypt_device_id

import stream_worker
stream_worker.start_worker(app)

# --------------------------
# DB initialization & fix
# --------------------------
def normalize_db_timestamps():
    """Convert any tz-naive timestamps in DB to UTC-aware (in-place)."""
    from datetime import timezone
    changed = 0
    # It is safe to iterate and update because dataset is small in dev.
    for t in Transaction.query.filter(Transaction.timestamp != None).all():
        if t.timestamp is not None and t.timestamp.tzinfo is None:
            t.timestamp = t.timestamp.replace(tzinfo=timezone.utc)
            changed += 1
    if changed:
        db.session.commit()
    return changed

with app.app_context():
    db.create_all()
    fixed = normalize_db_timestamps()
    if fixed:
        print(f"[startup] normalized {fixed} existing timestamps to UTC")

# --------------------------
# Helpers
# --------------------------
def require_login():
    if 'user_id' not in session:
        flash("Please login to continue.")
        return False
    return True

# --------------------------
# Routes
# --------------------------
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = (request.form.get('name') or "").strip()
        email = (request.form.get('email') or "").strip().lower()
        password_plain = request.form.get('password') or ""
        if not name or not email or not password_plain:
            flash("All fields are required.")
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for('register'))
        hashed = generate_password_hash(password_plain)
        user = User(name=name, email=email, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or "").strip().lower()
        password = request.form.get('password') or ""
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session.clear()
            session['user_id'] = user.id
            flash("Logged in.")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials.")
    return render_template('login.html')

from collections import Counter
import numpy as np
from datetime import timezone

@app.route('/dashboard')
def dashboard():
    if not require_login():
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    just_added = request.args.get('just_added', type=int)

    # Fetch all transactions
    txns = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.timestamp.desc()).limit(200).all()
    alerts = Alert.query.filter_by(user_id=user.id).order_by(Alert.generated_at.desc()).limit(50).all()

    # Ensure timestamps are tz-aware
    for t in txns:
        if t.timestamp and t.timestamp.tzinfo is None:
            t.timestamp = t.timestamp.replace(tzinfo=timezone.utc)
    for a in alerts:
        if a.generated_at and a.generated_at.tzinfo is None:
            a.generated_at = a.generated_at.replace(tzinfo=timezone.utc)

    # Separate deposits and withdrawals
    deposits = [t for t in txns if t.tx_type.lower() == "deposit"]
    withdraws = [t for t in txns if t.tx_type.lower() == "withdraw"]

    # Calculate totals
    total_deposits = sum(t.amount for t in deposits)
    total_withdraws = sum(t.amount for t in withdraws)

    # ----------------------------------------
    # 📊 Daily Summary Chart (Transaction Count Trend)
    # ----------------------------------------
    if txns:
        daily_counts = Counter([t.timestamp.date() for t in txns])
        dates_sorted = sorted(daily_counts.keys())
        counts = [daily_counts[d] for d in dates_sorted]

        mean_txn = np.mean(counts)
        std_txn = np.std(counts) if np.std(counts) > 0 else 1.0
        threshold = mean_txn + 2 * std_txn  # Anomaly threshold line
    else:
        dates_sorted, counts, threshold = [], [], 0

    return render_template(
        'dashboard.html',
        user=user,
        transactions=txns,
        deposits=deposits,
        withdraws=withdraws,
        total_deposits=total_deposits,
        total_withdraws=total_withdraws,
        alerts=alerts,
        chart_dates=[d.strftime("%Y-%m-%d") for d in dates_sorted],
        chart_counts=counts,
        threshold=threshold,
        just_added=just_added
    )


@app.route('/dashboard/report.pdf')
def dashboard_report_pdf():
    """
    Generates a downloadable PDF compliance report summarizing this user's
    transaction and alert activity -- satisfies the "PDF compliance report"
    deliverable from the proposal (admin dashboard section).
    """
    if not require_login():
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    all_txns = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.timestamp.desc()).all()
    all_alerts = Alert.query.filter_by(user_id=user.id).order_by(Alert.generated_at.desc()).all()

    total_deposits = sum(t.amount for t in all_txns if t.tx_type.lower() == "deposit")
    total_withdraws = sum(t.amount for t in all_txns if t.tx_type.lower() == "withdraw")

    severity_counts = {"High": 0, "Medium": 0, "Low": 0}
    status_counts = {"Pending": 0, "Resolved": 0}
    for a in all_alerts:
        if a.severity in severity_counts:
            severity_counts[a.severity] += 1
        if a.status in status_counts:
            status_counts[a.status] += 1

    generated_at = datetime.now(timezone.utc)

    html = render_template(
        'report_pdf.html',
        user=user,
        generated_at=generated_at,
        total_txns=len(all_txns),
        total_deposits=total_deposits,
        total_withdraws=total_withdraws,
        severity_counts=severity_counts,
        status_counts=status_counts,
        alerts=all_alerts[:50],
    )

    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf_buffer)
    if pisa_status.err:
        flash("Error generating PDF report.", "danger")
        return redirect(url_for('dashboard'))

    pdf_buffer.seek(0)
    return Response(
        pdf_buffer.getvalue(),
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename=fraud_report_user{user.id}.pdf'}
    )


@app.route('/train_model')
def train_model():
    if not require_login():
        return redirect(url_for('login'))
    user_id = session['user_id']
    res = ml_model.train_user_model(user_id)
    flash(f"Model training result: {res}", "info")
    return redirect(url_for('dashboard'))
@app.route('/generate_fake', methods=['GET', 'POST'])
def generate_fake():
    if not require_login():
        return redirect(url_for('login'))

    summary = None
    if request.method == 'POST':
        try:
            user_id = session['user_id']
            num_transactions = int(request.form['num_transactions'])
            start_date_str = request.form['start_date']
            end_date_str = request.form['end_date']
            deposit_min = float(request.form['deposit_min'])
            deposit_max = float(request.form['deposit_max'])
            withdraw_min = float(request.form['withdraw_min'])
            withdraw_max = float(request.form['withdraw_max'])

            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            from generate_fake_transactions import generate_fake_transactions
            transactions = generate_fake_transactions(
                user_id=user_id,
                num_transactions=num_transactions,
                start_date=start_date,
                end_date=end_date,
                deposit_range=(deposit_min, deposit_max),
                withdraw_range=(withdraw_min, withdraw_max)
            )

            db.session.add_all(transactions)
            db.session.commit()

            # Count deposit vs withdraw
            deposits = len([t for t in transactions if t.tx_type == "deposit"])
            withdrawals = len(transactions) - deposits

            summary = {"deposits": deposits, "withdrawals": withdrawals}
            flash(f"{len(transactions)} fake transactions generated successfully!", "success")
        except Exception as e:
            flash(f"Error: {e}", "danger")

    return render_template('generate_fake.html', summary=summary)


@app.route('/add_transaction', methods=['GET', 'POST'])
def add_transaction():
    if not require_login():
        return redirect(url_for('login'))

    user_id = session['user_id']

    # 🔒 Step 0: Block all transactions if the user has frozen their account
    current_user = User.query.get(user_id)
    if current_user and current_user.is_frozen:
        flash("🔒 Your account is frozen. Unfreeze it from Account Settings to continue.", "danger")
        return redirect(url_for('dashboard'))

    # 🚨 Step 1: Check for pending alerts before allowing new transaction
    pending_alerts = Alert.query.filter_by(user_id=user_id, status="Pending").count()
    if pending_alerts >= 2:
        flash("🚫 You have multiple suspicious transactions pending investigation. "
              "No further transactions are allowed until alerts are resolved.", "danger")
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            amount = float(request.form.get('amount') or 0.0)
            tx_type = request.form.get('type') or "withdraw"
            device_id = (request.form.get('device_id') or "").strip()
            timestamp_str = request.form['timestamp']

            # Convert HTML datetime-local → UTC datetime
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M')
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            except Exception as e:
                flash(f"Invalid timestamp format: {e}", "danger")
                return redirect(url_for('add_transaction'))

        except Exception:
            flash("Invalid input.", "danger")
            return redirect(url_for('add_transaction'))

        # ✅ Step 2: Create and save transaction
        # device_id is encrypted at rest (Fernet, crypto_utils.py) -- the raw
        # value never touches the DB. It's decrypted again inside
        # stream_worker before scoring, and can be decrypted for display in
        # templates via the `decrypt_device_id` Jinja filter registered above.
        txn_id = str(uuid.uuid4())
        txn = Transaction(
            transaction_id=txn_id,
            user_id=user_id,
            amount=amount,
            tx_type=tx_type,
            timestamp=timestamp,
            device_id=encrypt_device_id(device_id)
        )
        db.session.add(txn)
        db.session.commit()

        # ✅ Step 3: Hand off to background worker — scoring, alerting, and
        # daily-anomaly checks now all happen asynchronously (see stream_worker.py)
        stream_worker.enqueue_transaction(txn.id)
        flash("Transaction received — scoring in progress.", "info")

        return redirect(url_for('dashboard', just_added=txn.id))

    # ✅ GET → render transaction form
    return render_template('add_transaction.html')


@app.route('/transaction_status/<int:transaction_pk>')
def transaction_status(transaction_pk):
    if not require_login():
        return {'error': 'unauthorized'}, 401

    txn = Transaction.query.get(transaction_pk)
    if not txn or txn.user_id != session['user_id']:
        return {'error': 'not found'}, 404

    return {
        'scored': bool(txn.scored),
        'fraud_score': txn.fraud_score
    }


# --------------------------
# JSON API (JWT-protected) — Phase 6 deliverable
# --------------------------
# Proves token-based auth is understood, separate from the app's normal
# session-cookie login. The rest of the app stays session-based; this is a
# proof-of-concept API layer as scoped in the roadmap.

@csrf.exempt
@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or "").strip().lower()
    password = data.get('password') or ""

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return {'error': 'invalid credentials'}, 401

    access_token = create_access_token(identity=str(user.id))
    return {'access_token': access_token}, 200


@csrf.exempt
@app.route('/api/score', methods=['POST'])
@jwt_required()
def api_score():
    user_id = int(get_jwt_identity())
    data = request.get_json(silent=True) or {}

    try:
        amount = float(data.get('amount', 0.0))
        tx_type = data.get('tx_type', 'withdraw')
        device_id = (data.get('device_id') or "").strip()
        timestamp_str = data.get('timestamp')
        timestamp = (
            datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
            if timestamp_str else datetime.now(timezone.utc)
        )
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
    except Exception as e:
        return {'error': f'invalid input: {e}'}, 400

    txn_payload = {"amount": amount, "timestamp": timestamp, "device_id": device_id}
    score_res = ml_model.score_transaction(user_id, txn_payload)

    return {
        'user_id': user_id,
        'tx_type': tx_type,
        'score': score_res.get('score'),
        'is_anomaly': score_res.get('is_anomaly'),
        'severity': ml_model.severity_band(score_res.get('score', 0.0)) if score_res.get('score') is not None else None
    }, 200


@app.route('/resolve_alert/<alert_id>', methods=['POST'])
def resolve_alert(alert_id):
    if not require_login():
        return redirect(url_for('login'))

    alert = Alert.query.filter_by(alert_id=alert_id).first()
    if alert and alert.user_id == session['user_id']:
        alert.status = "Resolved"
        db.session.commit()
        flash("✅ Alert resolved successfully.", "success")

    return redirect(url_for('dashboard'))


@app.route('/alert/<alert_id>')
def alert_detail(alert_id):
    if not require_login():
        return redirect(url_for('login'))

    alert = Alert.query.filter_by(alert_id=alert_id).first()
    if not alert or alert.user_id != session['user_id']:
        flash("Alert not found.", "danger")
        return redirect(url_for('dashboard'))

    user = User.query.get(session['user_id'])
    label = Label.query.filter_by(alert_id=alert_id).first()
    return render_template('alert_detail.html', alert=alert, user=user, label=label)


@app.route('/alert/<alert_id>/feedback', methods=['POST'])
def alert_feedback(alert_id):
    """
    Analyst feedback loop: lets the user/analyst confirm whether a flagged
    transaction was actually fraud. This label is stored and folded back into
    the next training run (see ml_model.adaptive_contamination) so the model
    improves over time instead of being static.
    """
    if not require_login():
        return redirect(url_for('login'))

    alert = Alert.query.filter_by(alert_id=alert_id).first()
    if not alert or alert.user_id != session['user_id']:
        flash("Alert not found.", "danger")
        return redirect(url_for('dashboard'))

    decision = request.form.get('decision')  # "fraud" or "not_fraud"
    is_fraud = (decision == "fraud")

    existing = Label.query.filter_by(alert_id=alert_id).first()
    if existing:
        existing.is_fraud = is_fraud
        existing.notes = request.form.get('notes', '')
    else:
        label = Label(
            label_id=str(uuid.uuid4()),
            alert_id=alert_id,
            is_fraud=is_fraud,
            labeled_by=session['user_id'],
            notes=request.form.get('notes', '')
        )
        db.session.add(label)

    alert.status = "Resolved"
    db.session.commit()

    flash(
        "✅ Marked as confirmed fraud — model will weight this more heavily next training run."
        if is_fraud else
        "✅ Marked as false positive — noted for future model tuning.",
        "success"
    )
    return redirect(url_for('dashboard'))




@app.route('/train', methods=['POST'])
def train():
    if not require_login():
        return redirect(url_for('login'))
    user_id = session['user_id']
    res = ml_model.train_user_model(user_id)
    flash(f"Training result: {res}")
    return redirect(url_for('dashboard'))


@app.route('/my_alerts')
def my_alerts():
    """
    Customer-facing view of the user's own alerts — simpler than the analyst
    dashboard, framed as the responsive-web MVP of the customer portal.
    """
    if not require_login():
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    alerts = Alert.query.filter_by(user_id=user.id).order_by(Alert.generated_at.desc()).all()

    for a in alerts:
        if a.generated_at and a.generated_at.tzinfo is None:
            a.generated_at = a.generated_at.replace(tzinfo=timezone.utc)

    return render_template('my_alerts.html', user=user, alerts=alerts)


@app.route('/dispute/<alert_id>', methods=['GET', 'POST'])
def dispute(alert_id):
    """
    Lets the user submit a dispute reason for a flagged alert. Stored as a
    Dispute row (id, alert_id, reason, status, created_at) — directly maps
    to a proposal deliverable without needing a separate customer app.
    """
    if not require_login():
        return redirect(url_for('login'))

    alert = Alert.query.filter_by(alert_id=alert_id).first()
    if not alert or alert.user_id != session['user_id']:
        flash("Alert not found.", "danger")
        return redirect(url_for('my_alerts'))

    existing = Dispute.query.filter_by(alert_id=alert_id).first()

    if request.method == 'POST':
        if existing:
            flash("A dispute has already been filed for this alert.", "warning")
            return redirect(url_for('dispute', alert_id=alert_id))

        reason = (request.form.get('reason') or "").strip()
        if not reason:
            flash("Please provide a reason for the dispute.", "danger")
            return redirect(url_for('dispute', alert_id=alert_id))

        new_dispute = Dispute(
            dispute_id=str(uuid.uuid4()),
            alert_id=alert_id,
            user_id=session['user_id'],
            reason=reason,
            status="Open"
        )
        db.session.add(new_dispute)
        db.session.commit()
        flash("✅ Dispute submitted. Our team will review it shortly.", "success")
        return redirect(url_for('my_alerts'))

    return render_template('dispute.html', alert=alert, existing=existing)


@app.route('/account/freeze', methods=['GET', 'POST'])
def account_freeze():
    """
    Self-service account freeze — a real feature named explicitly in the
    proposal. When frozen, add_transaction blocks new transactions.
    """
    if not require_login():
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        user.is_frozen = not user.is_frozen
        db.session.commit()
        flash(
            "🔒 Your account is now frozen. New transactions are blocked."
            if user.is_frozen else
            "🔓 Your account has been unfrozen. Transactions are allowed again.",
            "success"
        )
        return redirect(url_for('account_freeze'))

    return render_template('account_freeze.html', user=user)


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for('login'))

# Optional dev route: force-normalize DB timestamps (dev only)
@app.route('/_dev/normalize_timestamps')
def dev_normalize():
    # WARNING: keep this protected or remove in production
    if not require_login():
        return redirect(url_for('login'))
    changed = normalize_db_timestamps()
    return f"Normalized {changed} timestamps."

if __name__ == '__main__':
    # run with `python app.py`
    app.run(debug=True, port=5000)
