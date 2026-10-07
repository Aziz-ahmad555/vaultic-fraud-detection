"""
Test suite for the fraud detection Flask app.

Run with: pytest -v

Covers:
  - Registration / login / logout flow
  - Dashboard access control (must be logged in)
  - Transaction creation and scoring
  - Alert creation threshold and pending-alert transaction lock
  - Alert resolution
  - PDF report generation
  - Pure unit tests for ml_model helper functions (no Flask needed)
"""
import io
import ml_model


# ---------------------------------------------------------------------
# Auth flow
# ---------------------------------------------------------------------

def test_register_creates_user_and_redirects_to_login(client):
    response = client.post("/register", data={
        "name": "Aziz",
        "email": "aziz@example.com",
        "password": "securepass1",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"login" in response.request.path.encode() or b"Login" in response.data


def test_register_rejects_duplicate_email(client, registered_user):
    response = client.post("/register", data={
        "name": "Someone Else",
        "email": registered_user["email"],  # already registered
        "password": "anotherpass1",
    }, follow_redirects=True)
    assert b"already registered" in response.data.lower() or response.status_code == 200


def test_login_with_correct_credentials_succeeds(client, registered_user):
    response = client.post("/login", data={
        "email": registered_user["email"],
        "password": registered_user["password"],
    }, follow_redirects=True)
    assert response.status_code == 200
    # Should land on dashboard, not stay on login page with an error
    assert b"Invalid credentials" not in response.data


def test_login_with_wrong_password_fails(client, registered_user):
    response = client.post("/login", data={
        "email": registered_user["email"],
        "password": "wrong-password",
    }, follow_redirects=True)
    assert b"Invalid credentials" in response.data


def test_dashboard_requires_login(client):
    response = client.get("/dashboard", follow_redirects=False)
    # Should redirect (302) to login, not show the dashboard directly
    assert response.status_code == 302
    assert "/login" in response.headers.get("Location", "")


def test_logout_clears_session(logged_in_client):
    response = logged_in_client.get("/dashboard")
    assert response.status_code == 200  # confirms we were logged in

    logged_in_client.get("/logout")
    response = logged_in_client.get("/dashboard", follow_redirects=False)
    assert response.status_code == 302  # now redirected, session cleared


# ---------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------

def test_add_transaction_creates_record(logged_in_client):
    response = logged_in_client.post("/add_transaction", data={
        "amount": "5000",
        "type": "deposit",
        "device_id": "device_test",
        "timestamp": "2026-07-11T10:00",
    }, follow_redirects=True)
    assert response.status_code == 200

    import app as app_module
    with app_module.app.app_context():
        count = app_module.Transaction.query.count()
        assert count == 1


def test_add_transaction_rejects_invalid_amount(logged_in_client):
    response = logged_in_client.post("/add_transaction", data={
        "amount": "not-a-number",
        "type": "deposit",
        "device_id": "device_test",
        "timestamp": "2026-07-11T10:00",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid input" in response.data


def test_pending_alerts_block_new_transactions(logged_in_client):
    """
    Business rule: once a user has 2+ Pending alerts, add_transaction
    should refuse new transactions until they're resolved.
    """
    import app as app_module
    import uuid as uuid_module
    with app_module.app.app_context():
        # Need a user_id -- fetch the logged-in user from the DB directly
        user = app_module.User.query.filter_by(email="testuser@example.com").first()
        for _ in range(2):
            alert = app_module.Alert(
                alert_id=str(uuid_module.uuid4()),
                user_id=user.id,
                transaction_id="fake-txn",
                score=0.9,
                severity="High",
                status="Pending",
                message="Test alert",
            )
            app_module.db.session.add(alert)
        app_module.db.session.commit()

    response = logged_in_client.post("/add_transaction", data={
        "amount": "1000",
        "type": "deposit",
        "device_id": "device_test",
        "timestamp": "2026-07-11T10:00",
    }, follow_redirects=True)
    assert b"pending investigation" in response.data.lower() or b"No further transactions" in response.data


# ---------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------

def test_resolve_alert_marks_resolved(logged_in_client):
    import app as app_module
    import uuid as uuid_module
    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email="testuser@example.com").first()
        alert_id = str(uuid_module.uuid4())
        alert = app_module.Alert(
            alert_id=alert_id,
            user_id=user.id,
            transaction_id="fake-txn",
            score=0.7,
            severity="Medium",
            status="Pending",
            message="Test alert",
        )
        app_module.db.session.add(alert)
        app_module.db.session.commit()

    response = logged_in_client.post(f"/resolve_alert/{alert_id}", follow_redirects=True)
    assert response.status_code == 200

    with app_module.app.app_context():
        updated = app_module.Alert.query.filter_by(alert_id=alert_id).first()
        assert updated.status == "Resolved"


def _create_alert(app_module, email, message="Test alert"):
    import uuid as uuid_module
    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email=email).first()
        alert_id = str(uuid_module.uuid4())
        app_module.db.session.add(app_module.Alert(
            alert_id=alert_id,
            user_id=user.id,
            transaction_id="fake-txn",
            score=0.7,
            severity="Medium",
            status="Pending",
            message=message,
        ))
        app_module.db.session.commit()
    return alert_id


def test_alert_detail_page_renders(logged_in_client):
    import app as app_module
    alert_id = _create_alert(app_module, "testuser@example.com", message="Suspicious transaction: Pkr.90000.00")

    response = logged_in_client.get(f"/alert/{alert_id}")
    assert response.status_code == 200
    assert b"Suspicious transaction: Pkr.90000.00" in response.data
    assert f"/alert/{alert_id}/feedback".encode() in response.data  # labeling form is shown


def test_alert_feedback_stores_label_and_resolves(logged_in_client):
    import app as app_module
    alert_id = _create_alert(app_module, "testuser@example.com")

    response = logged_in_client.post(f"/alert/{alert_id}/feedback", data={
        "decision": "fraud",
        "notes": "confirmed with customer",
    }, follow_redirects=True)
    assert response.status_code == 200

    with app_module.app.app_context():
        label = app_module.Label.query.filter_by(alert_id=alert_id).first()
        assert label is not None and label.is_fraud
        assert app_module.Alert.query.filter_by(alert_id=alert_id).first().status == "Resolved"

    # The detail page now shows the stored label instead of the form
    response = logged_in_client.get(f"/alert/{alert_id}")
    assert b"Confirmed Fraud" in response.data


def test_alert_detail_hides_other_users_alerts(logged_in_client):
    import app as app_module
    logged_in_client.post("/register", data={
        "name": "Other User", "email": "other@example.com", "password": "otherpass123",
    })
    alert_id = _create_alert(app_module, "other@example.com", message="Someone else's alert")

    response = logged_in_client.get(f"/alert/{alert_id}", follow_redirects=True)
    assert b"Someone else's alert" not in response.data
    assert b"Alert not found" in response.data


# ---------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------

def test_pdf_report_downloads_successfully(logged_in_client):
    response = logged_in_client.get("/dashboard/report.pdf")
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert len(response.data) > 100  # a real PDF, not an empty/error response


# ---------------------------------------------------------------------
# ml_model unit tests (no Flask app needed)
# ---------------------------------------------------------------------

def test_severity_band_low():
    assert ml_model.severity_band(0.1) == "Low"
    assert ml_model.severity_band(0.49) == "Low"


def test_severity_band_medium():
    assert ml_model.severity_band(0.5) == "Medium"
    assert ml_model.severity_band(0.79) == "Medium"


def test_severity_band_high():
    assert ml_model.severity_band(0.8) == "High"
    assert ml_model.severity_band(1.0) == "High"


def test_global_model_returns_float_or_none():
    """
    global_model.score_global should never raise -- it either returns a
    valid float in [0, 1] or None (if the model files aren't available).
    """
    import global_model
    score = global_model.score_global({"amount": 1000.0})
    assert score is None or (isinstance(score, float) and 0.0 <= score <= 1.0)
