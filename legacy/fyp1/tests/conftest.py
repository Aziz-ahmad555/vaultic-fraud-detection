import os
import tempfile
import importlib

import pytest


@pytest.fixture
def client():
    """
    Provides a Flask test client backed by a fresh, temporary SQLite
    database for each test -- never touches your real fraud_detection.db.
    """
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["FLASK_SECRET"] = "test-secret-key"
    os.environ.setdefault("MAIL_DEV_MODE", "1")

    # Import (or re-import) the app module fresh so it picks up the test DB.
    import app as app_module
    importlib.reload(app_module)

    app_module.app.config["TESTING"] = True
    app_module.app.config["WTF_CSRF_ENABLED"] = False

    with app_module.app.test_client() as test_client:
        with app_module.app.app_context():
            app_module.db.create_all()
            yield test_client
            app_module.db.session.remove()
            app_module.db.drop_all()

    os.close(db_fd)
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def registered_user(client):
    """Registers and returns a test user's credentials (does not log in)."""
    client.post("/register", data={
        "name": "Test User",
        "email": "testuser@example.com",
        "password": "testpass123",
    }, follow_redirects=True)
    return {"email": "testuser@example.com", "password": "testpass123"}


@pytest.fixture
def logged_in_client(client, registered_user):
    """Returns a test client that is already logged in as the test user."""
    client.post("/login", data={
        "email": registered_user["email"],
        "password": registered_user["password"],
    }, follow_redirects=True)
    return client
