# crypto_utils.py
"""
Encryption-at-rest helper for sensitive fields (currently: Transaction.device_id).

Uses Fernet (symmetric, authenticated encryption) from the `cryptography`
package. This demonstrates the concept of encryption-at-rest on a real field
rather than claiming full AES-256 coverage across the system, which the
roadmap explicitly scopes down for FYP-1.

Note: Fernet output is non-deterministic (same input encrypts to a different
ciphertext each time), so encrypted device_id values cannot be directly
compared or grouped in SQL. Anywhere the app needs to reason about device_id
(e.g. ml_model scoring), decrypt first via decrypt_device_id().
"""
import os
from cryptography.fernet import Fernet, InvalidToken

_key = os.environ.get("FERNET_KEY")
if not _key:
    raise RuntimeError(
        "FERNET_KEY is not set in the environment. "
        "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
        "and add it to your .env file."
    )

_fernet = Fernet(_key.encode())


def encrypt_device_id(plaintext_device_id):
    """Encrypt a device_id string for storage. Returns None if input is None/empty."""
    if not plaintext_device_id:
        return plaintext_device_id
    return _fernet.encrypt(plaintext_device_id.encode()).decode()


def decrypt_device_id(encrypted_device_id):
    """
    Decrypt a stored device_id back to plaintext for use in scoring/display.
    Returns the original value unchanged if it isn't valid Fernet ciphertext
    (handles old, pre-encryption rows gracefully instead of crashing).
    """
    if not encrypted_device_id:
        return encrypted_device_id
    try:
        return _fernet.decrypt(encrypted_device_id.encode()).decode()
    except (InvalidToken, ValueError):
        # Not encrypted (e.g. old row from before this feature existed) —
        # return as-is rather than failing.
        return encrypted_device_id