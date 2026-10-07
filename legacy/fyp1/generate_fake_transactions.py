import random
from datetime import datetime, timedelta, timezone
from models import Transaction
import uuid

def generate_fake_transactions(user_id, num_transactions, start_date, end_date, deposit_range, withdraw_range):
    transactions = []
    delta = (end_date - start_date).days
    for _ in range(num_transactions):
        timestamp = start_date + timedelta(days=random.randint(0, delta),
                                           hours=random.randint(0, 23),
                                           minutes=random.randint(0, 59))
        timestamp = timestamp.replace(tzinfo=timezone.utc)

        tx_type = random.choice(["deposit", "withdraw"])
        if tx_type == "deposit":
            amount = round(random.uniform(*deposit_range), 2)
        else:
            amount = round(random.uniform(*withdraw_range), 2)

        txn = Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user_id,
            amount=amount,
            tx_type=tx_type,
            timestamp=timestamp,
            device_id=f"device_{random.randint(1, 5)}"
        )
        transactions.append(txn)

    return transactions
