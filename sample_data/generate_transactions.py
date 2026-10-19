import os
import random
from datetime import datetime, timedelta
import pandas as pd


def generate_synthetic_transactions(output_csv: str = "sample_data/customer_transactions.csv"):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    random.seed(42)

    reference_date = datetime(2026, 10, 1)
    records = []
    transaction_id = 10001

    # Generate 300 customer personas
    for cust_id in range(1, 301):
        customer_code = f"CUST-{cust_id:04d}"
        persona = random.choices(["CHAMPION", "LOYAL", "AT_RISK", "LOST"], weights=[0.2, 0.35, 0.25, 0.2])[0]

        if persona == "CHAMPION":
            order_count = random.randint(10, 25)
            min_days_ago = 1
            max_days_ago = 30
            spend_range = (150.0, 600.0)
        elif persona == "LOYAL":
            order_count = random.randint(5, 12)
            min_days_ago = 5
            max_days_ago = 70
            spend_range = (50.0, 200.0)
        elif persona == "AT_RISK":
            order_count = random.randint(4, 10)
            min_days_ago = 90
            max_days_ago = 250
            spend_range = (120.0, 450.0)
        else:  # LOST
            order_count = random.randint(1, 3)
            min_days_ago = 180
            max_days_ago = 360
            spend_range = (20.0, 80.0)

        for _ in range(order_count):
            days_ago = random.randint(min_days_ago, max_days_ago)
            order_date = (reference_date - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            amount = round(random.uniform(*spend_range), 2)
            records.append({
                "transaction_id": transaction_id,
                "customer_id": customer_code,
                "order_date": order_date,
                "amount": amount,
            })
            transaction_id += 1

    df = pd.DataFrame(records).sort_values(by="order_date")
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == "__main__":
    generate_synthetic_transactions()
    print("Synthetic transactions dataset generated successfully.")
