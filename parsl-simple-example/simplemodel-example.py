import os
import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.data_provider.files import File


parsl.set_stream_logger(level=20)  # 20 = logging.INFO

# 1. Configure Parsl for parallel local execution across 4 workers
config = Config(
    executors=[
        HighThroughputExecutor(
            label="etl_workers",
            max_workers_per_node=4,
            provider=LocalProvider(init_blocks=1, max_blocks=1),
        )
    ]
)

# 2. STAGE 1 (Extract/Simulate): Generate messy raw data partitions in parallel
@python_app
def create_messy_partition(region: str, outputs: list = ()) -> str:
    import csv
    import random

    raw_rows = [
        ["TX-1001", "  ALICE@example.com ", "$45.50", "2026-11-01", region],
        ["TX-1002", "bob.smith@domain.org", " 120.00 USD", "2026-11-01", region],
        ["TX-1003", "Charlie.D@Test.co  ", "$15.75", "2026-11-02", region],
        ["TX-1004", "diana_prince@web.net", "89.99", "2026-11-03", region],
        ["TX-1001", "  ALICE@example.com ", "$45.50", "2026-11-01", region],
        ["TX-1005", "invalid-email-format", "$32.00", "2026-11-02", region],
        ["TX-1006", "eve@example.com", "-$15.00", "2026-11-03", region],
        ["TX-1007", "", "$55.00", "2026-11-04", region],
        ["TX-1008", "frank@example.com", "NULL", "2026-11-04", region],
    ]

    for i in range(1010, 1025):
        if random.random() < 0.8:
            price = f"${round(random.uniform(10, 250), 2)}"
            email = f"user{i}@shop.com"
        else:
            price = "CORRUPTED"
            email = "bad_email"
        raw_rows.append([f"TX-{i}", email, price, "2026-11-05", region])

    with open(outputs[0].filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tx_id", "customer_email", "amount", "date", "region"])
        writer.writerows(raw_rows)

    return region


# 3. STAGE 2 (Transform/Clean): Parallel data scrubbing per partition
@python_app
def clean_partition(inputs: list = (), outputs: list = ()) -> dict:
    import csv
    import re

    input_path = inputs[0].filepath
    output_path = outputs[0].filepath

    seen_tx_ids = set()
    cleaned_rows = []

    stats = {
        "region": "Unknown",
        "raw_count": 0,
        "clean_count": 0,
        "dropped_duplicates": 0,
        "dropped_bad_email": 0,
        "dropped_bad_amount": 0,
        "clean_revenue": 0.0,
    }

    email_regex = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

    with open(input_path, mode="r", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            stats["raw_count"] += 1
            stats["region"] = row["region"].strip().upper()

            tx_id = row["tx_id"].strip().upper()
            if tx_id in seen_tx_ids:
                stats["dropped_duplicates"] += 1
                continue

            email = row["customer_email"].strip().lower()
            if not email or not email_regex.match(email):
                stats["dropped_bad_email"] += 1
                continue

            raw_amt = row["amount"].replace("$", "").replace("USD", "").strip()
            try:
                amount = round(float(raw_amt), 2)
                if amount <= 0:
                    raise ValueError("Non-positive transaction amount")
            except ValueError:
                stats["dropped_bad_amount"] += 1
                continue

            seen_tx_ids.add(tx_id)
            stats["clean_count"] += 1
            stats["clean_revenue"] += amount

            cleaned_rows.append({
                "tx_id": tx_id,
                "customer_email": email,
                "amount_usd": f"{amount:.2f}",
                "date": row["date"].strip(),
                "region": stats["region"],
            })

    with open(output_path, mode="w", newline="") as outfile:
        fieldnames = ["tx_id", "customer_email", "amount_usd", "date", "region"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    stats["clean_revenue"] = round(stats["clean_revenue"], 2)
    return stats


# 4. STAGE 3 (Load/Merge): Unpack `*audit_stats` as positional args so Parsl
#    resolves every AppFuture BEFORE serializing with dill!
@python_app
def merge_and_audit(*audit_stats, inputs: list = (), outputs: list = ()) -> dict:
    import csv

    master_output = outputs[0].filepath
    fieldnames = ["tx_id", "customer_email", "amount_usd", "date", "region"]

    with open(master_output, mode="w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for clean_file in inputs:
            with open(clean_file.filepath, mode="r", newline="") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)

    return {
        "total_raw_rows": sum(s["raw_count"] for s in audit_stats),
        "total_clean_rows": sum(s["clean_count"] for s in audit_stats),
        "total_duplicates_removed": sum(s["dropped_duplicates"] for s in audit_stats),
        "total_bad_emails_removed": sum(s["dropped_bad_email"] for s in audit_stats),
        "total_bad_amounts_removed": sum(s["dropped_bad_amount"] for s in audit_stats),
        "total_verified_revenue": round(sum(s["clean_revenue"] for s in audit_stats), 2),
        "regional_breakdown": list(audit_stats),
    }


# 5. Orchestrate the Pipeline DAG
if __name__ == "__main__":
    # Load config inside `__main__` (crucial on macOS/Python 3.13 spawn multiprocessing)
    parsl.load(config)

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/clean", exist_ok=True)

    regions = ["north", "south", "east", "west", "central", "international"]
    clean_file_futures = []
    audit_stat_futures = []

    print(f"Dispatching parallel ETL pipeline for {len(regions)} regional partitions...")

    for reg in regions:
        raw_file = File(os.path.join(os.getcwd(), "data", "raw", f"raw_{reg}.csv"))
        clean_file = File(os.path.join(os.getcwd(), "data", "clean", f"clean_{reg}.csv"))

        raw_task = create_messy_partition(region=reg, outputs=[raw_file])

        clean_task = clean_partition(
            inputs=[raw_task.outputs[0]],
            outputs=[clean_file]
        )

        clean_file_futures.append(clean_task.outputs[0])
        audit_stat_futures.append(clean_task)

    master_file = File(os.path.join(os.getcwd(), "data", "master_clean_transactions.csv"))

    # Notice `*audit_stat_futures` (unpacked) or passing via `inputs`:
    final_audit_future = merge_and_audit(
        *audit_stat_futures,
        inputs=clean_file_futures,
        outputs=[master_file]
    )

    audit_report = final_audit_future.result()

    print("\nDATA CLEANING & QUALITY AUDIT REPORT")
    print(f"Total Raw Rows Ingested      : {audit_report['total_raw_rows']}")
    print(f"Total Clean Rows Exported    : {audit_report['total_clean_rows']}")
    print(f"  - Duplicates Dropped       : {audit_report['total_duplicates_removed']}")
    print(f"  - Invalid Emails Dropped   : {audit_report['total_bad_emails_removed']}")
    print(f"  - Corrupt Amounts Dropped  : {audit_report['total_bad_amounts_removed']}")
    print(f"Total Verified Revenue       : ${audit_report['total_verified_revenue']:,.2f}")
    print(f"Master Dataset Saved To      : {master_file.filepath}")

    parsl.dfk().cleanup()