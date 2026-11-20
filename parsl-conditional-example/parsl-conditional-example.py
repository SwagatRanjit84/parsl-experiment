import os
import parsl

from parsl import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
parsl.set_stream_logger(level=20)  # 20 = logging.INFO

# 1. Configure Parsl

config = Config(
    executors=[
        HighThroughputExecutor(
            label="local_htex",
            max_workers_per_node=4,
            cores_per_worker=1,
            provider=LocalProvider(
                init_blocks=1,
                min_blocks=1,
                max_blocks=1
            )
        )
    ]
)

parsl.load(config)

# 2. Router App

@python_app
def router_task(input_path, high_input, low_input, threshold=30.0):

    with open(input_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_sum = 0.0

    for line in lines:
        total_sum += sum(float(x) for x in line.split(","))

    print(f"[Router] Computed sum = {total_sum}")
    print(f"[Router] Threshold = {threshold}")

    if total_sum > threshold:

        print("[Router] HIGH branch activated")

        with open(high_input, "w") as f:
            f.write("\n".join(lines) + "\n")

        with open(low_input, "w") as f:
            f.write("SKIP\n")

    else:

        print("[Router] LOW branch activated")

        with open(high_input, "w") as f:
            f.write("SKIP\n")

        with open(low_input, "w") as f:
            f.write("\n".join(lines) + "\n")

    return high_input, low_input

# 3. Branch Worker App

@python_app
def branch_task(mode, router_result, output_path):

    high_input, low_input = router_result

    if mode == "HIGH":
        input_path = high_input
    else:
        input_path = low_input

    with open(input_path, "r") as f:
        content = f.read().strip()

    if content == "SKIP":

        print(f"[{mode}] Received SKIP signal. Bypassing computation.")

        with open(output_path, "w") as f:
            f.write(f"{mode}: SKIPPED\n")

        return output_path

    print(f"[{mode}] Active branch! Processing data...")

    lines = content.splitlines()
    results = []

    for line in lines:

        vals = [float(x) for x in line.split(",")]

        if mode == "HIGH":
            new_vals = [v * 10.0 for v in vals]

        else:
            new_vals = [v + 100.0 for v in vals]

        results.append(",".join(map(str, new_vals)))

    with open(output_path, "w") as f:
        f.write("\n".join(results) + "\n")

    return output_path

# 4. Create Input File

with open("input_data.txt", "w") as f:
    f.write(
        "1.0,2.0,3.0\n"
        "4.0,5.0,6.0\n"
        "7.0,8.0,9.0\n"
    )

# 5. Define File Paths

file_in = os.path.abspath("input_data.txt")

high_in = os.path.abspath("high_branch_input.txt")
low_in = os.path.abspath("low_branch_input.txt")

high_result = os.path.abspath("high_branch_output.txt")
low_result = os.path.abspath("low_branch_output.txt")

# 6. Submit Workflow

future_router = router_task(
    file_in,
    high_in,
    low_in,
    threshold=30.0
)

future_high = branch_task(
    "HIGH",
    future_router,
    high_result
)

future_low = branch_task(
    "LOW",
    future_router,
    low_result
)

# 7. Wait for Workflow Completion

result_high = future_high.result()
result_low = future_low.result()

print("\nWorkflow completed!")

print("HIGH output:", result_high)
print("LOW output:", result_low)
parsl.dfk().cleanup()