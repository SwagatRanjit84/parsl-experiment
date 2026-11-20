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


# 2. Define the processing function

@python_app
def stage_a(input_file, output_file):
    with open(input_file, "r") as f:
        data = f.read()

    result = f"Stage-A processed:\n{data}"

    with open(output_file, "w") as f:
        f.write(result)

    print("Stage-A completed")
    return output_file


@python_app
def stage_b(input_file, output_file):
    with open(input_file, "r") as f:
        data = f.read()

    result = f"Stage-B processed:\n{data}"

    with open(output_file, "w") as f:
        f.write(result)

    print("Stage-B completed")
    return output_file


@python_app
def stage_c(input_file, output_file):
    with open(input_file, "r") as f:
        data = f.read()

    result = f"Stage-C processed:\n{data}"

    with open(output_file, "w") as f:
        f.write(result)

    print("Stage-C completed")
    return output_file


@python_app
def stage_d(input_file_b, input_file_c, output_file):
    with open(input_file_b, "r") as f:
        data_b = f.read()

    with open(input_file_c, "r") as f:
        data_c = f.read()

    result = (
        "Stage-D combined results:\n\n"
        + data_b
        + "\n\n"
        + data_c
    )

    with open(output_file, "w") as f:
        f.write(result)

    print("Stage-D completed")
    return output_file



# 3. Define files

file_in = os.path.abspath("input_data.txt")

file_a = os.path.abspath("stage_a_out.txt")
file_b = os.path.abspath("stage_b_out.txt")
file_c = os.path.abspath("stage_c_out.txt")
file_d = os.path.abspath("stage_d_out.txt")


# 4. Create the workflow (submit apps)

# Stage A: First stage
future_a = stage_a(file_in, file_a)

# Stage B and Stage C: Parallel branches
future_b = stage_b(future_a, file_b)
future_c = stage_c(future_a, file_c)

# Stage D: Wait for both B and C
future_d = stage_d(future_b, future_c, file_d)


# 5. Wait for the final result

print("Final output:", future_d.result())

print("\nWorkflow completed successfully!")
parsl.dfk().cleanup()