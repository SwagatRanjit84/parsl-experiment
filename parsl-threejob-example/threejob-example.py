
import os
import parsl

from parsl import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
parsl.set_stream_logger(level=20)  # 20 = logging.INFO

# Configure Parsl
config = Config(
    executors=[
        HighThroughputExecutor(
            label="local_pipeline",
            provider=LocalProvider(),
            max_workers_per_node=4,
        )
    ]
)

parsl.load(config)

# Processor application
@python_app
def data_processor(stage_name, input_file, output_file):
    with open(input_file, "r") as f:
        data = f.read()

    processed = f"[{stage_name}] {data}"

    with open(output_file, "w") as f:
        f.write(processed)

    print(f"{stage_name} completed")
    return output_file

# Input file
file_in = os.path.abspath("input_data.txt")
file_mid1 = os.path.abspath("stage1_output.txt")
file_mid2 = os.path.abspath("stage2_output.txt")
file_out = os.path.abspath("final_output.txt")

# Stage A
job1 = data_processor("Stage-A", file_in, file_mid1)

# Stage B waits for Stage A
job2 = data_processor("Stage-B", job1, file_mid2)

# Stage C waits for Stage B
job3 = data_processor("Stage-C", job2, file_out)

# Wait for final stage and retrieve result
result = job3.result()

print("Pipeline completed successfully!")
print("Final output file:", result)

with open(file_out, "r") as f:
    print("Final content:", f.read())

parsl.dfk().cleanup()
