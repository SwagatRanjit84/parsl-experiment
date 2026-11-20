
import os
import json
import shlex
import parsl

from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl.data_provider.files import File

# CONFIGURATION
WORK_DIR = os.path.abspath("workflow_data")
os.makedirs(WORK_DIR, exist_ok=True)

config = Config(
    executors=[
        ThreadPoolExecutor(
            label="local_threads",
            max_threads=4
        )
    ]
)

# STAGE 1: NEURAL NETWORK TRAINING
# Implementation is in ml_training.py
@python_app
def train_ml_model(outputs=()):
    from ml_training import train_model
    model_path = outputs[0].filepath
    print("\n[STAGE 1] NEURAL NETWORK TRAINING", flush=True)
    return train_model(model_path)

# STAGE 2: OPENMP COMPILATION
@bash_app
def compile_openmp(outputs=()):

    binary_path = shlex.quote(outputs[0].filepath)
    return f"""
    set -e
    echo "[STAGE 2] OPENMP COMPILATION"
    rm -f prefix_sum_test.o prefix_sum_recur.o prefix_sum_tree.o
    gcc-16 -Wall -O2 -fopenmp -c prefix_sum_test.c -o prefix_sum_test.o
    gcc-16 -Wall -O2 -fopenmp -c prefix_sum_recur.c -o prefix_sum_recur.o
    gcc-16 -Wall -O2 -fopenmp -c prefix_sum_tree.c -o prefix_sum_tree.o
    gcc-16 prefix_sum_test.o prefix_sum_recur.o prefix_sum_tree.o \
        -o {binary_path} -fopenmp
    echo "[STAGE 2] OPENMP COMPILATION COMPLETED"
    """

# STAGE 3: NEURAL NETWORK INFERENCE
# Implementation is in ml_inference.py
@python_app
def run_ml_inference(inputs=()):
    from ml_inference import run_inference
    model_path = inputs[0].filepath
    print("\n[STAGE 3] NEURAL NETWORK INFERENCE", flush=True)
    return run_inference(model_path)

# STAGE 4: OPENMP PREFIX SUM EXECUTION
@bash_app
def run_openmp_prefix_sum(
    exponent=28,
    inputs=(),
    outputs=()
):

    binary_path = shlex.quote(inputs[0].filepath)
    result_path = shlex.quote(outputs[0].filepath)

    return f"""
    set -e
    echo "[STAGE 4] OPENMP PREFIX SUM EXECUTION"
    {binary_path} {exponent} {result_path}
    echo "[STAGE 4] OPENMP EXECUTION COMPLETED"
    """

# STAGE 5: RESULT AGGREGATION
@python_app
def generate_report(ml_metrics, inputs=(), outputs=()):
    import json
    print("\n[STAGE 5] RESULT AGGREGATION", flush=True)
    prefix_output_file = inputs[0].filepath
    report_path = outputs[0].filepath
    prefix_benchmarks = []

    # Read OpenMP benchmark results
    with open(prefix_output_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            try:
                size = int(parts[0])
                recur_time = float(parts[1])
                tree_time = float(parts[2])
                speedup = (
                    round(recur_time / tree_time, 2)
                    if tree_time > 0
                    else 0.0
                )
                prefix_benchmarks.append({
                    "array_size": size,
                    "recur_time_sec": recur_time,
                    "tree_time_sec": tree_time,
                    "tree_speedup_x": speedup
                })

            except ValueError:
                continue

    # Largest benchmark result
    largest_run = (
        max(
            prefix_benchmarks,
            key=lambda item: item["array_size"]
        )
        if prefix_benchmarks
        else {}
    )

    # Construct final report
    final_report = {
        "workflow_summary":
            "Hybrid Single Neuron Neural Network + OpenMP HPC Pipeline",
        "ml_inference_metrics": ml_metrics,
        "openmp_largest_scale_summary": largest_run,
        "openmp_full_scaling_table": prefix_benchmarks
    }

    # Save JSON report
    with open(report_path, "w") as f:
        json.dump(
            final_report,
            f,
            indent=4
        )
    print("[STAGE 5] REPORT GENERATED", flush=True)
    return final_report


# MAIN WORKFLOW EXECUTION
if __name__ == "__main__":
    parsl.load(config)
    # Define Parsl files
    model_file = File(
        os.path.join(WORK_DIR, "model.pt")
    )
    binary_file = File(
        os.path.abspath("prefix_sum_test")
    )
    prefix_result_file = File(
        os.path.join(WORK_DIR, "prefix_result.txt")
    )
    final_report_file = File(
        os.path.join(WORK_DIR, "final_report.json")
    )

    try:
        print("\n" + "=" * 65)
        print("       PARSL HYBRID ML + OPENMP HPC WORKFLOW")
        print("=" * 65)
        print("Workflow manager : Parsl")
        print("Executor         : ThreadPoolExecutor")
        print("Languages        : Python + PyTorch + C + OpenMP")
        print("Neural Network   : Single Hidden Neuron")
        print("Training Epochs  : 500")
        print("=" * 65)

        # STAGE 1: ML TRAINING
        print("\n[STAGE 1] Submitting ML training")

        train_future = train_ml_model(
            outputs=[model_file]
        )
        # STAGE 2: OPENMP COMPILATION
        print("\n[STAGE 2] Submitting OpenMP compilation")

        compile_future = compile_openmp(
            outputs=[binary_file]
        )

        # STAGE 3: ML INFERENCE
        # Depends on Stage 1
        print("\n[STAGE 3] Submitting ML inference")

        inference_future = run_ml_inference(
            inputs=[train_future.outputs[0]]
        )

        # STAGE 4: OPENMP PREFIX SUM
        # Depends on Stage 2
        print("\n[STAGE 4] Submitting OpenMP prefix sum")

        openmp_future = run_openmp_prefix_sum(
            exponent=28,
            inputs=[compile_future.outputs[0]],
            outputs=[prefix_result_file]
        )
        # STAGE 5: RESULT AGGREGATION
        # Depends on Stages 3 and 4
        print("\n[STAGE 5] Submitting result aggregation")

        report_future = generate_report(
            ml_metrics=inference_future,
            inputs=[openmp_future.outputs[0]],
            outputs=[final_report_file]
        )

        print("\n[PARSL] All workflow tasks submitted")

        # Wait for complete workflow
        result = report_future.result()

        # FINAL REPORT
        print("\n" + "=" * 65)
        print("             FINAL AGGREGATED REPORT")
        print("=" * 65)
        print(
            json.dumps(
                result,
                indent=4
            )
        )
        print("\n[PARSL] Report saved to:")
        print(final_report_file.filepath)
        print("\n[PARSL] WORKFLOW COMPLETED SUCCESSFULLY")

    except Exception as e:
        print("\n[PARSL] WORKFLOW FAILED")
        print("Error:", str(e))
        raise
    finally:
        print("\n[PARSL] Cleaning up")
        parsl.dfk().cleanup()
        print("[PARSL] Workflow finished")