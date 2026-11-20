
import json
import os

import parsl
from parsl.app.app import python_app, bash_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.data_provider.files import File

# CONFIGURATION
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

WORK_DIR = os.path.join(PROJECT_DIR, "workflow_data")
os.makedirs(WORK_DIR, exist_ok=True)

# Log files
TRAINING_LOG = os.path.join(WORK_DIR, "training.log")
INFERENCE_LOG = os.path.join(WORK_DIR, "inference.log")
COMPILE_LOG = os.path.join(WORK_DIR, "compile.log")
PREFIX_SUM_LOG = os.path.join(WORK_DIR, "prefix_sum.log")

# Output files
MODEL_PATH = os.path.join(WORK_DIR, "model.pt")
PREFIX_RESULT_PATH = os.path.join(WORK_DIR, "prefix_result.txt")
FINAL_REPORT_PATH = os.path.join(WORK_DIR, "final_report.json")

# Parsl logging
parsl.set_stream_logger(level=20)

config = Config(
    executors=[
        HighThroughputExecutor(
            label="local_executor",
            provider=LocalProvider(),
            max_workers_per_node=4,
        )
    ],
    retries=1,
)

# STAGE 1: NEURAL NETWORK TRAINING
@python_app
def train_ml_model(project_dir, log_path, outputs=()):
    import os
    import sys

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    os.chdir(project_dir)

    from ml_training import train_model

    model_path = outputs[0].filepath

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print("[STAGE 1] Starting neural network training", flush=True)

    return train_model(
        model_path,
        log_path=log_path,
    )


# STAGE 2: OPENMP COMPILATION
@bash_app
def compile_openmp(project_dir, log_path, outputs=()):
    import os
    import shlex

    project_path = shlex.quote(project_dir)
    binary_path = shlex.quote(outputs[0].filepath)
    log_file = shlex.quote(log_path)

    source1 = shlex.quote(
        os.path.join(project_dir, "prefix_sum_test.c")
    )
    source2 = shlex.quote(
        os.path.join(project_dir, "prefix_sum_recur.c")
    )
    source3 = shlex.quote(
        os.path.join(project_dir, "prefix_sum_tree.c")
    )

    return f"""
set -e

cd {project_path}

mkdir -p "$(dirname {log_file})"

exec > {log_file} 2>&1

echo "[STAGE 2] OPENMP COMPILATION"

echo "[COMPILATION] Removing previous object files"

rm -f prefix_sum_test.o
rm -f prefix_sum_recur.o
rm -f prefix_sum_tree.o

echo "[COMPILATION] Compiling prefix_sum_test.c"

gcc-16 -Wall -O2 -fopenmp \
    -c {source1} -o prefix_sum_test.o

echo "[COMPILATION] Compiling prefix_sum_recur.c"

gcc-16 -Wall -O2 -fopenmp \
    -c {source2} -o prefix_sum_recur.o

echo "[COMPILATION] Compiling prefix_sum_tree.c"

gcc-16 -Wall -O2 -fopenmp \
    -c {source3} -o prefix_sum_tree.o

echo "[COMPILATION] Linking executable"

gcc-16 \
    prefix_sum_test.o \
    prefix_sum_recur.o \
    prefix_sum_tree.o \
    -o {binary_path} \
    -fopenmp

echo "[STAGE 2] OPENMP COMPILATION COMPLETED"
"""


# STAGE 3: NEURAL NETWORK INFERENCE
# Depends on Stage 1
@python_app
def run_ml_inference(project_dir, log_path, inputs=()):
    import os
    import sys

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    os.chdir(project_dir)

    from ml_inference import run_inference

    model_path = inputs[0].filepath

    print("[STAGE 3] Starting neural network inference", flush=True)

    return run_inference(
        model_path,
        log_path=log_path,
    )


# STAGE 4: OPENMP PREFIX SUM EXECUTION
# Depends on Stage 2
@bash_app
def run_openmp_prefix_sum(
    project_dir,
    log_path,
    exponent=22,
    inputs=(),
    outputs=(),
):
    import os
    import shlex

    project_path = shlex.quote(project_dir)

    binary_path = shlex.quote(
        inputs[0].filepath
    )
    result_path = shlex.quote(
        outputs[0].filepath
    )
    log_file = shlex.quote(log_path)

    return f"""
set -e

cd {project_path}

mkdir -p "$(dirname {result_path})"
mkdir -p "$(dirname {log_file})"

exec > {log_file} 2>&1

echo "[STAGE 4] OPENMP PREFIX SUM EXECUTION"

echo "[OPENMP] Executable: {binary_path}"
echo "[OPENMP] Exponent: {exponent}"
echo "[OPENMP] Result file: {result_path}"

echo ""
echo "[OPENMP] Starting execution..."

{binary_path} {exponent} {result_path}

echo ""
echo "[STAGE 4] OPENMP EXECUTION COMPLETED"
"""


# STAGE 5: RESULT AGGREGATION
# Depends on Stages 3 and 4

@python_app
def generate_report(ml_metrics, inputs=(), outputs=()):
    import json
    import os

    prefix_output_file = inputs[0].filepath
    report_path = outputs[0].filepath

    prefix_benchmarks = []

    print("[STAGE 5] Reading OpenMP benchmark results", flush=True)

    with open(prefix_output_file, "r") as result_file:
        for line in result_file:
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

                prefix_benchmarks.append(
                    {
                        "array_size": size,
                        "recur_time_sec": recur_time,
                        "tree_time_sec": tree_time,
                        "tree_speedup_x": speedup,
                    }
                )

            except ValueError:
                continue

    largest_run = (
        max(
            prefix_benchmarks,
            key=lambda item: item["array_size"],
        )
        if prefix_benchmarks
        else {}
    )

    final_report = {
        "workflow_summary":
            "Hybrid Single Neuron Neural Network + OpenMP HPC Pipeline",
        "ml_inference_metrics": ml_metrics,
        "openmp_largest_scale_summary": largest_run,
        "openmp_full_scaling_table": prefix_benchmarks,
    }

    os.makedirs(
        os.path.dirname(report_path),
        exist_ok=True,
    )

    with open(report_path, "w") as report_file:
        json.dump(
            final_report,
            report_file,
            indent=4,
        )

    print("[STAGE 5] REPORT GENERATED", flush=True)

    return final_report


# MAIN WORKFLOW EXECUTION

if __name__ == "__main__":
    parsl_loaded = False

    try:
        parsl.load(config)
        parsl_loaded = True

        # Parsl file references
        model_file = File(MODEL_PATH)

        binary_file = File(
            os.path.join(PROJECT_DIR, "prefix_sum_test")
        )

        prefix_result_file = File(PREFIX_RESULT_PATH)
        final_report_file = File(FINAL_REPORT_PATH)

        # Workflow information
        print("\n" + "=" * 70)
        print("       PARSL HYBRID ML + OPENMP HPC WORKFLOW")
        print("=" * 70)

        print("Workflow manager : Parsl")
        print("Executor         : HighThroughputExecutor")
        print("Provider         : LocalProvider")
        print("Workers          : 4")
        print("Languages        : Python + PyTorch + C + OpenMP")
        print("Neural Network   : Single Hidden Neuron")
        print("Training Epochs  : 500")

        print("\nLOG FILES")
        print("Training log     :", TRAINING_LOG)
        print("Inference log    :", INFERENCE_LOG)
        print("Compilation log  :", COMPILE_LOG)
        print("Prefix sum log   :", PREFIX_SUM_LOG)

        print("\nOUTPUT FILES")
        print("Model            :", MODEL_PATH)
        print("Prefix results   :", PREFIX_RESULT_PATH)
        print("Final report     :", FINAL_REPORT_PATH)

        print("=" * 70)

        # STAGE 1: ML TRAINING

        print("\n[STAGE 1] Submitting ML training")

        train_future = train_ml_model(
            project_dir=PROJECT_DIR,
            log_path=TRAINING_LOG,
            outputs=[model_file],
        )

        # STAGE 2: OPENMP COMPILATION

        print("\n[STAGE 2] Submitting OpenMP compilation")

        compile_future = compile_openmp(
            project_dir=PROJECT_DIR,
            log_path=COMPILE_LOG,
            outputs=[binary_file],
        )

        # STAGE 3: ML INFERENCE
        # Depends on Stage 1

        print("\n[STAGE 3] Submitting ML inference")

        inference_future = run_ml_inference(
            project_dir=PROJECT_DIR,
            log_path=INFERENCE_LOG,
            inputs=[train_future.outputs[0]],
        )

        # STAGE 4: OPENMP PREFIX SUM
        # Depends on Stage 2

        print("\n[STAGE 4] Submitting OpenMP prefix sum")

        openmp_future = run_openmp_prefix_sum(
            project_dir=PROJECT_DIR,
            log_path=PREFIX_SUM_LOG,
            exponent=22,
            inputs=[compile_future.outputs[0]],
            outputs=[prefix_result_file],
        )

        # STAGE 5: RESULT AGGREGATION
        # Depends on Stages 3 and 4

        print("\n[STAGE 5] Submitting result aggregation")

        report_future = generate_report(
            ml_metrics=inference_future,
            inputs=[openmp_future.outputs[0]],
            outputs=[final_report_file],
        )

        # WAIT FOR WORKFLOW COMPLETION

        print("\n[PARSL] All workflow tasks submitted")
        print("[PARSL] Waiting for all stages to complete...")
        print("[PARSL] Training log   :", TRAINING_LOG)
        print("[PARSL] Inference log  :", INFERENCE_LOG)
        print("[PARSL] Prefix sum log :", PREFIX_SUM_LOG)

        result = report_future.result()

        # FINAL REPORT
        print("\n" + "=" * 70)
        print("FINAL AGGREGATED REPORT")
        print("=" * 70)

        print(
            json.dumps(
                result,
                indent=4,
            )
        )

        print("\n[PARSL] Report saved to:")
        print(final_report_file.filepath)

        print("\n[PARSL] WORKFLOW COMPLETED SUCCESSFULLY")

    except Exception as exc:
        print("\n[PARSL] WORKFLOW FAILED")
        print("Error:", str(exc))
        raise

    finally:
        if parsl_loaded:
            print("\n[PARSL] Cleaning up")
            parsl.dfk().cleanup()
            print("[PARSL] Workflow finished")