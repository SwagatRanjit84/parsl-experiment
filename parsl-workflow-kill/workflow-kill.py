import os
import signal
import time
import parsl
from parsl import bash_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

# Enable INFO logging to watch DFK state transitions and retry scheduling
parsl.set_stream_logger(level=20)  # 20 = logging.INFO

config = Config( 
    executors=[ HighThroughputExecutor(
            label="local_workflow_kill",
            provider=LocalProvider(),
            max_workers_per_node=4
        )],
    retries=3,  # Instruct Parsl to retry failed tasks once
    # Optional: custom retry_handler can also be passed here
)
parsl.load(config)

@bash_app
def run_job(job_id, duration, pid_file=None, stdout=parsl.AUTO_LOGNAME):
    # Only record PID if pid_file is passed AND hasn't been used yet
    pid_cmd = f"if [ ! -f {pid_file}.killed ]; then echo $$ > {pid_file}; fi && " if pid_file else ""
    return f"{pid_cmd}echo 'Job {job_id} running (PID: $$)'; sleep {duration}; echo 'Job {job_id} done'"

pid_path = os.path.abspath("job2.pid")
killed_flag = f"{pid_path}.killed"
for p in (pid_path, killed_flag):
    if os.path.exists(p):
        os.remove(p)

# 1. Launch all three jobs concurrently
f1 = run_job(1, 5)
f2 = run_job(2, 6, pid_file=pid_path)
f3 = run_job(3, 5)

# 2. Wait for Job 2's first attempt to start
while not os.path.exists(pid_path):
    time.sleep(0.1)

with open(pid_path, "r") as f:
    job2_pid = int(f.read().strip())

# 3. Kill Job 2's first attempt and mark it as killed so the retry isn't killed again
time.sleep(1.5)
open(killed_flag, "w").close()
print(f"\n INJECTING FAILURE: Killing Job 2 (PID {job2_pid}) \n")
os.kill(job2_pid, signal.SIGKILL)

# 4. Wait for all futures (Job 2 will take longer because it restarts)
for idx, fut in enumerate([f1, f2, f3], start=1):
    try:
        fut.result()
        print(f"Job {idx} final status: SUCCEEDED")
    except Exception as e:
        print(f"Job {idx} final status: FAILED ({type(e).__name__})")

# Cleanup
for p in (pid_path, killed_flag):
    if os.path.exists(p):
        os.remove(p)
parsl.dfk().cleanup()
