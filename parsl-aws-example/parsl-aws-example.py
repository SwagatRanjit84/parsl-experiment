import logging
from pathlib import Path

import parsl
from parsl import python_app
from parsl.addresses import address_by_query
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.jobs.states import JobState, JobStatus
from parsl.providers import AWSProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

REGION = "us-east-2"
AMI_ID = "ami-0b064c912c726f739"
INSTANCE_TYPE = "t3.small"
KEY_NAME = "parsl"

PROJECT_DIR = Path(__file__).resolve().parent
STATE_FILE = PROJECT_DIR / "awsproviderstate.json"


class SafeAWSProvider(AWSProvider):

    def status(self, job_ids):
        job_ids = list(job_ids)
        if not job_ids:
            return []

        response = self.client.describe_instances(InstanceIds=job_ids)
        instance_map = {}

        for reservation in response.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                instance_map[instance["InstanceId"]] = instance

        state_translation = {
            "pending": JobState.PENDING,
            "running": JobState.RUNNING,
            "stopping": JobState.FAILED,
            "stopped": JobState.FAILED,
            "shutting-down": JobState.FAILED,
            "terminated": JobState.FAILED,
        }

        all_states = []
        for job_id in job_ids:
            instance = instance_map.get(job_id)
            if instance is None:
                previous = self.resources.get(job_id, {})
                instance_status = previous.get("status", JobStatus(JobState.UNKNOWN))
            else:
                aws_state = instance["State"]["Name"]
                parsl_state = state_translation.get(aws_state, JobState.UNKNOWN)
                instance_status = JobStatus(parsl_state)
                if job_id in self.resources:
                    self.resources[job_id]["status"] = instance_status
            all_states.append(instance_status)

        return all_states


# Do NOT use #!/bin/bash or `set -u` here; AWSProvider embeds this block inside its own script.
# Export PIP_BREAK_SYSTEM_PACKAGES=1 so Ubuntu 24.04 allows system-wide pip installs.
worker_init = f"""
export DEBIAN_FRONTEND=noninteractive
export PIP_BREAK_SYSTEM_PACKAGES=1

apt-get update -y
apt-get install -y software-properties-common curl
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y python3.13 python3.13-venv python3.13-dev
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 - --break-system-packages
ln -sf /usr/bin/python3.13 /usr/bin/python3
ln -sf /usr/local/bin/pip /usr/bin/pip3
python3 -m pip install --break-system-packages "parsl=={parsl.__version__}" numpy scikit-learn
"""

provider = SafeAWSProvider(
    image_id=AMI_ID,
    instance_type=INSTANCE_TYPE,
    region=REGION,
    profile="default",
    key_name=KEY_NAME,
    init_blocks=1,
    min_blocks=0,
    max_blocks=1,
    worker_init=worker_init,
    state_file=str(STATE_FILE),
    # Pass an empty requirement or pip flag so AWSProvider's built-in pip3 call doesn't fail on PEP 668
    parsl_requirement=f"--break-system-packages parsl=={parsl.__version__}",
    linger=False,
)

executor = HighThroughputExecutor(
    label="aws_executor",
    # Tell the remote EC2 worker your public IP instead of your local 192.168.x.x LAN IP
    address="0.0.0.0",
    encrypted=False,
    worker_port=54000,
    interchange_port_range=(54051, 54100),
    provider=provider,
    max_workers_per_node=1,
)

executor.launch_cmd = executor.launch_cmd.replace("{addresses}", address_by_query())

config = Config(
    executors=[executor],
    retries=0,
)


@python_app
def linear_regression_prediction():
    import numpy as np
    from sklearn.linear_model import LinearRegression

    house_sizes = np.array([[1000], [1200], [1500], [1800], [2000], [2200]])
    house_prices = np.array([150, 180, 225, 270, 300, 330])

    model = LinearRegression()
    model.fit(house_sizes, house_prices)
    prediction = model.predict([[1500]])[0]

    return {
        "prediction": float(prediction),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }


def main():
    try:
        print("\nLoading Parsl configuration...")
        parsl.load(config)

        print("\nSubmitting linear regression task to AWS EC2...")
        future = linear_regression_prediction()

        print("\nWaiting for EC2 worker to execute the task (timeout=600s)...")
        result = future.result(timeout=600)

        print("\n====================================")
        print("LINEAR REGRESSION RESULTS")
        print("====================================")
        print("Predicted house price:", result["prediction"])
        print("Model slope:", result["slope"])
        print("Model intercept:", result["intercept"])

    except Exception:
        logging.exception("Workflow failed")
        raise
    finally:
        print("\nCleaning up Parsl resources...")
        parsl.clear()


if __name__ == "__main__":
    main()