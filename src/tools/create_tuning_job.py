# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create a Vertex MSRL tuning job and wait for JOB_STATE_RUNNING.

Standalone helper so we can decouple the (up to 45 min) CreateTuningJob +
wait-for-RUNNING phase from the downstream eval/train invocation. Prints
the resulting numeric tuning job id on the last line of stdout so it can
be captured by a wrapper script.

Once RUNNING, the script exits *without* cancelling the job. The job
then looks like a pre-existing one to `src/train.py` (and to `src/eval.py`
via `target=`): pass its id as `vertex_existing_job_id=...` to reuse it.
It keeps billing until cancelled, so cancel it yourself when done:
    gcloud ai tuning-jobs cancel <job_id> --project=<p> --region=<r>

Example (with ``src`` on PYTHONPATH):
    uv run python -m tools.create_tuning_job \\
        vertex_project=<your-gcp-project> \\
        vertex_base_model=gemini-3.5-flash
"""

from __future__ import annotations

import asyncio
import logging

import chz

from backends.vertex_msrl_rest import (
    VertexMsrlRestClient as VertexMsrlClient,
    default_endpoint,
)

logger = logging.getLogger(__name__)


@chz.chz
class CreateJobConfig:
    # Your GCP project (required). Pass e.g. vertex_project=<your-gcp-project>.
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_base_model: str = "gemini-3.5-flash"
    vertex_job_running_timeout_seconds: int = 45 * 60
    vertex_request_timeout: float = 600.0
    description: str = "msrl client eval smoke job"


async def run(config: CreateJobConfig) -> str:
    if not config.vertex_project:
        raise ValueError(
            "vertex_project is required; pass vertex_project=<your-gcp-project>."
        )
    endpoint = default_endpoint(config.vertex_location)
    logger.info("Using endpoint %s (location=%s)", endpoint, config.vertex_location)

    client = VertexMsrlClient(
        project=str(config.vertex_project),
        location=config.vertex_location,
        endpoint=endpoint,
        request_timeout=config.vertex_request_timeout,
    )
    try:
        job_id = await client.create_tuning_job(
            base_model=config.vertex_base_model,
            description=config.description,
        )
        logger.info("Created tuning job id=%s; waiting for RUNNING...", job_id)
        await client.wait_for_job_running(
            timeout=float(config.vertex_job_running_timeout_seconds)
        )
        logger.info("Tuning job %s is RUNNING.", job_id)
        # NOTE: keep as print (NOT logger) \u2014 this line goes to stdout so
        # shell wrappers can capture it as ``JOB_ID=$(create_tuning_job ...)``.
        # Logger output routes to stderr by default, which would break that.
        print(job_id)
        return job_id
    finally:
        await client.aclose()


def main(config: CreateJobConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
