# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Node state tests."""


<<<<<<< HEAD
from flwr.client.node_state import NodeState
from flwr.client.workload_state import WorkloadState
from flwr.proto.task_pb2 import TaskIns


def _run_dummy_task(state: WorkloadState) -> WorkloadState:
    if "counter" in state.state:
        state.state["counter"] += "1"
    else:
        state.state["counter"] = "1"

    return state


def test_multiworkload_in_node_state() -> None:
    """Test basic NodeState logic."""
    # Tasks to perform
    tasks = [TaskIns(workload_id=w_id) for w_id in [0, 1, 1, 2, 3, 2, 1, 5]]
    # the "tasks" is to count how many times each workload is executed
=======
from typing import cast

from flwr.client.node_state import NodeState
from flwr.common import ConfigsRecord, Context
from flwr.proto.task_pb2 import TaskIns  # pylint: disable=E0611


def _run_dummy_task(context: Context) -> Context:
    counter_value: str = "1"
    if "counter" in context.state.configs_records.keys():
        counter_value = cast(str, context.state.configs_records["counter"]["count"])
        counter_value += "1"

    context.state.configs_records["counter"] = ConfigsRecord({"count": counter_value})

    return context


def test_multirun_in_node_state() -> None:
    """Test basic NodeState logic."""
    # Tasks to perform
    tasks = [TaskIns(run_id=run_id) for run_id in [0, 1, 1, 2, 3, 2, 1, 5]]
    # the "tasks" is to count how many times each run is executed
>>>>>>> upstream/main
    expected_values = {0: "1", 1: "1" * 3, 2: "1" * 2, 3: "1", 5: "1"}

    # NodeState
    node_state = NodeState()

    for task in tasks:
<<<<<<< HEAD
        w_id = task.workload_id

        # Register
        node_state.register_workloadstate(workload_id=w_id)

        # Get workload state
        state = node_state.retrieve_workloadstate(workload_id=w_id)

        # Run "task"
        updated_state = _run_dummy_task(state)

        # Update workload state
        node_state.update_workloadstate(workload_id=w_id, workload_state=updated_state)

    # Verify values
    for w_id, state in node_state.workload_states.items():
        assert state.state["counter"] == expected_values[w_id]
=======
        run_id = task.run_id

        # Register
        node_state.register_context(run_id=run_id)

        # Get run state
        context = node_state.retrieve_context(run_id=run_id)

        # Run "task"
        updated_state = _run_dummy_task(context)

        # Update run state
        node_state.update_context(run_id=run_id, context=updated_state)

    # Verify values
    for run_id, context in node_state.run_contexts.items():
        assert (
            context.state.configs_records["counter"]["count"] == expected_values[run_id]
        )
>>>>>>> upstream/main
