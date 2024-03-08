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
"""Node state."""


from typing import Any, Dict

<<<<<<< HEAD
from flwr.client.workload_state import WorkloadState


class NodeState:
    """State of a node where client nodes execute workloads."""

    def __init__(self) -> None:
        self._meta: Dict[str, Any] = {}  # holds metadata about the node
        self.workload_states: Dict[int, WorkloadState] = {}

    def register_workloadstate(self, workload_id: int) -> None:
        """Register new workload state for this node."""
        if workload_id not in self.workload_states:
            self.workload_states[workload_id] = WorkloadState({})

    def retrieve_workloadstate(self, workload_id: int) -> WorkloadState:
        """Get workload state given a workload_id."""
        if workload_id in self.workload_states:
            return self.workload_states[workload_id]

        raise RuntimeError(
            f"WorkloadState for workload_id={workload_id} doesn't exist."
            " A workload must be registered before it can be retrieved or updated "
            " by a client."
        )

    def update_workloadstate(
        self, workload_id: int, workload_state: WorkloadState
    ) -> None:
        """Update workload state."""
        self.workload_states[workload_id] = workload_state
=======
from flwr.common import Context, RecordSet


class NodeState:
    """State of a node where client nodes execute runs."""

    def __init__(self) -> None:
        self._meta: Dict[str, Any] = {}  # holds metadata about the node
        self.run_contexts: Dict[int, Context] = {}

    def register_context(self, run_id: int) -> None:
        """Register new run context for this node."""
        if run_id not in self.run_contexts:
            self.run_contexts[run_id] = Context(state=RecordSet())

    def retrieve_context(self, run_id: int) -> Context:
        """Get run context given a run_id."""
        if run_id in self.run_contexts:
            return self.run_contexts[run_id]

        raise RuntimeError(
            f"Context for run_id={run_id} doesn't exist."
            " A run context must be registered before it can be retrieved or updated "
            " by a client."
        )

    def update_context(self, run_id: int, context: Context) -> None:
        """Update run context."""
        self.run_contexts[run_id] = context
>>>>>>> upstream/main
