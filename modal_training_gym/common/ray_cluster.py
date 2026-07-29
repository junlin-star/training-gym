"""Shared helpers for running a Ray cluster across Modal containers.

- `start_ray_head` / `start_ray_worker`: low-level primitives for spawning the
  Ray head/worker processes on a single container.
- `ModalRayCluster`: a base class that stitches those primitives together with
  Modal cluster discovery, the Ray `JobSubmissionClient`, dashboard port
  forwarding, and a worker keep-alive loop. Framework launchers (slime,
  verl, …) subclass or compose this.
"""

import asyncio
import inspect
import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modal.experimental import clustered

RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
_MAX_DIAGNOSTIC_NODES = 256
_MAX_DIAGNOSTIC_RESOURCE_FIELDS = 256
_MAX_DIAGNOSTIC_NODE_TEXT = 4_000
_MAX_DIAGNOSTIC_STATUS_TEXT = 12_000
_RAY_API_DIAGNOSTIC_TIMEOUT_SECONDS = 5.0
MODAL_CLUSTER_MEMBER_EVENT = "TRAINING_GYM_MODAL_CLUSTER_MEMBER"

# GPU families with an RDMA/EFA fabric; other types don't support it and would fail
# if it were forced on.
_RDMA_GPU_TYPES = frozenset({"H100", "H200", "B200", "B300", "GB200"})


def _supports_rdma(gpu_type: str) -> bool:
    """Whether *gpu_type* (e.g. ``"H100"`` or ``"H100:8"``) is RDMA/EFA-capable."""
    return gpu_type.split(":")[0].strip().upper() in _RDMA_GPU_TYPES


def clustered_if(
    use_clustered: bool, size: int, *, gpu_type: str
) -> Callable[[Callable], Callable]:
    """Return ``clustered(size, rdma=…)`` when *use_clustered*, else an identity
    decorator that registers the function as a plain ``@app.function``.

    RDMA/EFA is enabled only when clustered (i.e. multi-node) *and* the GPU family
    supports it (H100/H200/B200/B300/GB200).
    """
    if use_clustered:
        return clustered(size, rdma=_supports_rdma(gpu_type))  # pyright: ignore[reportCallIssue, reportOptionalCall]

    def _identity(fn: Callable) -> Callable:
        return fn

    return _identity


def start_ray_head(
    node_ip: str,
    n_nodes: int,
    *,
    init_retries: int = 30,
    worker_wait_retries: int = 60,
    extra_start_args: list[str] | None = None,
) -> None:
    """Start the Ray head and block until all workers have joined.

    Spawns `ray start --head` in the background, then polls `ray.init()` until
    the head is reachable and `ray.nodes()` until `n_nodes` are alive. Raises
    `RuntimeError` if either poll times out.
    """
    import ray

    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={node_ip}",
        "--dashboard-host=0.0.0.0",
    ]
    if extra_start_args:
        cmd.extend(extra_start_args)
    subprocess.Popen(cmd)

    for _ in range(init_retries):
        try:
            ray.init(address="auto")
            break
        except ConnectionError:
            time.sleep(1)
    else:
        raise RuntimeError("Failed to connect to the Ray head node")

    for _ in range(worker_wait_retries):
        alive_nodes = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for Ray workers: {len(alive_nodes)}/{n_nodes} alive")
        if len(alive_nodes) == n_nodes:
            print("All Ray nodes connected")
            return
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def start_ray_worker(
    node_ip: str,
    head_addr: str,
    *,
    head_port: int = RAY_PORT,
    connect_retries: int = 60,
    start_retries: int = 3,
    start_timeout_seconds: float = 60.0,
    retry_interval_seconds: float = 1.0,
    extra_start_args: list[str] | None = None,
) -> None:
    """Start a Ray worker after the head is reachable, failing on CLI errors.

    Cluster workers can enter this function while rank zero is still creating
    the durable training-attempt record. Polling the head port avoids spending
    Ray's own finite GCS retries before rank zero has launched the head.
    """
    if connect_retries < 1:
        raise ValueError("connect_retries must be positive")
    if start_retries < 1:
        raise ValueError("start_retries must be positive")
    if start_timeout_seconds <= 0:
        raise ValueError("start_timeout_seconds must be positive")
    if retry_interval_seconds < 0:
        raise ValueError("retry_interval_seconds cannot be negative")
    cmd = [
        "ray",
        "start",
        f"--node-ip-address={node_ip}",
        "--address",
        f"{head_addr}:{head_port}",
    ]
    if extra_start_args:
        cmd.extend(extra_start_args)

    head_error = ""
    for attempt in range(connect_retries):
        try:
            connection = socket.create_connection(
                (head_addr, head_port),
                timeout=1.0,
            )
            connection.close()
        except OSError as exc:
            head_error = f"{type(exc).__name__}: {exc}"
        else:
            break
        if attempt + 1 < connect_retries:
            time.sleep(retry_interval_seconds)
    else:
        raise RuntimeError(
            f"Ray head {head_addr}:{head_port} was not reachable after "
            f"{connect_retries} attempts: {head_error}"
        )

    last_error = ""
    for attempt in range(start_retries):
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                text=True,
                timeout=start_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            last_error = f"ray start timed out after {exc.timeout} seconds"
        else:
            if completed.returncode == 0:
                return
            output = ((completed.stdout or "") + (completed.stderr or ""))[-4_000:]
            last_error = f"ray start exited with code {completed.returncode}: {output}"

        # A timed-out `ray start` may already have spawned a raylet and session
        # directory. Retrying into that partial state deterministically fails
        # with "Ray processes already running", so restore a clean local node
        # before every subsequent attempt (and before surfacing final failure).
        try:
            stopped = subprocess.run(
                ["ray", "stop", "--force"],
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            last_error += f"; ray stop timed out after {exc.timeout} seconds"
            break
        if stopped.returncode != 0:
            stop_output = ((stopped.stdout or "") + (stopped.stderr or ""))[-2_000:]
            last_error += (
                f"; ray stop exited with code {stopped.returncode}: {stop_output}"
            )
            break
        if attempt + 1 < start_retries:
            time.sleep(retry_interval_seconds)
    raise RuntimeError(
        f"Failed to start Ray worker against {head_addr}:{head_port} "
        f"after {start_retries} attempts: {last_error}"
    )


@dataclass
class ModalRayJobResult:
    status: str
    is_success: bool
    message: str | None = None
    diagnostics: dict[str, Any] | None = None
    job_id: str | None = None


def _bounded_resource_map(resources: Any) -> dict[str, Any]:
    if not isinstance(resources, dict):
        return {}
    return dict(list(resources.items())[:_MAX_DIAGNOSTIC_RESOURCE_FIELDS])


def _bounded_node_value(field: str, value: Any) -> Any:
    if field == "Resources":
        return _bounded_resource_map(value)
    if isinstance(value, str):
        return value[-_MAX_DIAGNOSTIC_NODE_TEXT:]
    return value


def _capture_ray_api_snapshot() -> dict[str, Any]:
    import ray

    return {
        "nodes": list(ray.nodes()),
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
    }


def _call_with_timeout(
    fn: Callable[[], dict[str, Any]],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run a diagnostic call on a daemon thread with a hard caller deadline."""
    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.update(fn())
        except BaseException as exc:  # noqa: BLE001 - forwarded to caller
            error.append(exc)

    thread = threading.Thread(
        target=_target,
        name="training-gym-ray-diagnostics",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=max(0.0, timeout_seconds))
    if thread.is_alive():
        raise TimeoutError(
            f"Ray control-plane diagnostics exceeded {timeout_seconds:.1f} seconds"
        )
    if error:
        raise error[0]
    return result


def capture_ray_cluster_diagnostics() -> dict[str, Any]:
    """Capture a bounded, best-effort Ray control-plane snapshot.

    Modal's own container metrics remain the authority for host OOMs and
    preemptions. This snapshot records what Ray knew about every node when the
    job became terminal, which is the missing link for raylet heartbeat loss.
    """
    snapshot: dict[str, Any] = {"captured_at": int(time.time())}
    try:
        node_fields = (
            "NodeID",
            "Alive",
            "NodeManagerAddress",
            "NodeManagerHostname",
            "NodeManagerPort",
            "NodeName",
            "State",
            "StateMessage",
            "DeathReason",
            "DeathReasonMessage",
            "Resources",
        )
        ray_snapshot = _call_with_timeout(
            _capture_ray_api_snapshot,
            timeout_seconds=_RAY_API_DIAGNOSTIC_TIMEOUT_SECONDS,
        )
        raw_nodes = ray_snapshot["nodes"]
        snapshot["nodes"] = [
            {
                key: _bounded_node_value(key, node[key])
                for key in node_fields
                if key in node
            }
            for node in raw_nodes[:_MAX_DIAGNOSTIC_NODES]
        ]
        if len(raw_nodes) > _MAX_DIAGNOSTIC_NODES:
            snapshot["nodes_truncated"] = len(raw_nodes) - _MAX_DIAGNOSTIC_NODES
        snapshot["cluster_resources"] = _bounded_resource_map(
            ray_snapshot["cluster_resources"]
        )
        snapshot["available_resources"] = _bounded_resource_map(
            ray_snapshot["available_resources"]
        )
    except BaseException as exc:  # noqa: BLE001 - diagnostics must not mask failure
        snapshot["ray_api_error"] = (f"{type(exc).__name__}: {exc}")[
            -_MAX_DIAGNOSTIC_NODE_TEXT:
        ]

    try:
        status = subprocess.run(
            ["ray", "status"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        output = (status.stdout or "") + (status.stderr or "")
        snapshot["ray_status_exit_code"] = status.returncode
        snapshot["ray_status"] = output[-_MAX_DIAGNOSTIC_STATUS_TEXT:]
    except Exception as exc:  # noqa: BLE001 - diagnostics must not mask the failure
        snapshot["ray_status_error"] = (f"{type(exc).__name__}: {exc}")[
            -_MAX_DIAGNOSTIC_NODE_TEXT:
        ]
    return snapshot


class ModalRayCluster:
    """Base class for bootstrapping a Ray cluster inside Modal clustered functions.

    Manages cluster discovery, Ray head/worker startup,
    ``JobSubmissionClient`` creation, dashboard forwarding, and a
    worker keep-alive loop. Framework launchers (slime, etc.)
    compose or subclass this.

    Attributes (populated after ``discover_cluster()``)
    ---------------------------------------------------
    n_nodes : int
        Total cluster node count.
    cluster_id : str
        Modal's stable clustered-function invocation ID, when available.
    container_ipv4_ips : list[str]
        Container IPv4 addresses ordered by Modal rank.
    rank : int
        This container's rank (0 = head).
    head_addr : str
        IPv4 address of the head node.
    node_ip : str
        IPv4 address of this container.
    is_head : bool
        Property — ``True`` when ``rank == 0``.

    Methods
    -------
    discover_cluster(n_nodes)
        Populate rank, addresses, and node count from Modal cluster info.
    start_ray(init_retries=30, worker_wait_retries=60)
        Start Ray head or worker using discovered cluster state.
    start(n_nodes, init_retries=30, worker_wait_retries=60)
        Convenience: ``discover_cluster`` + ``start_ray``.
    forward_dashboard()
        Return a ``modal.forward`` context manager for the Ray dashboard.
    submit_and_tail(entrypoint, runtime_env=None)
        Async: submit a Ray job, stream logs, return final status.
    wait_forever(poll_seconds=10)
        Async: keep a worker container alive until termination.
    """

    def __init__(self) -> None:
        # Populated by discover_cluster(); see _discovered for readiness check.
        self.n_nodes: int = 0
        self.cluster_id: str = ""
        self.container_ipv4_ips: list[str] = []
        self.rank: int = 0
        self.head_addr: str = ""
        self.node_ip: str = ""
        self._client = None
        self.last_submitted_job_id: str | None = None
        self._discovered: bool = False
        self._started: bool = False

    def head_extra_start_args(self) -> list[str]:
        """Override to append flags to `ray start --head` (e.g. resource hints)."""
        return []

    def worker_extra_start_args(self) -> list[str]:
        """Override to append flags to `ray start` on worker ranks."""
        return []

    @property
    def is_head(self) -> bool:
        if not self._discovered:
            raise RuntimeError(
                "ModalRayCluster.discover_cluster() has not been called yet"
            )
        return self.rank == 0

    @property
    def client(self):
        """Ray `JobSubmissionClient`. Only valid on the head node."""
        if not self.is_head:
            raise RuntimeError("JobSubmissionClient is only available on the head node")
        return self._client

    def discover_cluster(self, n_nodes: int) -> None:
        """Populate cluster identity and addresses from Modal.

        Does not start Ray. Call this first if you need to read cluster state
        (e.g. to set framework-specific env vars) before `start_ray()` inherits
        the environment into the Ray daemon. Idempotent.
        """
        if self._discovered:
            return
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be positive, got {n_nodes}")

        if n_nodes == 1:
            # Modal may omit container IPv4s for size-1 clustered functions.
            cluster_id = ""
            ips = ["127.0.0.1"]
            rank, head_addr, node_ip = 0, ips[0], ips[0]
        else:
            import modal.experimental

            info = modal.experimental.get_cluster_info()
            ips = list(info.container_ipv4_ips or [])
            if len(ips) != n_nodes:
                raise RuntimeError(
                    f"Modal cluster size mismatch: expected {n_nodes} nodes, got {len(ips)}"
                )
            cluster_id = str(getattr(info, "cluster_id", "") or "")
            rank = int(info.rank)
            if not 0 <= rank < n_nodes:
                raise RuntimeError(
                    f"Modal cluster rank out of range: rank {rank}, size {n_nodes}"
                )
            head_addr = ips[0]
            node_ip = ips[rank]

        self.n_nodes = n_nodes
        self.cluster_id = cluster_id
        self.container_ipv4_ips = ips
        self.rank = rank
        self.head_addr = head_addr
        self.node_ip = node_ip
        self._discovered = True

    def identity_snapshot(self) -> dict[str, Any]:
        """Return the immutable fields that identify this Modal cluster attempt.

        The returned mapping owns its address list, so attaching it to a run
        record cannot be mutated through this object's internal state.
        """
        if not self._discovered:
            raise RuntimeError(
                "ModalRayCluster.discover_cluster() has not been called yet"
            )
        return {
            "schema_version": 1,
            "cluster_id": self.cluster_id or None,
            "node_count": self.n_nodes,
            "rank_ordered_container_ipv4_ips": list(self.container_ipv4_ips),
            "head_addr": self.head_addr,
        }

    def emit_member_identity(self, *, training_run_id: str) -> dict[str, Any]:
        """Emit the rank/IP/task binding needed to address one clustered container."""

        if not self._discovered:
            raise RuntimeError(
                "ModalRayCluster.discover_cluster() has not been called yet"
            )
        task_id = os.environ.get("MODAL_TASK_ID", "")
        if not task_id:
            raise RuntimeError("Modal did not provide MODAL_TASK_ID")
        if not training_run_id:
            raise RuntimeError("training run ID is required for cluster-member identity")
        value = {
            "schema_version": 1,
            "training_run_id": training_run_id,
            "cluster_id": self.cluster_id or None,
            "node_count": self.n_nodes,
            "rank": self.rank,
            "container_ipv4_ip": self.node_ip,
            "task_id": task_id,
        }
        print(
            f"{MODAL_CLUSTER_MEMBER_EVENT} "
            f"{json.dumps(value, sort_keys=True, separators=(',', ':'))}",
            flush=True,
        )
        return value

    def start_ray(
        self,
        *,
        init_retries: int = 30,
        worker_wait_retries: int = 60,
    ) -> None:
        """Start the Ray head or worker using previously-discovered cluster state.

        Requires `discover_cluster()` to have been called. On the head also
        creates a `JobSubmissionClient`. Idempotent.
        """
        if self._started:
            return
        if not self._discovered:
            raise RuntimeError("discover_cluster() must be called before start_ray()")

        if self.rank == 0:
            start_ray_head(
                self.node_ip,
                self.n_nodes,
                init_retries=init_retries,
                worker_wait_retries=worker_wait_retries,
                extra_start_args=self.head_extra_start_args(),
            )
            from ray.job_submission import JobSubmissionClient

            self._client = JobSubmissionClient(f"http://127.0.0.1:{RAY_DASHBOARD_PORT}")
        else:
            start_ray_worker(
                self.node_ip,
                self.head_addr,
                connect_retries=worker_wait_retries,
                extra_start_args=self.worker_extra_start_args(),
            )
        self._started = True

    def start(
        self,
        n_nodes: int,
        *,
        init_retries: int = 30,
        worker_wait_retries: int = 60,
    ) -> None:
        """Convenience: `discover_cluster(n_nodes)` followed by `start_ray(...)`."""
        self.discover_cluster(n_nodes)
        self.start_ray(
            init_retries=init_retries,
            worker_wait_retries=worker_wait_retries,
        )

    def forward_dashboard(self):
        """Return a `modal.forward` context manager for the Ray dashboard.

        Usable with either `with` or `async with`. Only valid on the head node.
        """
        import modal

        if not self.is_head:
            raise RuntimeError("forward_dashboard is only valid on the head node")
        return modal.forward(RAY_DASHBOARD_PORT)

    async def submit_and_tail(
        self,
        entrypoint: str,
        *,
        runtime_env: dict | None = None,
        max_retries: int = 35,
    ) -> ModalRayJobResult:
        """Submit a Ray job, stream its logs to stdout, and return the final status."""
        if not self.is_head:
            raise RuntimeError("submit_and_tail is only valid on the head node")
        assert self._client is not None
        job_id = self._client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_env or {},
        )
        self.last_submitted_job_id = job_id
        print(f"Submitted Ray job: {job_id}")

        _TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED"}
        retry_count = 0

        while retry_count < max_retries:
            log_stream = self._client.tail_job_logs(job_id)
            if inspect.isawaitable(log_stream):
                log_stream = await log_stream
            if hasattr(log_stream, "__aiter__"):
                async for line in log_stream:
                    print(line, end="", flush=True)
            else:
                for line in log_stream:
                    print(line, end="", flush=True)

            status = self._client.get_job_status(job_id).value
            if status in _TERMINAL:
                break
            retry_count += 1
            print(
                f"\n[ray] Log stream ended but job {job_id} is still {status}; "
                f"reconnecting in 2s... (retry {retry_count}/{max_retries})"
            )
            await asyncio.sleep(2)
        else:
            raise RuntimeError(
                f"Ray job {job_id} log stream disconnected {max_retries} times "
                f"without reaching terminal status (current: {status})"
            )

        print(f"\nFinal Ray job status: {status}")
        if status != "SUCCEEDED":
            # Surface Ray's recorded driver failure message in the exception
            # itself: the real traceback is streamed above but is easily buried
            # in rollout logs and is lost once logs roll off after termination.
            message = None
            try:
                info = self._client.get_job_info(job_id)
                if inspect.isawaitable(info):
                    info = await info
                message = getattr(info, "message", None) or None
            except Exception:  # noqa: BLE001 — best-effort enrichment
                pass
            diagnostics = capture_ray_cluster_diagnostics()
            suffix = f": {message}" if message else ""
            print(f"Ray job {job_id} finished with status: {status}{suffix}")
            return ModalRayJobResult(
                status=status,
                is_success=status == "SUCCEEDED",
                message=message,
                diagnostics=diagnostics,
                job_id=job_id,
            )
        return ModalRayJobResult(
            status=status,
            is_success=status == "SUCCEEDED",
            job_id=job_id,
        )

    async def wait_forever(self, poll_seconds: float = 10) -> None:
        """Keep a worker container alive until the head terminates the cluster."""
        while True:
            await asyncio.sleep(poll_seconds)
