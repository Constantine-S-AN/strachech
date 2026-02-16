"""Multiprocess orchestration for running multiple paper runners in isolation."""

from __future__ import annotations

import importlib
import multiprocessing as mp
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Any, Literal

import pandas as pd

from stratcheck.connectors import LivePaperRunner, PaperBrokerConnector

RunnerExecutionStatus = Literal["completed", "failed", "timeout"]


@dataclass(slots=True)
class RunnerResourceLimits:
    """Unified resource limits that can be applied per runner process."""

    max_runtime_seconds: float | None = None
    max_memory_mb: int | None = None
    max_cpu_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            msg = "max_runtime_seconds must be positive when provided."
            raise ValueError(msg)
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            msg = "max_memory_mb must be positive when provided."
            raise ValueError(msg)
        if self.max_cpu_seconds is not None and self.max_cpu_seconds <= 0:
            msg = "max_cpu_seconds must be positive when provided."
            raise ValueError(msg)


@dataclass(slots=True)
class RunnerTask:
    """One runner execution task."""

    task_id: str
    strategy_reference: str
    bars: pd.DataFrame
    symbol: str = "DEMO"
    namespace: str | None = None
    strategy_params: dict[str, Any] = field(default_factory=dict)
    connector_params: dict[str, Any] = field(default_factory=dict)
    runner_params: dict[str, Any] = field(default_factory=dict)
    resource_limits: RunnerResourceLimits | None = None

    def __post_init__(self) -> None:
        normalized_task_id = str(self.task_id).strip()
        if not normalized_task_id:
            msg = "task_id cannot be blank."
            raise ValueError(msg)
        self.task_id = normalized_task_id

        normalized_reference = str(self.strategy_reference).strip()
        if not normalized_reference:
            msg = "strategy_reference cannot be blank."
            raise ValueError(msg)
        if ":" not in normalized_reference:
            msg = "strategy_reference must be '<module>:<ClassName>'."
            raise ValueError(msg)
        self.strategy_reference = normalized_reference

        if not isinstance(self.bars, pd.DataFrame):
            msg = "bars must be a pandas DataFrame."
            raise ValueError(msg)

        self.symbol = str(self.symbol).upper()
        if self.namespace is not None:
            self.namespace = str(self.namespace).strip() or None

        self.strategy_params = dict(self.strategy_params)
        self.connector_params = dict(self.connector_params)
        self.runner_params = dict(self.runner_params)


@dataclass(slots=True, frozen=True)
class RunnerNamespacePaths:
    """Filesystem namespace for one runner."""

    namespace: str
    root_dir: Path
    sqlite_path: Path
    log_path: Path
    metrics_prom_path: Path
    metrics_csv_path: Path


@dataclass(slots=True)
class RunnerTaskResult:
    """Execution summary for one task managed by the orchestrator."""

    task_id: str
    namespace: str
    status: RunnerExecutionStatus
    runner_status: str | None
    run_id: str | None
    processed_bars: int
    orders_placed: int
    updates_written: int
    risk_events: int
    kill_reason: str | None
    sqlite_path: Path
    log_path: Path
    metrics_prom_path: Path
    metrics_csv_path: Path
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None
    exit_code: int | None = None


@dataclass(slots=True)
class _ActiveRunnerProcess:
    task: RunnerTask
    paths: RunnerNamespacePaths
    limits: RunnerResourceLimits
    process: mp.Process
    result_queue: mp.Queue
    started_monotonic: float


class MultiRunnerOrchestrator:
    """Manage multiple runner processes with namespace isolation."""

    def __init__(
        self,
        output_dir: str | Path = "reports/orchestrator",
        *,
        max_concurrent_runners: int | None = None,
        default_resource_limits: RunnerResourceLimits | None = None,
        process_start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
        poll_interval_seconds: float = 0.05,
        clean_namespace_dir: bool = False,
    ) -> None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        if max_concurrent_runners is None:
            cpu_count = mp.cpu_count()
            resolved_max_concurrent = max(cpu_count, 1)
        else:
            resolved_max_concurrent = int(max_concurrent_runners)
            if resolved_max_concurrent < 1:
                msg = "max_concurrent_runners must be >= 1."
                raise ValueError(msg)

        if poll_interval_seconds <= 0:
            msg = "poll_interval_seconds must be positive."
            raise ValueError(msg)

        self.output_dir = resolved_output_dir
        self.max_concurrent_runners = resolved_max_concurrent
        self.default_resource_limits = default_resource_limits or RunnerResourceLimits()
        self.process_start_method = process_start_method
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.clean_namespace_dir = bool(clean_namespace_dir)

    def run_tasks(self, tasks: list[RunnerTask]) -> list[RunnerTaskResult]:
        """Execute tasks with process-level isolation and resource controls."""
        if not tasks:
            return []

        self._validate_tasks(tasks)
        namespace_map = self._build_namespace_paths(tasks)
        process_context = mp.get_context(self.process_start_method)

        pending_index = 0
        active: dict[str, _ActiveRunnerProcess] = {}
        results_by_task_id: dict[str, RunnerTaskResult] = {}

        while pending_index < len(tasks) or active:
            while pending_index < len(tasks) and len(active) < self.max_concurrent_runners:
                task = tasks[pending_index]
                pending_index += 1

                merged_limits = _merge_resource_limits(
                    defaults=self.default_resource_limits,
                    overrides=task.resource_limits,
                )
                paths = namespace_map[task.task_id]
                result_queue: mp.Queue = process_context.Queue(maxsize=1)
                process = process_context.Process(
                    target=_runner_process_entrypoint,
                    args=(task, paths, merged_limits, result_queue),
                    name=f"stratcheck-runner-{task.task_id}",
                    daemon=False,
                )
                process.start()
                active[task.task_id] = _ActiveRunnerProcess(
                    task=task,
                    paths=paths,
                    limits=merged_limits,
                    process=process,
                    result_queue=result_queue,
                    started_monotonic=time.monotonic(),
                )

            completed_task_ids: list[str] = []
            for task_id, state in active.items():
                process = state.process
                elapsed_seconds = time.monotonic() - state.started_monotonic

                timeout_limit = state.limits.max_runtime_seconds
                if (
                    timeout_limit is not None
                    and elapsed_seconds > timeout_limit
                    and process.is_alive()
                ):
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)

                    results_by_task_id[task_id] = RunnerTaskResult(
                        task_id=state.task.task_id,
                        namespace=state.paths.namespace,
                        status="timeout",
                        runner_status=None,
                        run_id=None,
                        processed_bars=0,
                        orders_placed=0,
                        updates_written=0,
                        risk_events=0,
                        kill_reason=None,
                        sqlite_path=state.paths.sqlite_path,
                        log_path=state.paths.log_path,
                        metrics_prom_path=state.paths.metrics_prom_path,
                        metrics_csv_path=state.paths.metrics_csv_path,
                        duration_seconds=float(elapsed_seconds),
                        error_type="TimeoutError",
                        error_message=(
                            f"runner exceeded max_runtime_seconds={timeout_limit:.3f}"
                        ),
                        exit_code=process.exitcode,
                    )
                    completed_task_ids.append(task_id)
                    continue

                if process.is_alive():
                    continue

                process.join(timeout=0.1)
                payload = _read_process_payload(state.result_queue)
                if payload is not None:
                    results_by_task_id[task_id] = _payload_to_result(
                        payload=payload,
                        task_id=state.task.task_id,
                        paths=state.paths,
                    )
                else:
                    results_by_task_id[task_id] = RunnerTaskResult(
                        task_id=state.task.task_id,
                        namespace=state.paths.namespace,
                        status="failed",
                        runner_status=None,
                        run_id=None,
                        processed_bars=0,
                        orders_placed=0,
                        updates_written=0,
                        risk_events=0,
                        kill_reason=None,
                        sqlite_path=state.paths.sqlite_path,
                        log_path=state.paths.log_path,
                        metrics_prom_path=state.paths.metrics_prom_path,
                        metrics_csv_path=state.paths.metrics_csv_path,
                        duration_seconds=float(elapsed_seconds),
                        error_type="ProcessExitError",
                        error_message="runner process exited without result payload",
                        exit_code=process.exitcode,
                    )
                completed_task_ids.append(task_id)

            for task_id in completed_task_ids:
                state = active.pop(task_id)
                state.result_queue.close()
                state.result_queue.join_thread()

            if active:
                time.sleep(self.poll_interval_seconds)

        return [results_by_task_id[item.task_id] for item in tasks]

    def _validate_tasks(self, tasks: list[RunnerTask]) -> None:
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            msg = "task_id values must be unique."
            raise ValueError(msg)

        namespaces = [_effective_namespace(task) for task in tasks]
        if len(namespaces) != len(set(namespaces)):
            msg = "namespace values must be unique after normalization."
            raise ValueError(msg)

    def _build_namespace_paths(self, tasks: list[RunnerTask]) -> dict[str, RunnerNamespacePaths]:
        paths_by_task_id: dict[str, RunnerNamespacePaths] = {}
        for task in tasks:
            namespace = _effective_namespace(task)
            namespace_dir = self.output_dir / namespace
            if namespace_dir.exists() and self.clean_namespace_dir:
                shutil.rmtree(namespace_dir)
            namespace_dir.mkdir(parents=True, exist_ok=True)

            paths_by_task_id[task.task_id] = RunnerNamespacePaths(
                namespace=namespace,
                root_dir=namespace_dir,
                sqlite_path=namespace_dir / "paper_trading.sqlite",
                log_path=namespace_dir / "paper_runner.jsonl",
                metrics_prom_path=namespace_dir / "paper_metrics.prom",
                metrics_csv_path=namespace_dir / "paper_metrics.csv",
            )
        return paths_by_task_id


def _runner_process_entrypoint(
    task: RunnerTask,
    paths: RunnerNamespacePaths,
    limits: RunnerResourceLimits,
    result_queue: mp.Queue,
) -> None:
    start_time = time.monotonic()
    try:
        _apply_resource_limits(limits)

        strategy = _load_strategy(
            strategy_reference=task.strategy_reference,
            strategy_params=task.strategy_params,
        )
        connector = PaperBrokerConnector(**task.connector_params)
        runner = LivePaperRunner(
            connector=connector,
            sqlite_path=paths.sqlite_path,
            symbol=task.symbol,
            log_path=paths.log_path,
            metrics_prom_path=paths.metrics_prom_path,
            metrics_csv_path=paths.metrics_csv_path,
            **task.runner_params,
        )
        run_result = runner.run(strategy=strategy, bars=task.bars)

        result_queue.put(
            {
                "status": "completed",
                "runner_status": run_result.status,
                "run_id": run_result.run_id,
                "processed_bars": int(run_result.processed_bars),
                "orders_placed": int(run_result.orders_placed),
                "updates_written": int(run_result.updates_written),
                "risk_events": int(run_result.risk_events),
                "kill_reason": run_result.kill_reason,
                "duration_seconds": float(time.monotonic() - start_time),
                "error_type": None,
                "error_message": None,
                "exit_code": 0,
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "status": "failed",
                "runner_status": None,
                "run_id": None,
                "processed_bars": 0,
                "orders_placed": 0,
                "updates_written": 0,
                "risk_events": 0,
                "kill_reason": None,
                "duration_seconds": float(time.monotonic() - start_time),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "exit_code": 1,
            }
        )


def _load_strategy(strategy_reference: str, strategy_params: dict[str, Any]) -> Any:
    module_name, class_name = strategy_reference.split(":", 1)
    module_obj = importlib.import_module(module_name)
    strategy_class = getattr(module_obj, class_name, None)
    if strategy_class is None:
        msg = (
            "strategy class not found: "
            f"{strategy_reference!r}. Expected class {class_name!r} in module {module_name!r}."
        )
        raise ValueError(msg)
    return strategy_class(**dict(strategy_params))


def _read_process_payload(result_queue: mp.Queue) -> dict[str, Any] | None:
    try:
        return result_queue.get(timeout=0.2)
    except Empty:
        return None


def _payload_to_result(
    payload: dict[str, Any],
    *,
    task_id: str,
    paths: RunnerNamespacePaths,
) -> RunnerTaskResult:
    return RunnerTaskResult(
        task_id=task_id,
        namespace=paths.namespace,
        status=str(payload.get("status", "failed")),  # type: ignore[arg-type]
        runner_status=_as_optional_str(payload.get("runner_status")),
        run_id=_as_optional_str(payload.get("run_id")),
        processed_bars=int(payload.get("processed_bars", 0)),
        orders_placed=int(payload.get("orders_placed", 0)),
        updates_written=int(payload.get("updates_written", 0)),
        risk_events=int(payload.get("risk_events", 0)),
        kill_reason=_as_optional_str(payload.get("kill_reason")),
        sqlite_path=paths.sqlite_path,
        log_path=paths.log_path,
        metrics_prom_path=paths.metrics_prom_path,
        metrics_csv_path=paths.metrics_csv_path,
        duration_seconds=float(payload.get("duration_seconds", 0.0)),
        error_type=_as_optional_str(payload.get("error_type")),
        error_message=_as_optional_str(payload.get("error_message")),
        exit_code=_as_optional_int(payload.get("exit_code")),
    )


def _merge_resource_limits(
    defaults: RunnerResourceLimits,
    overrides: RunnerResourceLimits | None,
) -> RunnerResourceLimits:
    if overrides is None:
        return RunnerResourceLimits(
            max_runtime_seconds=defaults.max_runtime_seconds,
            max_memory_mb=defaults.max_memory_mb,
            max_cpu_seconds=defaults.max_cpu_seconds,
        )
    return RunnerResourceLimits(
        max_runtime_seconds=(
            overrides.max_runtime_seconds
            if overrides.max_runtime_seconds is not None
            else defaults.max_runtime_seconds
        ),
        max_memory_mb=(
            overrides.max_memory_mb
            if overrides.max_memory_mb is not None
            else defaults.max_memory_mb
        ),
        max_cpu_seconds=(
            overrides.max_cpu_seconds
            if overrides.max_cpu_seconds is not None
            else defaults.max_cpu_seconds
        ),
    )


def _apply_resource_limits(limits: RunnerResourceLimits) -> None:
    try:
        import resource
    except ImportError:
        return

    if limits.max_memory_mb is not None:
        memory_bytes = int(limits.max_memory_mb * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    if limits.max_cpu_seconds is not None:
        cpu_seconds = int(limits.max_cpu_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))


def _effective_namespace(task: RunnerTask) -> str:
    raw_namespace = task.namespace if task.namespace is not None else task.task_id
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(raw_namespace).strip()
    )
    if not normalized:
        msg = f"namespace for task_id={task.task_id!r} resolves to blank."
        raise ValueError(msg)
    return normalized


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "MultiRunnerOrchestrator",
    "RunnerExecutionStatus",
    "RunnerNamespacePaths",
    "RunnerResourceLimits",
    "RunnerTask",
    "RunnerTaskResult",
]
