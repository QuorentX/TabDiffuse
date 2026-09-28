"""Training page with a background job and a live loss chart."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config.settings import project_root
from src.diffusion.trainer import resolve_device
from src.services.experiments import ExperimentStore
from src.services.jobs import get_job
from src.services.pipeline import inverse_frequency_weights, load_bundle, train_model
from src.ui.components import card, show_exception
from src.utils.file_utils import humanize_seconds
from src.visualization.distributions import loss_figure


def _publish_epoch(active_job, snapshot: dict[str, Any]) -> None:
    """Copy an epoch snapshot onto the job without passing ``state`` twice.

    The trainer snapshot already contains ``state``. Spreading it and also
    passing ``state=`` raises TypeError before the first epoch can be recorded.
    """
    fields = dict(snapshot)
    fields["message"] = f"Epoch {snapshot['epoch']} complete"
    fields["state"] = "running"
    active_job.update(**fields)


def render(state: dict[str, Any]) -> None:
    st.title("Training")
    if state.get("demo_mode"):
        st.info("Demo Mode – Reduced training configuration")
    prepared = state.get("prepared")
    if prepared is None:
        st.info("Apply preprocessing before training.")
        return
    config = state["model_config"]
    job = get_job(st.session_state.session_key)
    status = job.snapshot()
    _hardware(config, status)
    buttons = st.columns(4)
    if buttons[0].button("Start training", type="primary"):
        _start(state, job)
    if buttons[1].button("Pause"):
        job.request_pause()
    if buttons[2].button("Resume"):
        job.request_resume()
    if buttons[3].button("Stop training"):
        job.request_stop()
    if st.button("Resume from checkpoint", help="Continues the current experiment. Increase the epoch count first."):
        _resume(state, job)
    if st.button("Save checkpoint snapshot"):
        _snapshot(state)
    _monitor(state)
    summary = state.get("training_summary")
    if summary:
        st.success("Training completed" if summary.get("state") == "completed" else f"Training status: {summary.get('state')}")
        columns = st.columns(3)
        with columns[0]:
            card("Best validation loss", f"{summary.get('best_val_loss'):.4f}" if summary.get("best_val_loss") is not None else "—", "")
        with columns[1]:
            card("Duration", humanize_seconds(summary.get("duration_seconds")), "")
        with columns[2]:
            card("Checkpoint", "Saved", summary.get("checkpoint") or "")


@st.fragment(run_every=1)
def _monitor(state: dict[str, Any]) -> None:
    job = get_job(st.session_state.session_key)
    status = job.snapshot()
    epochs = max(int(status.get("epochs") or 1), 1)
    epoch = int(status.get("epoch") or 0)
    st.progress(min(epoch / epochs, 1.0), text=f"Epoch {epoch} / {status.get('epochs') or '—'}")
    columns = st.columns(4)
    columns[0].metric("Training loss", "—" if status.get("train_loss") is None else f"{status['train_loss']:.4f}")
    columns[1].metric("Validation loss", "—" if status.get("val_loss") is None else f"{status['val_loss']:.4f}")
    columns[2].metric("Elapsed", humanize_seconds(status.get("elapsed_seconds")))
    columns[3].metric("Remaining", humanize_seconds(status.get("remaining_seconds")))
    st.caption(status.get("message") or status.get("state") or "Idle")
    if status.get("error"):
        st.error(status["error"])
    history = status.get("history") or {}
    if history.get("train_loss"):
        st.plotly_chart(loss_figure(history), use_container_width=True)
    result = status.get("result")
    if result and not status.get("published_to_session") and state.get("experiment_id"):
        try:
            state["training_summary"] = {key: value for key, value in result.items() if key != "history"}
            state["training_summary"]["history"] = result.get("history")
            state["bundle"] = load_bundle(
                project_root() / "models" / "checkpoints" / state["experiment_id"] / "checkpoint_best.pt",
                state["model_config"].get("device", "cpu"),
            )
            store = ExperimentStore(project_root())
            store.update(
                state["experiment_id"],
                {"status": result.get("state", "trained"), "training": state["training_summary"]},
            )
            state["experiment"] = store.load(state["experiment_id"])
        except Exception as exc:
            show_exception(exc, "publish_training")
        finally:
            job.update(published_to_session=True)


def _start(state: dict[str, Any], job) -> None:
    try:
        prepared = state["prepared"]
        config = dict(state["model_config"])
        preprocessor = prepared["preprocessor"]
        if config.get("condition_column"):
            config["condition_indices"] = preprocessor.column_indices(config["condition_column"])
        else:
            config["condition_indices"] = []
        state["model_config"] = config
        store = ExperimentStore(project_root())
        record = store.create(
            {
                "dataset_name": state.get("dataset_name"),
                "rows": int(len(state["frame"])),
                "columns": int(state["frame"].shape[1]),
                "seed": int(config.get("seed", 42)),
                "status": "training",
                "demo_mode": bool(state.get("demo_mode")),
                "model_config": config,
                "preprocessing": prepared["metadata"],
                "profile_warnings": (state.get("profile") or {}).get("warnings", [])[:30],
                "notes": state.get("notes"),
                "split": {
                    "train_rows": prepared["splits"]["train_rows"],
                    "validation_rows": prepared["splits"]["validation_rows"],
                    "holdout_rows": prepared["splits"]["holdout_rows"],
                },
            }
        )
        state["experiment_id"] = record["experiment_id"]
        state["experiment"] = record
        weights = None
        balance_column = config.get("condition_column") or state.get("target")
        if config.get("balance_batches") and balance_column in prepared["splits"]["train"].columns:
            weights = inverse_frequency_weights(prepared["splits"]["train"][balance_column])
        train_x = prepared["arrays"]["train"]
        val_x = prepared["arrays"]["validation"]

        def worker(active_job):
            def on_epoch(snapshot: dict[str, Any]) -> None:
                _publish_epoch(active_job, snapshot)

            return train_model(
                train_x,
                val_x,
                config,
                root=project_root(),
                experiment_id=record["experiment_id"],
                preprocessor=preprocessor,
                sample_weights=weights,
                stop_event=active_job.stop_event,
                pause_event=active_job.pause_event,
                on_epoch=on_epoch,
                mlflow_enabled=state["config"].mlflow.enabled,
                mlflow_uri=str(project_root() / state["config"].mlflow.tracking_uri),
            )

        job.start(worker)
        st.success(f"Training started as {record['experiment_id']}. Loss updates once per epoch.")
    except Exception as exc:
        show_exception(exc, "start_training")


def _resume(state: dict[str, Any], job) -> None:
    if not state.get("experiment_id") or state.get("prepared") is None:
        st.warning("Train an experiment before resuming it.")
        return
    try:
        from src.services.pipeline import resume_model

        prepared = state["prepared"]
        config = dict(state["model_config"])

        def worker(active_job):
            def on_epoch(snapshot: dict[str, Any]) -> None:
                _publish_epoch(active_job, snapshot)

            return resume_model(
                prepared["arrays"]["train"],
                prepared["arrays"]["validation"],
                config,
                root=project_root(),
                experiment_id=state["experiment_id"],
                stop_event=active_job.stop_event,
                pause_event=active_job.pause_event,
                on_epoch=on_epoch,
            )

        job.start(worker)
        st.success(f"Resuming {state['experiment_id']}.")
    except Exception as exc:
        show_exception(exc, "resume_training")


def _snapshot(state: dict[str, Any]) -> None:
    summary = state.get("training_summary")
    if not summary or not state.get("experiment_id"):
        st.warning("There is no completed checkpoint to snapshot yet.")
        return
    source = project_root() / "models" / "checkpoints" / state["experiment_id"] / "checkpoint_latest.pt"
    if not source.exists():
        st.warning("The latest checkpoint file is not on disk yet.")
        return
    from datetime import datetime

    target = source.with_name(f"snapshot_{datetime.now().strftime('%H%M%S')}.pt")
    target.write_bytes(source.read_bytes())
    st.success(f"Snapshot written to {target.name}. Existing experiments were not overwritten.")


def _hardware(config: dict[str, Any], status: dict[str, Any]) -> None:
    try:
        import psutil
        import torch

        memory = psutil.virtual_memory()
        device = str(resolve_device("auto"))
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device"
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = round(torch.cuda.memory_allocated() / (1024**2), 1)
        columns = st.columns(3)
        with columns[0]:
            card("Device setting", str(config.get("device", "auto")), device)
        with columns[1]:
            card("GPU", gpu, "" if gpu_mem is None else f"{gpu_mem} MB allocated")
        with columns[2]:
            card("System memory", f"{memory.percent:.0f}% used", status.get("state", "idle"))
    except Exception:
        st.caption("Hardware details are unavailable in this environment.")
