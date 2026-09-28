"""Model configuration page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.diffusion.trainer import estimate_memory_mb, resolve_device
from src.ui.state import go


def render(state: dict[str, Any]) -> None:
    st.title("Model Configuration")
    if state.get("demo_mode"):
        st.info("Demo Mode – Reduced training configuration is active. You can still edit the values below.")
    config = state["model_config"]
    prepared = state.get("prepared")
    if st.button("Use recommended configuration"):
        _recommend(state)
        st.rerun()

    st.markdown("### Diffusion")
    left, right = st.columns(2)
    with left:
        config["sde"] = st.selectbox("SDE type", ["vp"], index=0, help="Variance-preserving SDE.")
        config["schedule"] = st.selectbox("Noise schedule", ["linear", "cosine"], index=0 if config.get("schedule") == "linear" else 1)
        config["steps"] = st.number_input("Diffusion steps", min_value=10, max_value=2000, value=int(config.get("steps", 200)), step=10, help="Discrete time grid used during training. Sampling can use its own step count.")
    with right:
        config["sampler"] = st.selectbox("Sampler", ["reverse_sde", "probability_flow"], index=0 if config.get("sampler") == "reverse_sde" else 1)
        config["time_sampling"] = st.selectbox("Training time sampling", ["discrete", "continuous"], index=0 if config.get("time_sampling", "discrete") == "discrete" else 1)
        config["beta_min"] = st.number_input("Beta min", value=float(config.get("beta_min", 0.1)), format="%.3f")
        config["beta_max"] = st.number_input("Beta max", value=float(config.get("beta_max", 20.0)), format="%.3f")

    st.markdown("### Network")
    net = st.columns(3)
    config["hidden_dim"] = int(net[0].number_input("Hidden dimensions", min_value=32, max_value=1024, value=int(config.get("hidden_dim", 256)), step=32))
    config["layers"] = int(net[1].number_input("Number of layers", min_value=1, max_value=8, value=int(config.get("layers", 4))))
    config["dropout"] = float(net[2].number_input("Dropout", min_value=0.0, max_value=0.5, value=float(config.get("dropout", 0.1)), step=0.05))

    st.markdown("### Training")
    train = st.columns(3)
    config["epochs"] = int(train[0].number_input("Epochs", min_value=1, max_value=500, value=int(config.get("epochs", 80))))
    config["batch_size"] = int(train[1].number_input("Batch size", min_value=8, max_value=4096, value=int(config.get("batch_size", 256)), step=8))
    config["learning_rate"] = float(train[2].number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=float(config.get("learning_rate", 1e-4)), format="%.6f"))
    train2 = st.columns(3)
    config["weight_decay"] = float(train2[0].number_input("Weight decay", min_value=0.0, max_value=0.1, value=float(config.get("weight_decay", 1e-4)), format="%.6f"))
    config["seed"] = int(train2[1].number_input("Random seed", min_value=0, max_value=100000, value=int(config.get("seed", 42))))
    config["early_stopping_patience"] = int(train2[2].number_input("Early stopping patience", min_value=0, max_value=50, value=int(config.get("early_stopping_patience", 0)), help="0 disables early stopping."))

    st.markdown("### Hardware")
    config["device"] = st.selectbox("Device", ["auto", "cpu", "cuda"], index=["auto", "cpu", "cuda"].index(str(config.get("device", "auto"))))
    try:
        device = resolve_device(config["device"] if config["device"] != "cuda" else "auto")
        st.caption(f"Resolved device preview: {device}. Selecting CUDA fails at training time if no GPU is available.")
    except Exception as exc:
        st.caption(str(exc))
    feature_dim = prepared["feature_dim"] if prepared else 32
    estimate = estimate_memory_mb(feature_dim, config["hidden_dim"], config["layers"], config["batch_size"])
    st.write(
        {
            "Feature dimension": feature_dim if prepared else "Available after preprocessing",
            "Approximate parameter count": int(estimate["parameters"]),
            "Rough training memory (MB)": round(estimate["estimated_training_mb"], 1),
        }
    )

    columns = ["(none)"]
    if prepared:
        columns += list(prepared["output_columns"])
    current = config.get("condition_column") or "(none)"
    if current not in columns:
        current = "(none)"
    selected = st.selectbox(
        "Condition column for classifier-free training",
        columns,
        index=columns.index(current),
        help="Optional. The clean encoding of this column is passed to the score network, with dropout so unconditional sampling still works.",
    )
    config["condition_column"] = None if selected == "(none)" else selected
    config["balance_batches"] = st.checkbox(
        "Balance training batches using the condition or target column",
        value=bool(config.get("balance_batches", False)),
        help="Inverse-frequency sampling changes the training distribution. The effect is recorded in the experiment.",
    )
    with st.expander("What this configuration does"):
        st.markdown(
            """
The score network is trained with denoising score matching on the continuous representation
produced by preprocessing. Diffusion steps discretize time during training when time sampling is discrete.
Sampling later integrates the reverse SDE or the probability-flow ODE for the selected number of steps.
            """
        )
    if st.button("Continue to training"):
        go("Training")


def _recommend(state: dict[str, Any]) -> None:
    rows = len(state["frame"]) if state.get("frame") is not None else 1000
    config = state["model_config"]
    if rows < 2000 or state.get("demo_mode"):
        config.update({"epochs": 20, "batch_size": 64, "hidden_dim": 128, "layers": 3, "steps": 80, "sampling_steps": 80})
    elif rows < 20000:
        config.update({"epochs": 60, "batch_size": 128, "hidden_dim": 256, "layers": 4, "steps": 200, "sampling_steps": 150})
    else:
        config.update({"epochs": 40, "batch_size": 256, "hidden_dim": 256, "layers": 4, "steps": 200, "sampling_steps": 100})
