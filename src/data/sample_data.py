"""Synthetic demonstration table. This is not real customer data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import project_root


def generate_sample_dataset(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Simulate a mixed-type lending table with a learnable approval signal.

    The result is demonstration data. It should not be described as a real extract.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 67, size=n_rows)
    education = rng.choice(
        ["High School", "Bachelor", "Master", "Doctorate"],
        size=n_rows,
        p=[0.30, 0.42, 0.20, 0.08],
    )
    edu_score = pd.Series(education).map(
        {"High School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3}
    ).to_numpy()
    gender = rng.choice(["Female", "Male", "Non-binary"], size=n_rows, p=[0.47, 0.49, 0.04])
    occupation = rng.choice(
        ["Analyst", "Teacher", "Engineer", "Nurse", "Sales", "Technician", "Manager", "Designer"],
        size=n_rows,
    )
    city = rng.choice(
        ["Austin", "Denver", "Seattle", "Chicago", "Atlanta", "Boston", "Phoenix", "Miami"],
        size=n_rows,
    )
    marital_status = rng.choice(["Single", "Married", "Divorced"], size=n_rows, p=[0.42, 0.46, 0.12])
    employment = np.clip(rng.normal((age - 20) * 0.42, 2.8), 0, np.maximum(age - 18, 0))
    income = (
        22000
        + age * 480
        + edu_score * 8500
        + employment * 1200
        + rng.normal(0, 6500, size=n_rows)
    )
    income = np.clip(income, 14000, None)
    credit = np.clip(
        520 + income / 2500 + employment * 3.5 + edu_score * 12 - (age - 40) * 0.3 + rng.normal(0, 32, size=n_rows),
        320,
        845,
    )
    expense_ratio = rng.uniform(0.28, 0.72, size=n_rows)
    monthly_expense = np.clip(income * expense_ratio / 12 + rng.normal(0, 180, size=n_rows), 400, None)
    loan_amount = np.clip(income * rng.uniform(0.5, 2.2, size=n_rows), 1500, None)
    logit = (
        -1.5
        + (credit - 640) / 70
        + (income - 55000) / 35000
        - (monthly_expense / (income / 12)) * 1.4
        + employment * 0.04
    )
    approval = rng.random(n_rows) < 1 / (1 + np.exp(-logit))
    start = np.datetime64("2015-01-01")
    offsets = rng.integers(0, 365 * 9, size=n_rows)
    dates = start + offsets.astype("timedelta64[D]")

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:06d}" for index in range(1, n_rows + 1)],
            "customer_age": age.astype(int),
            "gender": gender,
            "income": np.round(income, 2),
            "education": education,
            "occupation": occupation,
            "city": city,
            "credit_score": np.rint(credit).astype(int),
            "loan_amount": np.round(loan_amount, 2),
            "employment_years": np.round(employment, 1),
            "marital_status": marital_status,
            "loan_status": np.where(approval, "Approved", "Rejected"),
            "monthly_expense": np.round(monthly_expense, 2),
            "account_open_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        }
    )
    missing_idx = rng.choice(n_rows, size=max(1, n_rows // 40), replace=False)
    frame.loc[frame.index[missing_idx[: len(missing_idx) // 2]], "income"] = np.nan
    frame.loc[frame.index[missing_idx[len(missing_idx) // 2 :]], "city"] = np.nan
    duplicate_source = frame.sample(n=max(1, n_rows // 100), random_state=seed)
    frame = pd.concat([frame, duplicate_source], ignore_index=True)
    return frame


def generate_retail_dataset(n_rows: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Simulate a mixed-type retail table. This is not a real transaction extract."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 72, size=n_rows)
    membership = rng.choice(["Basic", "Silver", "Gold"], size=n_rows, p=[0.48, 0.34, 0.18])
    member_score = pd.Series(membership).map({"Basic": 0, "Silver": 1, "Gold": 2}).to_numpy()
    city = rng.choice(
        ["Austin", "Denver", "Seattle", "Chicago", "Atlanta", "Boston"],
        size=n_rows,
    )
    category = rng.choice(
        ["Electronics", "Grocery", "Apparel", "Home", "Beauty"],
        size=n_rows,
        p=[0.22, 0.28, 0.24, 0.16, 0.10],
    )
    channel = rng.choice(["Online", "Store"], size=n_rows, p=[0.62, 0.38])
    orders = np.clip(rng.normal(6 + member_score * 3 + age * 0.02, 2.2), 1, None)
    discount = np.clip(rng.normal(8 + (category == "Apparel") * 4, 3.5), 0, 40)
    annual_spend = np.clip(
        400 + age * 18 + member_score * 350 + orders * 70 - discount * 8 + rng.normal(0, 180, size=n_rows),
        120,
        None,
    )
    returns = np.clip(rng.poisson(0.4 + discount / 30 + (channel == "Online") * 0.3), 0, 8).astype(int)
    logit = -1.1 + returns * 0.55 + discount / 25 - member_score * 0.45 - orders * 0.08
    churn = rng.random(n_rows) < 1 / (1 + np.exp(-logit))
    dates = np.datetime64("2018-01-01") + rng.integers(0, 365 * 6, size=n_rows).astype("timedelta64[D]")
    frame = pd.DataFrame(
        {
            "order_id": [f"R{index:06d}" for index in range(1, n_rows + 1)],
            "buyer_age": age.astype(int),
            "membership": membership,
            "city": city,
            "product_category": category,
            "channel": channel,
            "annual_spend": np.round(annual_spend, 2),
            "orders_last_year": np.rint(orders).astype(int),
            "discount_percent": np.round(discount, 1),
            "return_count": returns,
            "signup_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "churn_status": np.where(churn, "Churned", "Retained"),
        }
    )
    missing_idx = rng.choice(n_rows, size=max(1, n_rows // 40), replace=False)
    frame.loc[frame.index[missing_idx[: len(missing_idx) // 2]], "annual_spend"] = np.nan
    frame.loc[frame.index[missing_idx[len(missing_idx) // 2 :]], "city"] = np.nan
    duplicates = frame.sample(n=max(1, n_rows // 100), random_state=seed)
    return pd.concat([frame, duplicates], ignore_index=True)


DEMO_SETS: dict[str, dict[str, object]] = {
    "A": {
        "key": "A",
        "filename": "demo_set_a_lending.csv",
        "label": "Set A — Lending",
        "rows": 1000,
        "seed": 42,
        "target": "loan_status",
        "description": (
            "Simulated lending table for the main demonstration. It is not a real customer extract. "
            "loan_status depends on credit, income, expenses, and employment."
        ),
    },
    "B": {
        "key": "B",
        "filename": "demo_set_b_retail.csv",
        "label": "Set B — Retail",
        "rows": 1000,
        "seed": 7,
        "target": "churn_status",
        "description": (
            "Simulated retail table for a second processing run. It is not a real transaction extract. "
            "churn_status depends on returns, discount, membership, and order count."
        ),
    },
}


def demo_set_path(key: str) -> Path:
    spec = DEMO_SETS[key]
    return project_root() / "data" / "raw" / str(spec["filename"])


def write_sample_dataset(path: Path | None = None, n_rows: int = 1200, seed: int = 42) -> Path:
    destination = path or (project_root() / "data" / "raw" / "sample_customer_data.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_dataset(n_rows=n_rows, seed=seed).to_csv(destination, index=False)
    return destination


def write_demo_sets(n_rows: int = 1000) -> dict[str, Path]:
    """Write both demonstration tables used for testing and processing."""
    paths = {
        "A": demo_set_path("A"),
        "B": demo_set_path("B"),
    }
    paths["A"].parent.mkdir(parents=True, exist_ok=True)
    generate_sample_dataset(n_rows=n_rows, seed=int(DEMO_SETS["A"]["seed"])).to_csv(paths["A"], index=False)
    generate_retail_dataset(n_rows=n_rows, seed=int(DEMO_SETS["B"]["seed"])).to_csv(paths["B"], index=False)
    write_sample_dataset(n_rows=n_rows, seed=int(DEMO_SETS["A"]["seed"]))
    return paths


if __name__ == "__main__":
    for name, output in write_demo_sets().items():
        print(name, output)
