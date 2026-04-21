import importlib.util
import pathlib
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Customer 30-Day Spend",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def generate_demo_transactions(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = pd.Timestamp.today().normalize()
    customers = [f"C{str(i).zfill(4)}" for i in range(1, 31)]
    categories = ["Home", "Beauty", "Electronics", "Grocery", "Sports"]
    channels = ["Online", "Store", "Mobile"]
    rows = []
    for cid in customers:
        num_orders = rng.integers(10, 28)
        dates = rng.choice(
            pd.date_range(today - pd.Timedelta(days=200), today), size=num_orders
        )
        for dt in dates:
            rows.append(
                {
                    "customer_id": cid,
                    "order_date": dt,
                    "order_amount": max(rng.normal(80, 25), 5),
                    "category": rng.choice(categories),
                    "channel": rng.choice(channels, p=[0.55, 0.25, 0.2]),
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values("order_date").reset_index(drop=True)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


@st.cache_data
def load_transactions(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    date_col = _find_column(df, ["order_date", "invoice_date", "transaction_date"])
    amount_col = _find_column(df, ["order_amount", "amount", "totalamount", "sales"])
    customer_col = _find_column(df, ["customer_id", "cust_id", "user_id"])
    if not (date_col and amount_col and customer_col):
        raise ValueError(
            "Expected columns for customer_id, order_date, and order_amount (case-insensitive)."
        )
    df = df.rename(
        columns={
            date_col: "order_date",
            amount_col: "order_amount",
            customer_col: "customer_id",
        }
    )
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def build_customer_features(
    txns: pd.DataFrame, history_window: int = 90, target_horizon: int = 30
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if txns.empty:
        return pd.DataFrame(), pd.Timestamp.today()
    cutoff = txns["order_date"].max().normalize()
    recent_start = cutoff - pd.Timedelta(days=history_window)
    hist = txns[txns["order_date"] >= recent_start]
    base = (
        hist.groupby("customer_id")
        .agg(
            spend_30d_hist=("order_amount", lambda x: x[hist["order_date"] >= cutoff - pd.Timedelta(days=target_horizon)].sum()),
            spend_90d_hist=("order_amount", "sum"),
            orders_90d=("order_amount", "count"),
            avg_order_value=("order_amount", "mean"),
            last_purchase=("order_date", "max"),
        )
        .reset_index()
    )
    base["recency_days"] = (cutoff - base["last_purchase"]).dt.days
    return base, cutoff


# Features expected by the trained RandomForest model
MODEL_FEATURE_COLS = [
    "frequency",
    "total_spend",
    "avg_order_value",
    "total_quantity",
    "unique_products",
    "recency_days",
    "tenure_days",
]


def build_model_features_for_customer(raw_df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    """
    Build model-ready features for a single customer from the cleaned Kaggle dataset.

    This mirrors the logic used during model training / inference in src.inference.py.
    """
    cdf = raw_df[raw_df["CustomerID_str"] == customer_id].copy()
    if cdf.empty:
        return pd.DataFrame(columns=MODEL_FEATURE_COLS)

    last_date = cdf["InvoiceDate"].max()
    past_df = cdf[cdf["InvoiceDate"] <= last_date]

    features = pd.DataFrame(
        [
            {
                "frequency": past_df["InvoiceNo"].nunique(),
                "total_spend": past_df["TransactionAmount"].sum(),
                "avg_order_value": past_df["TransactionAmount"].mean(),
                "total_quantity": past_df["Quantity"].sum(),
                "unique_products": past_df["StockCode"].nunique(),
                # At prediction time we are at the customer's latest purchase date
                "recency_days": 0,
                "tenure_days": (last_date - past_df["InvoiceDate"].min()).days,
            }
        ]
    )

    return features


def predict_spend(row: pd.Series, model=None, feature_cols: list[str] | None = None) -> float:
    if model is not None and feature_cols:
        X = pd.DataFrame([row[feature_cols].tolist()], columns=feature_cols)
        return float(model.predict(X)[0])
    baseline = row.get("spend_30d_hist", 0.0)
    trend = row.get("spend_90d_hist", 0.0) / max(row.get("orders_90d", 1), 1)
    return float(max(baseline * 1.05, trend))


def load_local_model(module_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("local_model", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load model module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]

    feature_cols = getattr(module, "FEATURE_COLS", None)

    if hasattr(module, "get_model"):
        model = module.get_model()
        return model, feature_cols
    if hasattr(module, "model"):
        model = module.model
        return model, feature_cols
    if hasattr(module, "predict") and callable(module.predict):
        class _Wrapper:
            def predict(self, X):
                return module.predict(X)

        return _Wrapper(), feature_cols

    raise ValueError("Module must expose predict(X) or model/get_model().")


def render_customer_snapshot(txns: pd.DataFrame, features: pd.DataFrame, customer_id: str):
    c_txn = txns[txns["customer_id"] == customer_id]
    c_feat = features[features["customer_id"] == customer_id].iloc[0]

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Recent stats")
        st.metric("Spend last 30d", f"${c_feat['spend_30d_hist']:.2f}")
        st.metric("Spend last 90d", f"${c_feat['spend_90d_hist']:.2f}")
        st.metric("Orders last 90d", int(c_feat["orders_90d"]))
        st.metric("Recency (days)", int(c_feat["recency_days"]))

    with right:
        st.subheader("Spend over time")
        agg = (
            c_txn.groupby("order_date")
            .agg(daily_spend=("order_amount", "sum"), orders=("order_amount", "count"))
            .reset_index()
        )
        fig = px.bar(agg, x="order_date", y="daily_spend", hover_data=["orders"])
        fig.update_layout(margin=dict(t=10, b=40), height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent transactions")
    st.dataframe(
        c_txn.sort_values("order_date", ascending=False).head(50),
        use_container_width=True,
        height=320,
    )


def main():
    st.title("Predicting Short-Term Customer Spend (30-Day CLV)")
    st.caption(
        "Pre-loaded transactions and optional bundled model to explore customers and predict next 30-day spend."
    )

    with st.sidebar:
        st.header("Customer")
        st.caption("Select a customer to view stats and prediction.")

    txns = None
    local_model = None
    feature_cols_override: list[str] | None = None

    base_dir = pathlib.Path(__file__).parent

    # Preferred: use cleaned Kaggle dataset + trained RandomForest model
    cleaned_path = base_dir / "data" / "processed" / "cleaned_transactions.csv"
    rf_model_path = base_dir / "models" / "rf_customer_spend_model.pkl"

    raw_txn: pd.DataFrame | None = None
    rf_model = None
    has_real_pipeline = False

    if cleaned_path.exists() and rf_model_path.exists():
        try:
            raw_txn = pd.read_csv(cleaned_path)
            raw_txn["InvoiceDate"] = pd.to_datetime(raw_txn["InvoiceDate"])
            raw_txn["TransactionAmount"] = raw_txn["Quantity"] * raw_txn["UnitPrice"]
            raw_txn["CustomerID_str"] = raw_txn["CustomerID"].astype(str)

            rf_model = joblib.load(rf_model_path)
            has_real_pipeline = True

            # Normalize for the existing feature builder / UI
            txns = raw_txn.rename(
                columns={
                    "CustomerID_str": "customer_id",
                    "InvoiceDate": "order_date",
                    "TransactionAmount": "order_amount",
                }
            )[["customer_id", "order_date", "order_amount"]]
        except Exception as exc:
            st.error(
                f"Failed to load cleaned dataset / trained model, falling back to demo data: {exc}"
            )
            has_real_pipeline = False
            raw_txn = None
            rf_model = None

    # Fallback: original demo CSV / synthetic generator and optional local Python model
    if txns is None:
        data_path = base_dir / "transactions.csv"
        if data_path.exists():
            try:
                txns = load_transactions(data_path)
                st.success(f"Loaded {len(txns):,} transactions from {data_path.name}.")
            except Exception as exc:
                st.error(f"Bundled data could not be read: {exc}")
                txns = generate_demo_transactions()
                st.info("Fell back to demo data because bundled data failed.")
        else:
            txns = generate_demo_transactions()
            st.info(
                "Using generated demo data. Place a CSV named transactions.csv beside app.py to override."
            )

        model_path = base_dir / "model.py"
        if model_path.exists():
            try:
                local_model, feature_cols_override = load_local_model(model_path)
                st.success(f"Bundled Python model loaded from {model_path.name}.")
            except Exception as exc:
                st.error(f"Bundled model could not be loaded: {exc}")
                feature_cols_override = None
                local_model = None

    features, cutoff = build_customer_features(txns)
    if features.empty:
        st.warning("No features available after processing transactions.")
        st.stop()

    customer_id = st.sidebar.selectbox(
        "Customer", features["customer_id"].sort_values().unique()
    )
    st.sidebar.markdown("---")
    view = st.sidebar.radio(
        "Sections",
        ["Overview", "Transactions", "Detailed Analysis", "Feature Details"],
        index=0,
        help="Navigate between analysis views",
    )

    selected_row = features[features["customer_id"] == customer_id].iloc[0]

    # Choose prediction source: trained RF model (real pipeline) > optional local model > heuristic
    prediction: float
    per_customer_predictions = None

    if has_real_pipeline and rf_model is not None and raw_txn is not None:
        model_feats = build_model_features_for_customer(raw_txn, customer_id)
        if model_feats.empty:
            prediction = predict_spend(selected_row, model=None, feature_cols=None)
        else:
            try:
                prediction = float(rf_model.predict(model_feats[MODEL_FEATURE_COLS])[0])
            except Exception as exc:
                st.error(f"Model prediction failed, using heuristic instead: {exc}")
                prediction = predict_spend(selected_row, model=None, feature_cols=None)

        # Compute predictions for all customers (for charts)
        preds_map: dict[str, float] = {}
        for cid in features["customer_id"].unique():
            feats_all = build_model_features_for_customer(raw_txn, cid)
            if feats_all.empty:
                continue
            try:
                preds_map[cid] = float(
                    rf_model.predict(feats_all[MODEL_FEATURE_COLS])[0]
                )
            except Exception:
                continue
        if preds_map:
            per_customer_predictions = features.copy()
            per_customer_predictions["pred_30d_spend"] = per_customer_predictions[
                "customer_id"
            ].map(preds_map)
    else:
        feature_cols = feature_cols_override or (
            local_model.feature_names_in_.tolist()
            if local_model is not None and hasattr(local_model, "feature_names_in_")
            else None
        )
        prediction = predict_spend(
            selected_row, model=local_model, feature_cols=feature_cols
        )

        # For heuristic / local model path, compute predictions across all customers
        if feature_cols is not None and local_model is not None:
            try:
                per_customer_predictions = features.copy()
                X_all = per_customer_predictions[feature_cols]
                per_customer_predictions["pred_30d_spend"] = local_model.predict(X_all)
            except Exception:
                per_customer_predictions = None
        else:
            # Use heuristic for all rows
            per_customer_predictions = features.copy()
            per_customer_predictions["pred_30d_spend"] = per_customer_predictions.apply(
                lambda r: predict_spend(r, model=None, feature_cols=None), axis=1
            )

    if view == "Overview":
        st.markdown(
            f"**Prediction for next 30 days (cutoff {cutoff.date()}):** `${prediction:,.2f}`"
        )
        render_customer_snapshot(txns, features, customer_id)

        # Additional graphs focused on predicted 30-day spend
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.subheader("This customer: history vs prediction")
            cust_df = pd.DataFrame(
                {
                    "Metric": ["Last 30 days spend", "Predicted next 30 days"],
                    "Amount": [selected_row["spend_30d_hist"], prediction],
                }
            )
            fig_cust = px.bar(
                cust_df,
                x="Metric",
                y="Amount",
                text_auto=".2s",
                labels={"Amount": "Amount"},
            )
            fig_cust.update_layout(margin=dict(t=30, b=40), height=320)
            st.plotly_chart(fig_cust, use_container_width=True)

        with col_b:
            if per_customer_predictions is not None and not per_customer_predictions[
                "pred_30d_spend"
            ].isna().all():
                st.subheader("Distribution of predicted 30-day spend")
                fig_hist = px.histogram(
                    per_customer_predictions.dropna(
                        subset=["pred_30d_spend"]
                    ),
                    x="pred_30d_spend",
                    nbins=30,
                    labels={"pred_30d_spend": "Predicted 30-day spend"},
                )
                fig_hist.update_layout(margin=dict(t=30, b=40), height=320)
                st.plotly_chart(fig_hist, use_container_width=True)

    elif view == "Transactions":
        st.subheader("All transactions for selected customer")
        c_txn = txns[txns["customer_id"] == customer_id].sort_values(
            "order_date", ascending=False
        )
        st.dataframe(c_txn, use_container_width=True, height=500)

    elif view == "Detailed Analysis":
        st.subheader("Detailed analysis of predicted 30-day spend")

        if per_customer_predictions is None or per_customer_predictions[
            "pred_30d_spend"
        ].isna().all():
            st.warning(
                "Detailed analysis is unavailable because predictions for all customers could not be computed."
            )
        else:
            df_ana = per_customer_predictions.dropna(subset=["pred_30d_spend"]).copy()

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown("**Prediction vs 90d spend**")
                fig1 = px.scatter(
                    df_ana,
                    x="spend_90d_hist",
                    y="pred_30d_spend",
                    hover_data=["customer_id", "orders_90d"],
                    labels={
                        "spend_90d_hist": "Historical 90d spend",
                        "pred_30d_spend": "Predicted 30d spend",
                    },
                )
                fig1.update_layout(margin=dict(t=40, b=40), height=360)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                st.markdown("**Prediction vs order count (90d)**")
                fig2 = px.scatter(
                    df_ana,
                    x="orders_90d",
                    y="pred_30d_spend",
                    hover_data=["customer_id", "spend_90d_hist"],
                    labels={
                        "orders_90d": "Orders in last 90 days",
                        "pred_30d_spend": "Predicted 30d spend",
                    },
                )
                fig2.update_layout(margin=dict(t=40, b=40), height=360)
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2, gap="large")
            with col3:
                st.markdown("**Prediction vs recency**")
                fig3 = px.scatter(
                    df_ana,
                    x="recency_days",
                    y="pred_30d_spend",
                    hover_data=["customer_id"],
                    labels={
                        "recency_days": "Days since last purchase (90d window)",
                        "pred_30d_spend": "Predicted 30d spend",
                    },
                )
                fig3.update_layout(margin=dict(t=40, b=40), height=360)
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                st.markdown("**Average order value vs prediction**")
                fig4 = px.scatter(
                    df_ana,
                    x="avg_order_value",
                    y="pred_30d_spend",
                    hover_data=["customer_id"],
                    labels={
                        "avg_order_value": "Average order value (90d)",
                        "pred_30d_spend": "Predicted 30d spend",
                    },
                )
                fig4.update_layout(margin=dict(t=40, b=40), height=360)
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("---")
            st.subheader("Top 20 customers by predicted 30-day spend")
            top20 = (
                df_ana.sort_values("pred_30d_spend", ascending=False)
                .head(20)[
                    [
                        "customer_id",
                        "pred_30d_spend",
                        "spend_90d_hist",
                        "orders_90d",
                        "recency_days",
                    ]
                ]
            )
            st.dataframe(top20, use_container_width=True, height=400)

    elif view == "Feature Details":
        st.subheader("Feature values for this customer")
        st.write(selected_row.to_frame().rename(columns={0: "value"}))
        if feature_cols_override:
            st.caption("Using model feature order from FEATURE_COLS or model.feature_names_in_.")
        else:
            st.caption("Using heuristic prediction; add FEATURE_COLS in model.py to control order.")


if __name__ == "__main__":
    main()
