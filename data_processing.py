import pandas as pd

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened", "Satisfaction_Score"
]

def load_and_prepare(uploaded_file):
    df = pd.read_csv(uploaded_file)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    df["Submission_Weekday"] = df["Submission_Date"].dt.day_name()
    df["Queue_Category"] = pd.cut(
        pd.to_numeric(df["Queue_Length"], errors="coerce"),
        bins=[-1, 5, 15, float("inf")], labels=["Low", "Medium", "High"]
    )
    df["SLA_Breached"] = (
        pd.to_numeric(df["Processing_Days"], errors="coerce") >
        pd.to_numeric(df["SLA_Days"], errors="coerce")
    ).astype("Int64")
    return df

def quality_summary(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_request_ids": int(df["Request_ID"].duplicated().sum()),
    }

def eligible_completed(df):
    return df[
        (df["Status"] == "Completed") &
        df["Processing_Days"].notna() &
        df["SLA_Days"].notna()
    ].copy()
