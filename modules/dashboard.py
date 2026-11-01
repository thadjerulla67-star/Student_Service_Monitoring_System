import pandas as pd
import plotly.express as px
from .data_processing import eligible_completed

def dashboard_data(df):
    completed = eligible_completed(df)
    total = len(df)
    pending = int((df["Status"] == "Pending").sum())
    breached = int(completed["SLA_Breached"].sum()) if len(completed) else 0
    rate = (breached / len(completed) * 100) if len(completed) else 0
    flagged = completed[completed["SLA_Breached"] == 1].sort_values("Queue_Length", ascending=False)
    return total, pending, breached, rate, flagged

def queue_chart(df):
    counts = df["Status"].value_counts().rename_axis("Status").reset_index(name="Requests")
    return px.bar(counts, x="Status", y="Requests", title="Current Request Status")
