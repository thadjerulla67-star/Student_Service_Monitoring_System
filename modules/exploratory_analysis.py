import pandas as pd
import plotly.express as px
from .data_processing import eligible_completed

def make_eda_figures(df):
    completed = eligible_completed(df)
    service = px.bar(df["Service_Type"].value_counts().rename_axis("Service Type").reset_index(name="Requests"),
                     x="Service Type", y="Requests", title="Requests by Service Type")
    target = completed["SLA_Breached"].value_counts().rename(index={0:"Within SLA",1:"SLA Breached"}).rename_axis("Outcome").reset_index(name="Records")
    distribution = px.pie(target, names="Outcome", values="Records", title="SLA Outcome Distribution (Completed Records)", hole=.35)
    queue = completed.groupby("Queue_Category", observed=False)["SLA_Breached"].mean().mul(100).reset_index(name="Breach Rate (%)")
    queue_fig = px.bar(queue, x="Queue_Category", y="Breach Rate (%)", title="SLA Breach Rate by Queue Category")
    return service, distribution, queue_fig
