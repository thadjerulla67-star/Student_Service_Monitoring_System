import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from modules.data_processing import load_and_prepare, quality_summary, eligible_completed
from modules.exploratory_analysis import make_eda_figures
from modules.prediction import load_model, predict_one
from modules.dashboard import dashboard_data, queue_chart

st.set_page_config(page_title="Student Service Request Monitoring System", page_icon="🎓", layout="wide")
BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "student_service_requests.csv"

@st.cache_data
def read_default_data():
    return load_and_prepare(DEFAULT_DATA)

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

st.sidebar.title("🎓 Student Services")
st.sidebar.caption("SLA monitoring and decision support")
page = st.sidebar.selectbox("Navigate", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.divider()
st.sidebar.caption("Model: Decision Tree Classifier | Random state: 42")
st.sidebar.caption("Human review is required for every flagged case.")

if "active_df" not in st.session_state:
    st.session_state.active_df = read_default_data()

df = st.session_state.active_df

if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.subheader("A Streamlit decision-support application")
    st.write("Monitor student service requests, explore historical SLA performance, and generate model-based predictions to support staff review.")
    a,b,c = st.columns(3)
    completed = eligible_completed(df)
    with a: metric_card("Dataset records", f"{len(df):,}")
    with b: metric_card("Completed records", f"{len(completed):,}")
    with c: metric_card("Pending requests", f"{(df.Status == 'Pending').sum():,}")
    st.info("Predictions are advisory only. Authorized student-services staff must verify the request and record the final decision.")
    st.markdown("### Application modules")
    st.markdown("- **Dataset:** upload and validate a CSV file.\n- **Data Analysis:** inspect statistics and interactive charts.\n- **Prediction:** enter request details and obtain a model prediction.\n- **Dashboard:** review workload and completed requests that exceeded their SLA.\n- **System Design:** view actors, requirements, use case diagram, and workflow.")
    st.markdown("### Important model scope")
    st.warning("The midterm model was trained on completed requests and defines SLA_Breached using actual Processing_Days compared with SLA_Days. It is a reproduction of the midterm classification task, not a validated early-warning model for pending requests.")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    upload = st.file_uploader("Upload student service requests CSV", type=["csv"])
    if upload is not None:
        try:
            uploaded_df = load_and_prepare(upload)
            st.session_state.active_df = uploaded_df
            df = uploaded_df
            st.success("CSV uploaded and required columns validated.")
        except Exception as exc:
            st.error(f"Could not use this file: {exc}")
    st.caption("Showing the currently active dataset. The supplied midterm CSV is loaded by default.")
    summary = quality_summary(df)
    x1,x2,x3,x4 = st.columns(4)
    x1.metric("Rows", summary["rows"]); x2.metric("Columns", summary["columns"])
    x3.metric("Missing cells", summary["missing_cells"]); x4.metric("Duplicate Request IDs", summary["duplicate_request_ids"])
    st.markdown("### Data preview")
    st.dataframe(df.head(30), use_container_width=True)
    st.markdown("### Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing Values"), use_container_width=True)
    st.markdown("### Data types")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)
    st.download_button("Download active dataset", df.to_csv(index=False).encode("utf-8"), "student_service_requests_active.csv", "text/csv")

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    completed = eligible_completed(df)
    st.markdown("### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)
    st.markdown("### SLA class distribution")
    if len(completed):
        counts = completed["SLA_Breached"].value_counts().rename(index={0:"Within SLA (0)",1:"SLA Breached (1)"})
        st.dataframe(counts.rename("Records"), use_container_width=True)
        st.write(f"Eligible completed records: **{len(completed)}** of **{len(df)}** total records.")
    else:
        st.warning("No completed records with valid processing and SLA days are available.")
    st.markdown("### Interactive charts")
    f1,f2,f3 = make_eda_figures(df)
    st.plotly_chart(f1, use_container_width=True)
    st.plotly_chart(f2, use_container_width=True)
    st.plotly_chart(f3, use_container_width=True)

elif page == "Prediction":
    st.title("SLA Breach Prediction")
    st.warning("This page reuses the midterm model. It predicts the midterm target for a manually entered feature profile. Because the training rows are completed requests, do not treat the result as a confirmed forecast for a pending request.")
    model = load_model()
    service_options = sorted(df.Service_Type.dropna().astype(str).unique().tolist())
    requester_options = sorted(df.Requester_Type.dropna().astype(str).unique().tolist())
    channel_options = sorted(df.Requested_Channel.dropna().astype(str).unique().tolist())
    priority_options = sorted(df.Priority.dropna().astype(str).unique().tolist())
    office_options = sorted(df.Assigned_Office.dropna().astype(str).unique().tolist())
    weekday_options = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    with st.form("prediction_form"):
        left,right = st.columns(2)
        with left:
            service = st.selectbox("Service Type", service_options)
            requester = st.selectbox("Requester Type", requester_options)
            channel = st.selectbox("Requested Channel", channel_options)
            priority = st.selectbox("Priority", priority_options)
            office = st.selectbox("Assigned Office", office_options)
        with right:
            queue = st.number_input("Queue Length", min_value=0, max_value=10000, value=10, step=1)
            docs = st.selectbox("Required Documents Complete", sorted(df.Required_Documents_Complete.dropna().astype(str).unique()))
            sla = st.number_input("SLA Days", min_value=1, max_value=365, value=5, step=1)
            weekday = st.selectbox("Submission Weekday", weekday_options)
        submitted = st.form_submit_button("Predict SLA Outcome", type="primary")
    if submitted:
        qcat = "Low" if queue <= 5 else ("Medium" if queue <= 15 else "High")
        values = {"Service_Type":service,"Requester_Type":requester,"Requested_Channel":channel,"Priority":priority,
                  "Assigned_Office":office,"Queue_Length":int(queue),"Required_Documents_Complete":docs,
                  "SLA_Days":int(sla),"Submission_Weekday":weekday,"Queue_Category":qcat}
        label, prob = predict_one(model, values)
        if label == 1: st.error("Model prediction: SLA Breached (1)")
        else: st.success("Model prediction: Within SLA (0)")
        if prob is not None: st.metric("Model-estimated breach probability", f"{prob*100:.1f}%")
        st.caption("This is a model output, not an automatic decision. Verify request details and document the authorized staff review.")
    st.markdown("### Midterm model evaluation")
    completed = eligible_completed(df)
    if len(completed):
        X_eval = completed[["Service_Type","Requester_Type","Requested_Channel","Priority","Assigned_Office","Queue_Length","Required_Documents_Complete","SLA_Days","Submission_Weekday","Queue_Category"]]
        y_eval = completed["SLA_Breached"].astype(int)
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(X_eval, y_eval, test_size=.20, random_state=42, stratify=y_eval)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred,labels=[0,1])
        a,b,c,d = st.columns(4)
        a.metric("Accuracy",f"{accuracy_score(y_test,y_pred):.3f}")
        b.metric("Precision",f"{precision_score(y_test,y_pred,zero_division=0):.3f}")
        c.metric("Recall",f"{recall_score(y_test,y_pred,zero_division=0):.3f}")
        d.metric("F1-score",f"{f1_score(y_test,y_pred,zero_division=0):.3f}")
        st.write("Confusion matrix (actual rows; predicted columns; order: 0, 1)")
        st.dataframe(pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]))
        st.caption("Metrics are recalculated from the currently active dataset using the midterm 80/20 split settings. Uploading a different dataset can change these values.")
    st.markdown("### Five sample predictions")
    samples = df.dropna(subset=["Queue_Length","SLA_Days","Submission_Weekday","Queue_Category"]).head(5)
    if len(samples):
        rows=[]
        for _, r in samples.iterrows():
            vals={k:r[k] for k in ["Service_Type","Requester_Type","Requested_Channel","Priority","Assigned_Office","Queue_Length","Required_Documents_Complete","SLA_Days","Submission_Weekday","Queue_Category"]}
            pred, pr = predict_one(model, vals)
            rows.append({"Request_ID":r["Request_ID"],"Predicted Outcome":"SLA Breached" if pred else "Within SLA","Breach Probability":round(pr,3) if pr is not None else None})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    total,pending,breached,rate,flagged = dashboard_data(df)
    completed_count = len(eligible_completed(df))
    a,b,c,d = st.columns(4)
    a.metric("Total requests",f"{total:,}"); b.metric("Pending requests",f"{pending:,}")
    c.metric("Completed SLA breaches",f"{breached:,}"); d.metric("Completed-record breach rate",f"{rate:.1f}%")
    st.plotly_chart(queue_chart(df), use_container_width=True)
    st.markdown("### Staff review queue — completed records that exceeded SLA")
    st.caption("This is a retrospective queue based on actual processing time, not a live prediction queue.")
    if len(flagged):
        st.dataframe(flagged[["Request_ID","Service_Type","Assigned_Office","Queue_Length","Processing_Days","SLA_Days","SLA_Breached"]], use_container_width=True)
        st.markdown("**Suggested human follow-up:** verify the request history, identify the cause of delay, check whether documents or office workload contributed, contact the appropriate office, and record the staff action.")
    else:
        st.success("No completed SLA-breached records in the active dataset.")
    st.markdown("### Interpretation")
    st.write(f"There are **{pending} pending** requests requiring normal operational monitoring. The completed-record breach rate is **{rate:.1f}%** ({breached} of {completed_count} eligible completed requests).")

elif page == "System Design":
    st.title("System Design and SAD Integration")
    st.markdown("### Actors")
    st.write("- **Administrator:** views service volume, SLA performance, delay risk, flagged requests, and queue status.\n- **Student Services Staff:** reviews individual requests and verifies supporting information before taking action.")
    st.markdown("### Functional requirements")
    for item in [
        "FR1 — The system shall allow an authorized user to upload a CSV file and validate required columns.",
        "FR2 — The system shall display dataset previews, missing values, duplicate IDs, and descriptive statistics.",
        "FR3 — The system shall display interactive charts for service volume and SLA outcomes.",
        "FR4 — The system shall load the saved midterm Decision Tree pipeline and accept feature inputs for prediction.",
        "FR5 — The system shall display workload indicators and records requiring staff review."
    ]: st.write("• "+item)
    st.markdown("### Non-functional requirements")
    for item in [
        "NFR1 — Usability: navigation and labels shall be understandable to student-services users.",
        "NFR2 — Reliability: invalid files and missing required columns shall produce clear error messages.",
        "NFR3 — Security: use synthetic or authorized data; do not publish passwords or sensitive student records."
    ]: st.write("• "+item)
    st.markdown("### Use case diagram")
    img = BASE / "assets" / "use_case_diagram.png"
    st.image(str(img), caption="Administrator use case diagram supplied with the assignment", use_container_width=True)
    st.markdown("### Analytical decision workflow")
    st.code("CSV Upload → Schema Validation → Data Preview / EDA → Saved Model Inference → Prediction + Metrics → Staff Verification → Record Final Human Decision", language="text")
    st.markdown("### Human decision and limitations")
    st.write("The Administrator can inspect system-level results. Student Services Staff should verify the request details and supporting records. The model does not approve, reject, or deny student services automatically.")
    st.warning("The original model uses completed cases to define the target from actual Processing_Days. For genuine early-warning deployment, retraining and validation with only information available at submission time would be required.")
