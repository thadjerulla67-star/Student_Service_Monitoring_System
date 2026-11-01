# Student Service Request Monitoring System

Streamlit application for the Section B laboratory exercise. It reuses the supplied midterm Decision Tree classification workflow and the supplied student service request CSV.

## Run locally
1. Install Python 3.11 or 3.12.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional but recommended).
4. Install packages: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`

## GitHub / Streamlit Community Cloud
Upload the **contents** of this project folder to the root of a GitHub repository. In Streamlit Community Cloud, select that repository, branch `main`, and main file path `app.py`.

## Pages
Home, Dataset, Data Analysis, Prediction, Dashboard, and System Design.

## Midterm model definition
- Eligible training records: `Status == Completed` and non-null `Processing_Days` and `SLA_Days`.
- Target: `SLA_Breached = 1` when `Processing_Days > SLA_Days`, otherwise `0`.
- Features: Service_Type, Requester_Type, Requested_Channel, Priority, Assigned_Office, Queue_Length, Required_Documents_Complete, SLA_Days, Submission_Weekday, Queue_Category.
- Train/test split: 80/20, `random_state=42`, stratified.
- Classifier: `DecisionTreeClassifier(random_state=42)` in a preprocessing pipeline.

## Important limitation
Because the target is calculated using actual processing duration and the model was trained on completed requests, this reproduces the midterm classification task. It should not be represented as a validated prediction of delay for a still-pending request. Predictions are decision support only; staff must review and make final decisions.

Use synthetic or properly authorized records for public deployment. Do not upload private student records or credentials.\n