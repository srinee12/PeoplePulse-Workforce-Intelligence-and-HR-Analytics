import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

st.set_page_config(page_title="PeoplePulse | HR Intelligence", page_icon="✦", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0b1220 0%,#101b2d 55%,#111827 100%); color:#edf2f7}
[data-testid="stMetric"] {background:#172338;padding:18px;border:1px solid #26364d;border-radius:14px}
section[data-testid="stSidebar"] {background:#101a2b}
div[data-testid="stTabs"] button {font-weight:650}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    d = "data/"
    e = pd.read_csv(d+"employees.csv", parse_dates=["hire_date"])
    t = pd.read_csv(d+"training_records.csv", parse_dates=["enrollment_date"])
    p = pd.read_csv(d+"performance_reviews.csv")
    a = pd.read_csv(d+"monthly_attendance.csv")
    r = pd.read_csv(d+"recruitment_pipeline.csv", parse_dates=["application_date"])
    return e,t,p,a,r

try:
    emp, training, perf, attendance, recruit = load_data()
except Exception as ex:
    st.error(f"Could not load project data: {ex}. Run this app from the project root.")
    st.stop()

st.title("✦ PeoplePulse HR Intelligence")
st.caption("Enterprise workforce analytics • Synthetic portfolio dataset • 2021–2025")
with st.sidebar:
    st.header("Control Center")
    departments = ["All"] + sorted(emp.department.unique().tolist())
    dept = st.selectbox("Department", departments)
    locations = ["All"] + sorted(emp.location.unique().tolist())
    loc = st.selectbox("Location", locations)
    levels = ["All"] + sorted(emp.job_level.unique().tolist())
    level = st.selectbox("Job level", levels)
    status = st.selectbox("Employment status", ["All","Active","Exited"])
    st.divider()
    st.caption("Filters apply across employee-based views.")

f = emp.copy()
if dept!="All": f=f[f.department==dept]
if loc!="All": f=f[f.location==loc]
if level!="All": f=f[f.job_level==level]
if status!="All": f=f[f.employment_status==status]
ids = set(f.employee_id)
ft = training[training.employee_id.isin(ids)]
fp = perf[perf.employee_id.isin(ids)]
fa = attendance[attendance.employee_id.isin(ids)]

tabs=st.tabs(["Workforce Overview","Attrition Intelligence","Performance & Learning","Recruitment Funnel","Employee 360°"])
with tabs[0]:
    st.subheader("Workforce Overview")
    c=st.columns(4)
    c[0].metric("Employees",f"{len(f):,}")
    c[1].metric("Attrition rate",f"{f.attrition_flag.mean()*100:.1f}%" if len(f) else "—")
    c[2].metric("Average tenure",f"{f.tenure_years.mean():.1f} yrs" if len(f) else "—")
    c[3].metric("Average salary",f"₹{f.annual_salary_inr.mean()/100000:.1f}L" if len(f) else "—")
    c1,c2=st.columns(2)
    with c1:
        by= f.groupby("department",as_index=False).agg(headcount=("employee_id","count"))
        st.plotly_chart(px.bar(by,x="department",y="headcount",color="headcount",title="Headcount by department",template="plotly_dark"),use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(f,names="employment_status",hole=.58,title="Employment mix",template="plotly_dark"),use_container_width=True)
    c3,c4=st.columns(2)
    with c3:
        st.plotly_chart(px.histogram(f,x="age",color="gender",title="Age distribution",template="plotly_dark"),use_container_width=True)
    with c4:
        st.plotly_chart(px.box(f,x="job_level",y="annual_salary_inr",color="job_level",title="Salary distribution by level",template="plotly_dark"),use_container_width=True)
    st.download_button("Download filtered workforce CSV",f.to_csv(index=False).encode(), "filtered_workforce.csv","text/csv")

with tabs[1]:
    st.subheader("Attrition Intelligence")
    st.info("Model is trained on synthetic historical labels. Scores are exploratory, not decisions about real employees.")
    features=["department","job_level","location","age","tenure_years","annual_salary_inr","satisfaction_score","engagement_score","overtime"]
    X=emp[features]; y=emp.attrition_flag
    cat=["department","job_level","location","overtime"]
    num=[x for x in features if x not in cat]
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num)])
    model=Pipeline([("prep",pre),("clf",LogisticRegression(max_iter=700,class_weight="balanced"))])
    model.fit(X,y)
    pred=model.predict_proba(f[features])[:,1] if len(f) else np.array([])
    if len(pred):
        risk=f[["employee_id","department","job_level","location","satisfaction_score","engagement_score","overtime"]].copy()
        risk["risk_score"]=pred
        risk["risk_band"]=pd.cut(pred,[-.01,.33,.66,1.01],labels=["Low","Medium","High"])
        a1,a2,a3=st.columns(3)
        a1.metric("High-risk cohort",f"{(risk.risk_band=='High').sum():,}")
        a2.metric("Average predicted risk",f"{pred.mean()*100:.1f}%")
        a3.metric("Model ROC-AUC (in-sample)",f"{roc_auc_score(y,model.predict_proba(X)[:,1]):.3f}")
        st.plotly_chart(px.histogram(risk,x="risk_score",color="risk_band",nbins=24,title="Predicted attrition risk distribution",template="plotly_dark"),use_container_width=True)
        st.dataframe(risk.sort_values("risk_score",ascending=False).head(100),use_container_width=True)
        st.download_button("Download risk-scored cohort",risk.to_csv(index=False).encode(),"attrition_risk.csv","text/csv")
    else: st.warning("No employees match the selected filters.")

with tabs[2]:
    st.subheader("Performance & Learning Analytics")
    p1,p2=st.columns(2)
    with p1:
        avg=fp.groupby("review_year",as_index=False).performance_rating.mean()
        st.plotly_chart(px.line(avg,x="review_year",y="performance_rating",markers=True,title="Average performance rating trend",template="plotly_dark"),use_container_width=True)
    with p2:
        cat=ft.groupby("course_category",as_index=False).agg(enrollments=("training_id","count"),score=("assessment_score","mean"))
        st.plotly_chart(px.bar(cat,x="course_category",y="enrollments",color="score",title="Learning participation & average score",template="plotly_dark"),use_container_width=True)
    completed=(ft.completion_status=="Completed").mean()*100 if len(ft) else 0
    k=st.columns(3); k[0].metric("Training enrollments",f"{len(ft):,}"); k[1].metric("Completion rate",f"{completed:.1f}%"); k[2].metric("Avg review rating",f"{fp.performance_rating.mean():.2f}" if len(fp) else "—")
    st.plotly_chart(px.scatter(fp,x="goal_completion_pct",y="performance_rating",color="promotion_recommended",opacity=.55,title="Goal completion vs performance",template="plotly_dark"),use_container_width=True)

with tabs[3]:
    st.subheader("Recruitment Funnel Analytics")
    stage_order=["Applied","Screening","Interview","Assessment","Offer","Hired","Rejected"]
    rc=recruit.copy()
    if dept!="All": rc=rc[rc.role_department==dept]
    if loc!="All": rc=rc[rc.location==loc]
    q=rc.current_stage.value_counts().reindex(stage_order,fill_value=0).rename_axis("stage").reset_index(name="applications")
    q1,q2,q3=st.columns(3)
    q1.metric("Applications",f"{len(rc):,}")
    q2.metric("Hires",f"{(rc.current_stage=='Hired').sum():,}")
    q3.metric("Avg time to hire",f"{rc.time_to_hire_days.mean():.1f} days" if rc.time_to_hire_days.notna().any() else "—")
    st.plotly_chart(px.funnel(q,x="applications",y="stage",title="Recruitment stage volume",template="plotly_dark"),use_container_width=True)
    src=rc.groupby("candidate_source",as_index=False).agg(applications=("application_id","count"),hires=("current_stage",lambda x:(x=="Hired").sum()))
    src["hire_conversion_pct"]=src.hires/src.applications*100
    st.plotly_chart(px.bar(src,x="candidate_source",y="hire_conversion_pct",color="applications",title="Hire conversion by source (%)",template="plotly_dark"),use_container_width=True)
    st.dataframe(src.sort_values("hire_conversion_pct",ascending=False),use_container_width=True)

with tabs[4]:
    st.subheader("Employee 360° Explorer")
    chosen=st.selectbox("Select employee",f.employee_id.tolist() if len(f) else [])
    if chosen:
        row=emp[emp.employee_id==chosen].iloc[0]
        m=st.columns(4)
        m[0].metric("Department",row.department); m[1].metric("Tenure",f"{row.tenure_years} yrs")
        m[2].metric("Engagement",f"{row.engagement_score}/100"); m[3].metric("Satisfaction",f"{row.satisfaction_score}/5")
        st.write("**Profile**")
        st.dataframe(row.to_frame("Value"),use_container_width=True)
        st.write("**Performance history**")
        st.dataframe(perf[perf.employee_id==chosen].sort_values("review_year"),use_container_width=True)
        st.write("**Training history**")
        st.dataframe(training[training.employee_id==chosen].sort_values("enrollment_date",ascending=False),use_container_width=True)
        st.write("**Attendance history**")
        at=attendance[attendance.employee_id==chosen]
        st.plotly_chart(px.line(at,x="month",y=["days_absent","late_arrivals"],title="Monthly attendance signals",template="plotly_dark"),use_container_width=True)
    else: st.warning("No employees match the selected filters.")

st.divider()
st.caption("PeoplePulse HR Intelligence • Portfolio demonstration • All records are synthetic and generated for analytics practice.")
