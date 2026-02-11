import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import r2_score
from streamlit_option_menu import option_menu

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fleet AI Intelligence",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

.kpi-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.kpi-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #3b82f6;
}

.dashboard-title {
    font-size: 42px;
    font-weight: bold;
    color: #60a5fa;
}

.dashboard-sub {
    font-size: 18px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/auto-mpg.csv")

df.replace("?", pd.NA, inplace=True)
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df.dropna(inplace=True)

# ---------------- MODEL TRAIN / LOAD ----------------
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

if not os.path.exists("models/fuel_model.pkl"):

    os.makedirs("models", exist_ok=True)

    X = df[['cylinders','displacement','horsepower','weight','acceleration']]
    y = df['mpg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    cluster_model = KMeans(n_clusters=3, random_state=42)
    cluster_model.fit(X)

    joblib.dump(model, "models/fuel_model.pkl")
    joblib.dump(cluster_model, "models/cluster_model.pkl")

else:
    model = joblib.load("models/fuel_model.pkl")
    cluster_model = joblib.load("models/cluster_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="dashboard-title">🚚 Fleet AI Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-sub">Predictive Fuel Analytics & Fleet Optimization Platform</div>', unsafe_allow_html=True)

st.divider()

# ---------------- NAVIGATION MENU ----------------
selected = option_menu(
    menu_title=None,
    options=["Overview","Prediction","Segmentation","Trend"],
    icons=["bar-chart","cpu","diagram-3","graph-up"],
    orientation="horizontal"
)

# ---------------- KPI SECTION ----------------
X = df[['cylinders','displacement','horsepower','weight','acceleration']]
y = df['mpg']
accuracy = r2_score(y, model.predict(X))

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f'<div class="kpi-card"><h3>Average MPG</h3><h2>{round(df["mpg"].mean(),2)}</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="kpi-card"><h3>Best Efficiency</h3><h2>{df["mpg"].max()}</h2></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="kpi-card"><h3>Worst Efficiency</h3><h2>{df["mpg"].min()}</h2></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="kpi-card"><h3>Model Accuracy</h3><h2>{round(accuracy,2)}</h2></div>', unsafe_allow_html=True)

st.divider()

# ---------------- OVERVIEW ----------------
if selected == "Overview":

    st.subheader("📊 Fleet Performance Overview")

    col1, col2 = st.columns(2)

    fig1 = px.histogram(df, x="mpg", nbins=30, template="plotly_dark")
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(df, x="weight", y="mpg", color="cylinders", template="plotly_dark")
    col2.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.head(20), use_container_width=True)

# ---------------- PREDICTION ----------------
elif selected == "Prediction":

    st.subheader("🤖 Fuel Efficiency Prediction")

    col1, col2 = st.columns(2)

    cyl = col1.slider("Cylinders", 3, 8, 4)
    disp = col1.slider("Displacement", 60, 500, 150)
    hp = col1.slider("Horsepower", 40, 250, 100)

    wt = col2.slider("Weight", 1500, 5000, 2500)
    acc = col2.slider("Acceleration", 8, 25, 15)

    if st.button("Predict Fuel Efficiency"):

        pred = model.predict([[cyl, disp, hp, wt, acc]])[0]

        if pred > 30:
            category = "🟢 High Efficiency"
        elif pred > 20:
            category = "🟡 Moderate Efficiency"
        else:
            category = "🔴 Low Efficiency"

        st.success(f"Predicted MPG: {round(pred,2)}")
        st.info(f"Efficiency Category: {category}")

# ---------------- SEGMENTATION ----------------
elif selected == "Segmentation":

    st.subheader("🔍 Fleet Segmentation")

    X_cluster = df[['cylinders','displacement','horsepower','weight','acceleration']]
    df['Cluster'] = cluster_model.predict(X_cluster)

    labels = {
        0:"Efficient Fleet",
        1:"Average Fleet",
        2:"High Consumption Fleet"
    }

    df["Cluster Label"] = df["Cluster"].map(labels)

    fig = px.scatter(df, x="weight", y="mpg", color="Cluster Label", template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

# ---------------- TREND ----------------
elif selected == "Trend":

    st.subheader("📈 Efficiency Trend")

    trend = df.groupby("model year")["mpg"].mean().reset_index()

    fig = px.line(trend, x="model year", y="mpg", markers=True, template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)
