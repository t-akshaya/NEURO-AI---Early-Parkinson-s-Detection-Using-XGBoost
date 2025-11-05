# ===============================================================
# 🌟 NeuroAI - Early Parkinson's Detection using XGBoost 🌟
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
import csv
import os

# ---------------------------------------------------------------
# 🎨 Streamlit Page Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="NeuroAI - Parkinson's Detection", layout="wide")

# ---------------------------------------------------------------
# 🌌 Add Full Background Image
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://wallpapers.com/images/hd/professional-background-2b3p57557lfg0y5g.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.85);
    }

    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e3f2fd, #ede7f6);
    color: #1a237e;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background-color: #5e35b1;
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #7e57c2;
    transform: scale(1.05);
}
.title-header {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #311b92;
    margin-bottom: 5px;
    text-shadow: 1px 1px 3px #b39ddb;
}
.sub-header {
    text-align: center;
    font-size: 20px;
    color: #512da8;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🧭 Navigation Menu (Updated Order)
# ---------------------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Prediction", "Insights", "Analytics & Education"],
    icons=["house", "activity", "bar-chart", "database"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0!important",
            "background-color": "rgba(197, 202, 233, 0.8)",
            "width": "100%",
            "margin": "0",
            "border-radius": "0px",
            "box-shadow": "0 2px 10px rgba(0, 0, 0, 0.1)",
            "backdrop-filter": "blur(6px)",
        },
        "icon": {"color": "#1a237e", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "color": "#1a237e",
            "transition": "all 0.3s ease",
        },
        "nav-link:hover": {
            "background-color": "rgba(92, 107, 192, 0.3)",
            "color": "#0d47a1",
        },
        "nav-link-selected": {
            "background-color": "rgba(63, 81, 181, 0.9)",
            "color": "white",
            "font-weight": "bold",
        },
    }
)

# ---------------------------------------------------------------
# 📊 Load and Train Model Function
# ---------------------------------------------------------------
@st.cache_resource
def load_and_train(random_state=42):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    data = pd.read_csv(url)

    df_majority = data[data["status"] == 1]
    df_minority = data[data["status"] == 0]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=random_state)
    data_balanced = pd.concat([df_majority, df_minority_upsampled])

    X = data_balanced.drop(["name", "status"], axis=1)
    y = data_balanced["status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=random_state)

    model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                          subsample=0.9, colsample_bytree=0.9, random_state=random_state,
                          use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cv = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

    return model, scaler, X_test, y_test, y_pred, y_prob, cm, acc, rocauc, cv

model, scaler, X_test, y_test, y_pred, y_prob, cm, acc, rocauc, cv = load_and_train()

# ---------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------
feature_names = [
    'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer',
    'MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA',
    'NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE'
]

# ---------------------------------------------------------------
# 🏠 Home Tab
# ---------------------------------------------------------------
if selected == "Home":
    st.markdown("""<div style="text-align:center; padding:30px 10px 10px 10px;">
        <h1 style="color:#4B0082; font-size:48px; font-weight:bold; margin-bottom:10px;">
            🧠 NeuroAI - Early Parkinson's Detection using XGBoost
        </h1>
        <h4 style="color:#6A5ACD; font-weight:500; margin-top:0;">
            Empowering early diagnosis through AI & biomedical data analysis
        </h4>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="display:flex; justify-content:center; gap:30px; margin:20px 0;">
        <img src="https://www.statnews.com/wp-content/uploads/2021/12/AdobeStock_118570981-1600x900.jpeg" 
             style="width:350px; height:250px; object-fit:cover; border-radius:15px; box-shadow: 3px 3px 15px rgba(0,0,0,0.2);">
        <img src="https://nrtimes.co.uk/wp-content/uploads/2023/04/AdobeStock_483897781-1000x600.jpeg" 
             style="width:350px; height:250px; object-fit:cover; border-radius:15px; box-shadow: 3px 3px 15px rgba(0,0,0,0.2);">
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background: linear-gradient(145deg,#f0f0f5,#d9d9f0);
                padding:20px; border-radius:15px; margin:20px 0; 
                box-shadow: 3px 3px 15px rgba(0,0,0,0.2);">
        <h3 style="color:#4B0082; margin-bottom:10px;">🧩 Overview:</h3>
        <p style="font-size:16px; color:#333333; line-height:1.6; margin:0;">
            Parkinson’s disease is a progressive neurological disorder that affects movement.<br>
            Our model uses <b>XGBoost</b>, a cutting-edge ML algorithm, to detect early signs using voice and biomedical markers.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="background: linear-gradient(145deg,#d9f0f5,#b3d9f0);
                padding:20px; border-radius:15px; margin:20px 0; 
                box-shadow: 3px 3px 15px rgba(0,0,0,0.2);">
        <h3 style="color:#4B0082; margin-bottom:10px;">💡 Key Facts:</h3>
        <ul style="font-size:16px; color:#333333; line-height:1.8; margin:0; padding-left:20px;">
            <li>🧠 Caused by degeneration of dopamine neurons.</li>
            <li>👵 Common after age 60.</li>
            <li>⚕ Symptoms: tremors, stiffness, slow movement.</li>
            <li>🧬 Influenced by genetic and environmental factors.</li>
            <li>📊 AI models can identify early vocal biomarkers.</li>
        </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<p style="text-align:center; color:#6A5ACD; font-style:italic; margin-top:30px; margin-bottom:20px;">
        "Empowering healthcare with AI for early detection and better lives."
    </p>""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# 🔮 Prediction Tab
# ---------------------------------------------------------------
elif selected == "Prediction":
    st.markdown('<div class="title-header">🧬 Predict Parkinson’s from Patient Features</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a file or manually input patient data to predict Parkinson’s disease</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload Patient Data (CSV, TXT, Excel)", type=None)

    def load_file(file):
        try:
            ext = file.name.split('.')[-1].lower()
            if ext in ['csv','txt']:
                df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip')
            elif ext in ['xls','xlsx']:
                df = pd.read_excel(file)
            else:
                st.error("Unsupported file format.")
                return None
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    # Initialize session state for patient history
    if 'patient_history' not in st.session_state:
        st.session_state.patient_history = []

    history_file = "patient_history.csv"
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        st.session_state.patient_history = [history_df]
    else:
        history_df = pd.DataFrame()

    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            missing = [f for f in feature_names if f not in df.columns]
            if missing:
                st.error(f"Missing features: {missing}")
            else:
                X_scaled = scaler.transform(df[feature_names])
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:,1]
                df["Prediction"] = ["🧠 Parkinson’s" if p==1 else "💪 Healthy" for p in preds]
                df["Probability"] = probs.round(2)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Results", csv, "Predictions.csv", "text/csv")
    else:
        # --- Manual Input ---
        st.subheader("👤 Patient Details")
        cols = st.columns(3)
        name = cols[0].text_input("Patient Name")
        age = cols[1].number_input("Age", min_value=0, max_value=120, value=30)
        gender = cols[2].selectbox("Gender", ["Male", "Female", "Other"])

        st.subheader("🧬 Feature Values")
        input_cols = st.columns(3)
        inputs = []
        for i, f in enumerate(feature_names):
            with input_cols[i % 3]:
                inputs.append(st.number_input(f, value=0.0, step=0.001))

        if st.button("🔍 Predict", use_container_width=True):
            arr = np.array(inputs).reshape(1, -1)
            arr_scaled = scaler.transform(arr)
            pred = model.predict(arr_scaled)[0]
            prob = model.predict_proba(arr_scaled)[0][1]

            result_dict = {
                "Name": name if name else "N/A",
                "Age": age,
                "Gender": gender,
                "Prediction": "Parkinson’s" if pred==1 else "Healthy",
                "Probability": prob
            }

            # Append to session state and save permanently
            df_result = pd.DataFrame([result_dict])
            st.session_state.patient_history.append(df_result)

            if os.path.exists(history_file):
                df_result.to_csv(history_file, mode='a', header=False, index=False)
            else:
                df_result.to_csv(history_file, index=False)

            # --- Display result in colorful card ---
            if pred == 1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(120deg, #ff8a80, #e53935);
                    color: white;
                    padding: 25px;
                    border-radius: 20px;
                    text-align: center;
                    font-size: 22px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                ">
                    🧠 Parkinson’s Detected<br>
                    Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}<br>
                    Probability: {prob:.2f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(120deg, #90caf9, #42a5f5);
                    color: white;
                    padding: 25px;
                    border-radius: 20px;
                    text-align: center;
                    font-size: 22px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                ">
                    💪 Healthy<br>
                    Patient: {name if name else 'N/A'} | Age: {age} | Gender: {gender}<br>
                    Probability of Parkinson’s: {prob:.2f}
                </div>
                """, unsafe_allow_html=True)

    # --- Display Patient History as a normal table ---
    if st.session_state.patient_history:
        st.markdown("### 📝 Patient History")
        full_history = pd.concat(st.session_state.patient_history, ignore_index=True)
        st.dataframe(full_history)

        # Option to delete all history
        if st.button("🗑 Clear All History"):
            st.session_state.patient_history = []
            if os.path.exists(history_file):
                os.remove(history_file)
            st.success("All patient history cleared!")



# ---------------------------------------------------------------
# 📊 Insights Tab
# ---------------------------------------------------------------
elif selected == "Insights":
    st.markdown("""
    <h2 style="text-align:center; color:#4A148C; margin-top:30px;">
        📈 Model Performance Insights
    </h2>
""", unsafe_allow_html=True)



    # Performance Metrics
    st.markdown(f"""
<div style="display:flex; justify-content:center; margin-top:20px;">
    <table style="border-collapse: collapse; width: 50%; text-align:center; font-size:16px;">
        <tr style="background-color:#d1c4e9; color:#4A148C;">
            <th style="padding:10px; border:1px solid #b39ddb;">Metric</th>
            <th style="padding:10px; border:1px solid #b39ddb;">Value</th>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #b39ddb;">Accuracy</td>
            <td style="padding:10px; border:1px solid #b39ddb;">{acc*100:.2f}%</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #b39ddb;">ROC-AUC</td>
            <td style="padding:10px; border:1px solid #b39ddb;">{rocauc:.3f}</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #b39ddb;">Cross-Validation Mean</td>
            <td style="padding:10px; border:1px solid #b39ddb;">{cv.mean()*100:.2f}%</td>
        </tr>
        <tr>
            <td style="padding:10px; border:1px solid #b39ddb;">CV Std</td>
            <td style="padding:10px; border:1px solid #b39ddb;">{cv.std()*100:.2f}%</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)



    # Healthy Feature Overview Section
    st.markdown("""
    <div style="text-align:center; margin-top:30px;">
        <h2 style="color:#4A148C;">🧬 Healthy Individual Feature Overview</h2>
        <p style="color:#5e35b1; font-size:17px;">
            Typical biometric and voice feature values observed in non-affected individuals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    healthy_features = {
          'MDVP:Fo(Hz)': '120 - 220 Hz',
    'MDVP:Fhi(Hz)': '150 - 260 Hz',
    'MDVP:Flo(Hz)': '85 - 180 Hz',
    'MDVP:Jitter(%)': '0.001 - 0.006',
    'MDVP:Shimmer': '0.005 - 0.04',
    'HNR': '20 - 35 dB',
    'NHR': '0.005 - 0.03',
    'RPDE': '0.3 - 0.6',
    'DFA': '0.6 - 0.85',
    'PPE': '0.05 - 0.25'
    }

    features_list = list(healthy_features.items())
    for row in range(2):
        cols = st.columns(5)
        for i in range(5):
            key, value = features_list[row*5 + i]
            with cols[i]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #e1bee7, #d1c4e9);
                    border-radius: 15px;
                    padding: 15px 10px;
                    text-align: center;
                    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
                ">
                    <h4 style="color:#4A148C; font-size:15px; margin-bottom:8px;">{key}</h4>
                    <p style="color:#1A237E; font-size:16px; font-weight:bold; margin:0;">{value}</p>
                </div>
                """, unsafe_allow_html=True)
        if row == 0:
            st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

    # Prevention Tips
    st.markdown("""
    <div style="text-align:center; margin-top:40px;">
        <h2 style="color:#4A148C;">🌿 Prevention & Lifestyle Recommendations</h2>
        <p style="color:#5e35b1; font-size:17px;">While Parkinson’s can’t always be prevented, certain habits can lower risk and improve neural health.</p>
    </div>

    <div style="background: linear-gradient(145deg,#d7c6f3,#b39ddb);
                padding:20px; border-radius:15px; margin-top:25px;
                box-shadow: 3px 3px 15px rgba(0,0,0,0.2);">
        <ul style="font-size:16px; color:#212121; line-height:1.9;">
            <li>🥦 Maintain a balanced diet rich in antioxidants (fruits, vegetables, whole grains).</li>
            <li>🏃 Engage in regular physical activity — yoga, walking, and strength training.</li>
            <li>🧘 Manage stress through mindfulness or meditation to reduce neural strain.</li>
            <li>🧩 Keep your brain active — learn new skills, puzzles, reading, or socializing.</li>
            <li>💧 Ensure proper hydration and sleep to support brain detoxification processes.</li>
            <li>🚭 Avoid smoking and limit exposure to environmental toxins.</li>
            <li>🩺 Regular neurological check-ups if any early symptoms appear.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# 📊 Dataset Tab → Replaced with Analytics / Educational Tab
# ---------------------------------------------------------------
elif selected == "Analytics & Education":  # Keep the menu selection same
    st.markdown("""
    <h2 style="text-align:center; color:#4A148C;">
        🧠 Patient Analytics & Educational Insights
    </h2>
""", unsafe_allow_html=True)


    # -------------------------
    # ⿡ Interactive Patient Dashboard
    # -------------------------
    st.markdown("### 📊 Interactive Patient Dashboard")

    import plotly.express as px

    # Load dataset if not already loaded
    @st.cache_data
    def load_dataset():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
        df = pd.read_csv(url)
        # Add dummy age if not present
        if 'age' not in df.columns:
            np.random.seed(42)
            df['age'] = np.random.randint(40, 80, size=len(df))
        return df

    data = load_dataset()

    # Feature selection for comparison
    feature_options = ['age', 'MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)',
                       'MDVP:Jitter(%)','MDVP:Shimmer','HNR','RPDE','DFA','PPE']
    feature_selected = st.selectbox("Select Feature to Compare (Affected vs Healthy):", feature_options)

    # Split data
    healthy_data = data[data['status']==0]
    affected_data = data[data['status']==1]

    # Plot histogram + boxplot
    if feature_selected in data.columns:
        fig = px.histogram(
            data,
            x=feature_selected,
            color=data['status'].map({0:'Healthy', 1:'Parkinson’s'}),
            labels={'color':'Patient Status'},
            barmode='overlay',
            marginal='box',
            color_discrete_map={'Healthy':'#66bb6a', 'Parkinson’s':'#ef5350'},
            hover_data=data.columns
        )
        fig.update_layout(
            title=f"Distribution of {feature_selected} by Patient Status",
            xaxis_title=feature_selected,
            yaxis_title="Count",
            legend_title="Status",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -------------------------
    # ⿢ Summary Metrics as Cards
    # -------------------------
    st.markdown("### 📊 Summary Metrics")
    metrics = {
        "Total Patients": data.shape[0],
        "Healthy Patients": (data['status']==0).sum(),
        "Affected Patients": (data['status']==1).sum(),
        "Average Age (Affected)": int(affected_data['age'].mean())
    }

    cols = st.columns(4, gap="large")
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #b3e5fc, #81d4fa);
                border-radius: 15px;
                padding: 25px 10px;
                text-align: center;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
                height: 130px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <h4 style="color:#01579b; font-size:16px; margin-bottom:8px;">{key}</h4>
                <p style="color:#0d47a1; font-size:22px; font-weight:bold; margin:0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------
    # ⿣ Educational / Informative Section (Insights-style)
    # -------------------------
    st.markdown("### 📘 Feature Explanations")
    feature_info = {
        'MDVP:Fo(Hz)': 'Average vocal fundamental frequency (pitch).',
        'MDVP:Fhi(Hz)': 'Highest pitch detected.',
        'MDVP:Flo(Hz)': 'Lowest pitch detected.',
        'MDVP:Jitter(%)': 'Frequency variation in voice.',
        'MDVP:Shimmer': 'Amplitude variation in voice.',
        'HNR': 'Harmonics-to-noise ratio; higher means clearer voice.',
        'RPDE': 'Recurrence period density entropy; measures signal predictability.',
        'DFA': 'Detrended fluctuation analysis; detects long-term correlations.',
        'PPE': 'Pitch period entropy; variability in pitch periods.'
    }

    features_list = list(feature_info.items())

    # Split into 2 rows (first 5 cards, remaining 4)
    row1 = features_list[:5]
    row2 = features_list[5:]

    def display_insights_row(row):
        cols = st.columns(len(row), gap="large")
        for i, (key, desc) in enumerate(row):
            with cols[i]:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #e1bee7, #d1c4e9);
                    border-radius: 15px;
                    padding: 20px 10px;
                    text-align: center;
                    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
                    height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <h4 style="color:#4A148C; font-size:16px; margin-bottom:8px;">{key}</h4>
                    <p style="color:#1A237E; font-size:14px; font-weight:bold; margin:0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    display_insights_row(row1)
    st.markdown("<br>", unsafe_allow_html=True)  # spacing
    display_insights_row(row2)

    st.markdown("---")
