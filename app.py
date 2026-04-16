"""
=============================================================================
PLACEMENT READINESS & JOB ROLE PREDICTOR — STREAMLIT WEBAPP
=============================================================================
Loads trained model artifacts from models/ folder.
Run: streamlit run app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Readiness Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] label { color: #a0c4ff !important; font-size:0.82rem !important; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #a0c4ff !important; }

    .welcome-banner {
        background: linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);
        border-radius:16px; padding:26px 36px; color:white; margin-bottom:20px;
    }
    .welcome-banner h1 { color:white; font-size:1.75rem; margin:0 0 5px; }
    .welcome-banner p  { color:#bfdbfe; margin:0; font-size:0.95rem; }

    .metric-box {
        background:white; border-radius:12px; padding:14px 16px;
        box-shadow:0 2px 10px rgba(0,0,0,0.06); text-align:center;
    }

    .role-card {
        background:white; border-radius:16px; padding:22px;
        text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.08);
        border-top:5px solid; height:100%;
    }
    .role-card.gold   { border-color:#f59e0b; }
    .role-card.silver { border-color:#9ca3af; }
    .role-card.bronze { border-color:#b45309; }
    .role-title { font-size:1rem; font-weight:700; margin:6px 0 2px; color:#1e293b; }
    .conf-pct   { font-size:2rem; font-weight:800; margin:6px 0; }
    .gold   .conf-pct { color:#f59e0b; }
    .silver .conf-pct { color:#9ca3af; }
    .bronze .conf-pct { color:#b45309; }
    .role-badge { font-size:0.72rem; background:#e2e8f0; color:#475569;
                  padding:3px 10px; border-radius:20px; }

    .readiness-box {
        background:white; border-radius:14px; padding:20px 24px;
        box-shadow:0 4px 16px rgba(0,0,0,0.07); text-align:center;
    }
    .readiness-score { font-size:3.2rem; font-weight:900; }

    .section-header {
        font-size:1.25rem; font-weight:700; color:#1e293b;
        border-left:5px solid #3b82f6; padding-left:12px;
        margin:22px 0 14px;
    }

    .gap-row { margin-bottom:10px; }
    .gap-label { font-size:0.83rem; color:#334155; font-weight:500; margin-bottom:3px; }
    .gap-bar-bg { background:#e2e8f0; border-radius:8px; height:10px; overflow:hidden; }
    .gap-bar-fill { height:10px; border-radius:8px; }

    .roadmap-step {
        background:white; border-radius:12px; padding:15px 18px;
        margin-bottom:10px; box-shadow:0 2px 8px rgba(0,0,0,0.06);
        border-left:4px solid #3b82f6;
    }
    .step-phase { font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
    .step-title { font-size:0.97rem; font-weight:700; color:#1e293b; margin:2px 0; }
    .step-desc  { font-size:0.83rem; color:#475569; line-height:1.5; }

    .course-card {
        background:white; border-radius:10px; padding:13px 15px;
        margin-bottom:10px; box-shadow:0 2px 8px rgba(0,0,0,0.05);
    }
    .course-platform { font-size:0.68rem; background:#dbeafe; color:#1d4ed8;
        padding:2px 8px; border-radius:12px; font-weight:700; }
    .course-title { font-size:0.9rem; font-weight:600; color:#1e293b; }
    .course-desc  { font-size:0.78rem; color:#64748b; }

    #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match training data exactly)
# ─────────────────────────────────────────────────────────────────────────────
BRANCHES        = ['AI', 'CSE', 'Chemical', 'Civil', 'ECE', 'Electrical', 'Mechanical', 'Meta']
STATUS_OPTIONS  = ['4th year student', 'Alumni (Graduated)']
PROJECT_COUNTS  = ['0', '1-2', '3-4', '5+']
PROJECT_DOMAINS = ['Software Development', 'Data Science / AI', 'Core Engineering',
                   'Robotics / Embedded Systems', 'Mixed domains']
INTERNSHIPS     = ['No internship', 'Software Development Internship',
                   'Data Science / AI Internship', 'Core Engineering Internship',
                   'Electronics / Embedded Internship']
PREP_DOMAINS    = ['Software Development', 'Data Science / AI', 'Core Engineering',
                   'Embedded Systems / Electronics', 'Consulting / Management']
TOOLS_LIST      = [
    'Machine Learning / Deep Learning',
    'Data Analysis (Python / Excel / Pandas)',
    'Data Visualization (PowerBI / Tableau)',
    'CAD tools (SolidWorks / CATIA / AutoCAD)',
    'ANSYS / Simulation tools',
    'MATLAB / Simulink',
    'Embedded systems / microcontrollers',
    'Circuit design tools'
]

ROLE_ICONS = {
    'Software Developer':         '💻',
    'Data Analyst':               '📊',
    'Data Scientist':             '🔬',
    'Machine Learning Engineer':  '🤖',
    'DevOps Engineer':            '⚙️',
    'Embedded Systems Engineer':  '🔌',
    'Mechanical Design Engineer': '🏗️',
    'Manufacturing Engineer':     '🏭',
    'Civil Engineer':             '🌉',
}

# Skill requirements for gap analysis — matches notebook's SKILL_REQS
SKILL_REQS = {
    'Software Developer':         {'Python_Proficiency':4,'CPP_Proficiency':4,'Java_Proficiency':4,
                                   'DSA_Understanding':4,'OOP_Understanding':4,'OS_Understanding':3,
                                   'Database_SQL_Understanding':3},
    'Data Analyst':               {'Python_Proficiency':4,'Database_SQL_Understanding':5,
                                   'DSA_Understanding':3,'OOP_Understanding':3},
    'Data Scientist':             {'Python_Proficiency':5,'Database_SQL_Understanding':4,
                                   'DSA_Understanding':4,'MATLAB_Proficiency':3},
    'Machine Learning Engineer':  {'Python_Proficiency':5,'CPP_Proficiency':3,
                                   'DSA_Understanding':4,'Database_SQL_Understanding':3,
                                   'OOP_Understanding':4},
    'DevOps Engineer':            {'Python_Proficiency':3,'OS_Understanding':4,
                                   'Database_SQL_Understanding':3,'OOP_Understanding':3},
    'Embedded Systems Engineer':  {'CPP_Proficiency':4,'MATLAB_Proficiency':3,
                                   'OS_Understanding':4,'OOP_Understanding':3},
    'Mechanical Design Engineer': {'MATLAB_Proficiency':4,'Python_Proficiency':2},
    'Manufacturing Engineer':     {'MATLAB_Proficiency':3,'Python_Proficiency':2},
    'Civil Engineer':             {'MATLAB_Proficiency':3},
}

SKILL_DISPLAY = {
    'Python_Proficiency':        'Python',
    'CPP_Proficiency':           'C / C++',
    'Java_Proficiency':          'Java',
    'MATLAB_Proficiency':        'MATLAB',
    'DSA_Understanding':         'DSA',
    'Database_SQL_Understanding':'Database / SQL',
    'OOP_Understanding':         'OOP',
    'OS_Understanding':          'Operating Systems',
}

SKILL_TIPS = {
    'Python_Proficiency':        'Complete Python for Everybody (Coursera). Practice on HackerRank daily.',
    'CPP_Proficiency':           'Solve 50+ C++ problems on Codeforces. Study STL and memory management.',
    'Java_Proficiency':          'Build a Spring Boot project. Practice OOP design patterns in Java.',
    'MATLAB_Proficiency':        'Complete MATLAB Onramp (MathWorks, free). Work on a simulation project.',
    'DSA_Understanding':         'Solve 100+ LeetCode problems (Easy→Medium). Follow Striver\'s DSA sheet.',
    'Database_SQL_Understanding':'Complete SQL for Data Science (Coursera). Practice on SQLZoo & HackerRank.',
    'OOP_Understanding':         'Study SOLID principles. Implement design patterns in your preferred language.',
    'OS_Understanding':          'Study OS from Galvin. Practice process scheduling and memory management.',
}

# ─────────────────────────────────────────────────────────────────────────────
# ROADMAPS
# ─────────────────────────────────────────────────────────────────────────────
ROADMAPS = {
    'Software Developer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Strengthen DSA & OOP",       "desc":"Solve 50+ LeetCode problems (arrays, trees, DP). Revise SOLID principles and design patterns. Build a Java/Python OOP mini-project."},
        {"phase":"Phase 2 · Weeks 5–10", "title":"Build Full-Stack Projects",   "desc":"Build 2 end-to-end REST API projects with PostgreSQL/MySQL. Use Git/GitHub for every project. Deploy on Heroku or AWS EC2."},
        {"phase":"Phase 3 · Weeks 11–14","title":"System Design Basics",        "desc":"Learn load balancers, caching, and microservices architecture. Study HLD/LLD. Practice explaining architecture decisions out loud."},
        {"phase":"Phase 4 · Weeks 15–16","title":"Interview & Resume Prep",     "desc":"2–3 mock interviews per week. Quantify resume bullets ('Reduced API latency by 30%'). Polish GitHub README files."},
    ],
    'Data Analyst': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Master SQL & Excel",          "desc":"Complete advanced SQL (window functions, CTEs, subqueries). Learn Pivot Tables, VLOOKUP, Power Query in Excel."},
        {"phase":"Phase 2 · Weeks 5–8",  "title":"Python for Data Analysis",    "desc":"Become proficient in Pandas, NumPy, Matplotlib, Seaborn. Analyse 3 real Kaggle datasets with full EDA + storytelling."},
        {"phase":"Phase 3 · Weeks 9–12", "title":"Business Intelligence Tools", "desc":"Build 2–3 interactive dashboards in Power BI or Tableau connected to a live database."},
        {"phase":"Phase 4 · Weeks 13–16","title":"Domain Portfolio",            "desc":"Pick 1 domain (finance/healthcare/e-commerce). Complete an end-to-end analytics project with insights presentation."},
    ],
    'Data Scientist': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Maths & Statistics Foundation","desc":"Revise Linear Algebra, Probability, and Statistics (hypothesis testing). Use Khan Academy + 3Blue1Brown."},
        {"phase":"Phase 2 · Weeks 5–10", "title":"Core ML Algorithms",           "desc":"Implement Regression, Decision Trees, RF, SVM, KNN from scratch. Deeply understand bias-variance tradeoff."},
        {"phase":"Phase 3 · Weeks 11–14","title":"End-to-End ML Projects",       "desc":"Complete 2 Kaggle competitions. Build full pipeline: data ingestion → EDA → feature engineering → model → evaluation."},
        {"phase":"Phase 4 · Weeks 15–16","title":"Deploy & Publish",             "desc":"Deploy a model as a FastAPI endpoint on Render. Write a Medium article explaining your project."},
    ],
    'Machine Learning Engineer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Python & ML Fundamentals",   "desc":"Master Scikit-learn, Pandas, NumPy. Understand cross-validation, hyperparameter tuning, and evaluation metrics."},
        {"phase":"Phase 2 · Weeks 5–10", "title":"Deep Learning & Frameworks",  "desc":"Learn TensorFlow or PyTorch. Build CNNs, RNNs, Transformers. Complete fast.ai or deeplearning.ai specialisation."},
        {"phase":"Phase 3 · Weeks 11–14","title":"MLOps & Deployment",          "desc":"Learn Docker, MLflow, CI/CD pipelines. Deploy models with FastAPI + Docker. Understand drift detection."},
        {"phase":"Phase 4 · Weeks 15–16","title":"Research + Open Source",      "desc":"Read 2–3 papers on Arxiv. Contribute to an open-source ML project. Build a personal ML portfolio website."},
    ],
    'DevOps Engineer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Linux & Shell Scripting",    "desc":"Get comfortable with Linux commands and Bash scripting. Understand file systems, processes, and networking basics."},
        {"phase":"Phase 2 · Weeks 5–9",  "title":"Docker & Kubernetes",         "desc":"Containerise an app with Docker. Deploy on a Kubernetes cluster. Learn Helm charts and basic cluster management."},
        {"phase":"Phase 3 · Weeks 10–13","title":"CI/CD & Cloud",              "desc":"Set up a Jenkins or GitHub Actions pipeline. Get AWS/GCP/Azure fundamentals (free tier). Earn one cloud certification."},
        {"phase":"Phase 4 · Weeks 14–16","title":"Monitoring & Portfolio",      "desc":"Set up Prometheus + Grafana. Document 2–3 DevOps projects on GitHub with detailed READMEs."},
    ],
    'Embedded Systems Engineer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"C/C++ & Microcontrollers",   "desc":"Master pointers, memory management, bitwise ops in C. Program Arduino/STM32/ESP32. Work with GPIO, interrupts, timers."},
        {"phase":"Phase 2 · Weeks 5–9",  "title":"Communication Protocols",    "desc":"Implement UART, SPI, I2C. Interface sensors (IMU, temperature). Build a hardware + software mini-project."},
        {"phase":"Phase 3 · Weeks 10–13","title":"RTOS & Circuit Design",      "desc":"Learn FreeRTOS (tasks, queues, semaphores). Design PCBs using KiCad or EasyEDA. Understand signal conditioning."},
        {"phase":"Phase 4 · Weeks 14–16","title":"Portfolio & Certification",   "desc":"Build a complete embedded project (IoT device or robot). Document on GitHub + Hackaday."},
    ],
    'Mechanical Design Engineer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"CAD Proficiency",            "desc":"Master SolidWorks or CATIA — assemblies, drawings, GD&T. Complete 10+ complex models. Pursue CSWA certification."},
        {"phase":"Phase 2 · Weeks 5–9",  "title":"FEA & Simulation",           "desc":"Learn ANSYS Mechanical for static, thermal, and fatigue analysis. Validate designs against material limits."},
        {"phase":"Phase 3 · Weeks 10–13","title":"Manufacturing Knowledge",    "desc":"Study casting, forging, CNC. Understand Design for Manufacturability (DFM) principles."},
        {"phase":"Phase 4 · Weeks 14–16","title":"Design Portfolio",           "desc":"Build 3–5 design projects. Document with engineering drawings + simulation results."},
    ],
    'Manufacturing Engineer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Manufacturing Processes",    "desc":"Study casting, forging, machining, welding, additive manufacturing. Understand process selection and tolerance analysis."},
        {"phase":"Phase 2 · Weeks 5–8",  "title":"Quality & Lean Manufacturing","desc":"Learn Six Sigma DMAIC, 5S, Kaizen, SPC. Understand ISO 9001. Practice with quality control case studies."},
        {"phase":"Phase 3 · Weeks 9–12", "title":"CAD & Simulation",           "desc":"Become proficient in SolidWorks manufacturing drawings. Learn basic ANSYS for stress analysis."},
        {"phase":"Phase 4 · Weeks 13–16","title":"Industry Certifications",    "desc":"Pursue Six Sigma Yellow/Green Belt. Build a process optimisation project. Network on LinkedIn."},
    ],
    'Civil Engineer': [
        {"phase":"Phase 1 · Weeks 1–4",  "title":"Core Structural Concepts",   "desc":"Revise structural mechanics, fluid mechanics, soil mechanics. Study IS code provisions for RCC and steel design."},
        {"phase":"Phase 2 · Weeks 5–9",  "title":"Software Tools",             "desc":"Master AutoCAD 2D/3D. Learn STAAD.Pro or ETABS for structural analysis."},
        {"phase":"Phase 3 · Weeks 10–13","title":"Estimation & Project Mgmt", "desc":"Learn BOQ preparation, rate analysis, MS Project. Study contract management and construction safety."},
        {"phase":"Phase 4 · Weeks 14–16","title":"Certifications & Portfolio", "desc":"Pursue GATE if higher studies desired. Build a portfolio with 2–3 structural design projects + drawings."},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# COURSE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
COURSES = {
    'Software Developer': [
        {"platform":"LeetCode",   "title":"LeetCode 75 Study Plan",                   "desc":"Top 75 DSA problems covering all major patterns. Free & structured.",       "url":"https://leetcode.com/studyplan/leetcode-75/"},
        {"platform":"Coursera",   "title":"Meta Back-End Developer Certificate",       "desc":"Django, APIs, Databases, Python — by Meta engineers.",                      "url":"https://www.coursera.org/professional-certificates/meta-back-end-developer"},
        {"platform":"YouTube",    "title":"Traversy Media – Full Stack Crash Courses", "desc":"Free in-depth tutorials on JS, React, Node.js.",                            "url":"https://www.youtube.com/@TraversyMedia"},
        {"platform":"Udemy",      "title":"The Complete 2024 Web Dev Bootcamp",        "desc":"HTML→CSS→JS→React→Node→MongoDB by Dr. Angela Yu.",                         "url":"https://www.udemy.com/course/the-complete-web-development-bootcamp/"},
    ],
    'Data Analyst': [
        {"platform":"Coursera",   "title":"Google Data Analytics Certificate",         "desc":"8-course program: SQL, R, Tableau, data storytelling.",                     "url":"https://www.coursera.org/professional-certificates/google-data-analytics"},
        {"platform":"YouTube",    "title":"Alex The Analyst – SQL & Power BI",         "desc":"Best free channel for SQL queries and Power BI dashboards.",                 "url":"https://www.youtube.com/@AlexTheAnalyst"},
        {"platform":"Kaggle",     "title":"Pandas & SQL Courses (Free)",               "desc":"Hands-on micro-courses by Kaggle. Perfect for quick skill building.",        "url":"https://www.kaggle.com/learn"},
        {"platform":"Udemy",      "title":"Microsoft Power BI – Up & Running",         "desc":"Build real-world dashboards. By Maven Analytics.",                          "url":"https://www.udemy.com/course/microsoft-power-bi-up-running-with-power-bi-desktop/"},
    ],
    'Data Scientist': [
        {"platform":"Coursera",   "title":"IBM Data Science Professional Certificate", "desc":"10-course series: Python, SQL, ML, visualisation, capstone.",                "url":"https://www.coursera.org/professional-certificates/ibm-data-science"},
        {"platform":"fast.ai",    "title":"Practical Deep Learning for Coders",        "desc":"Top-down deep learning approach. Free. By Jeremy Howard.",                  "url":"https://course.fast.ai/"},
        {"platform":"Kaggle",     "title":"Intro to ML + Feature Engineering",         "desc":"Free hands-on learning tracks for competitions.",                           "url":"https://www.kaggle.com/learn/intro-to-machine-learning"},
        {"platform":"YouTube",    "title":"StatQuest with Josh Starmer",              "desc":"Best channel to understand statistics and ML algorithms visually.",          "url":"https://www.youtube.com/@statquest"},
    ],
    'Machine Learning Engineer': [
        {"platform":"deeplearning.ai","title":"Machine Learning Specialization",       "desc":"Andrew Ng's 3-course ML specialisation. Industry standard.",                "url":"https://www.coursera.org/specializations/machine-learning-introduction"},
        {"platform":"deeplearning.ai","title":"MLOps Specialization",                  "desc":"4 courses on production ML, pipelines, and monitoring.",                    "url":"https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops"},
        {"platform":"YouTube",    "title":"Andrej Karpathy – Neural Networks Zero to Hero","desc":"Build GPT from scratch. Best for understanding LLM internals.",          "url":"https://www.youtube.com/@AndrejKarpathy"},
        {"platform":"Udemy",      "title":"PyTorch for Deep Learning Bootcamp",        "desc":"Hands-on PyTorch: CNNs, RNNs, GANs, Transfer Learning.",                   "url":"https://www.udemy.com/course/pytorch-for-deep-learning/"},
    ],
    'DevOps Engineer': [
        {"platform":"Coursera",   "title":"Google Cloud DevOps Engineer Certificate",  "desc":"SRE practices, CI/CD, monitoring, incident management.",                    "url":"https://www.coursera.org/professional-certificates/sre-devops-engineer-google-cloud"},
        {"platform":"Udemy",      "title":"Docker & Kubernetes: The Complete Guide",   "desc":"Most comprehensive Docker + K8s course. By Stephen Grider.",               "url":"https://www.udemy.com/course/docker-and-kubernetes-the-complete-guide/"},
        {"platform":"YouTube",    "title":"TechWorld with Nana – DevOps Tutorials",   "desc":"Free tutorials on Jenkins, Terraform, Kubernetes, AWS.",                    "url":"https://www.youtube.com/@TechWorldwithNana"},
        {"platform":"AWS",        "title":"AWS Cloud Practitioner (Free Prep)",        "desc":"Free exam prep on AWS Skill Builder platform.",                             "url":"https://explore.skillbuilder.aws/learn"},
    ],
    'Embedded Systems Engineer': [
        {"platform":"edX",        "title":"Embedded Systems – Shape the World (UTAustin)","desc":"Hands-on ARM Cortex-M4 programming. Free to audit.",                    "url":"https://www.edx.org/learn/embedded-systems/the-university-of-texas-at-austin-embedded-systems-shape-the-world-microcontroller-inputoutput"},
        {"platform":"YouTube",    "title":"Dronebot Workshop – Arduino & Electronics","desc":"Practical tutorials on microcontrollers and sensor interfacing.",             "url":"https://www.youtube.com/@DronebotWorkshop"},
        {"platform":"Udemy",      "title":"Mastering Microcontroller with Embedded C", "desc":"GPIO, timers, UART, SPI, I2C from scratch.",                               "url":"https://www.udemy.com/course/mastering-microcontroller-with-peripheral-driver-development/"},
        {"platform":"Coursera",   "title":"Introduction to FPGA Design",               "desc":"Verilog/VHDL for FPGA programming. By University of Colorado.",             "url":"https://www.coursera.org/learn/fpga-hardware-description-languages"},
    ],
    'Mechanical Design Engineer': [
        {"platform":"YouTube",    "title":"SOLIDWORKS Tutorials by CADImagineer",     "desc":"Free structured SolidWorks from beginner to advanced.",                     "url":"https://www.youtube.com/@CADImagineer"},
        {"platform":"Udemy",      "title":"ANSYS Mechanical FEA Simulation",           "desc":"Static, thermal, modal, fatigue simulation from scratch.",                  "url":"https://www.udemy.com/course/ansys-mechanical-fea-simulation/"},
        {"platform":"Coursera",   "title":"CAD and Digital Manufacturing Specialization","desc":"Autodesk tools and digital manufacturing workflows.",                     "url":"https://www.coursera.org/specializations/cad-digital-manufacturing"},
        {"platform":"YouTube",    "title":"NPTEL – Mechanical Engineering (IIT)",     "desc":"Free IIT lecture series on machine design and manufacturing.",               "url":"https://www.youtube.com/c/iit"},
    ],
    'Manufacturing Engineer': [
        {"platform":"Coursera",   "title":"Lean Six Sigma Yellow Belt",                "desc":"Process improvement methodology. Widely respected credential.",             "url":"https://www.coursera.org/learn/six-sigma-define-measure-advanced"},
        {"platform":"YouTube",    "title":"The Manufacturing Guy",                    "desc":"Practical videos on CNC, casting, quality systems.",                        "url":"https://www.youtube.com/@themanufacturingguy"},
        {"platform":"Udemy",      "title":"SolidWorks for Manufacturing Engineers",    "desc":"Technical drawings, GD&T, BOMs for production-ready designs.",             "url":"https://www.udemy.com/course/solidworks-for-beginners-and-job-seekers/"},
        {"platform":"edX",        "title":"Supply Chain Management MicroMasters (MIT)","desc":"MIT's supply chain programme — production planning, logistics.",            "url":"https://www.edx.org/masters/micromasters/mitx-supply-chain-management"},
    ],
    'Civil Engineer': [
        {"platform":"YouTube",    "title":"Structville – Structural Design Tutorials", "desc":"RCC and steel design as per IS/BS codes. Very practical.",                 "url":"https://www.youtube.com/@Structville"},
        {"platform":"Udemy",      "title":"AutoCAD 2024 – From Zero to Hero",          "desc":"2D + 3D civil engineering drawings. Beginner friendly.",                    "url":"https://www.udemy.com/course/autocad-2019-from-zero-to-hero/"},
        {"platform":"Coursera",   "title":"Construction Project Management",           "desc":"By Columbia University. Planning, scheduling, risk, cost.",                  "url":"https://www.coursera.org/learn/construction-project-management"},
        {"platform":"YouTube",    "title":"NPTEL – Civil Engineering Lectures (IIT)", "desc":"Free IIT lecture series covering all core civil engineering topics.",        "url":"https://www.youtube.com/c/iit"},
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML model... ⏳")
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), "models")
    try:
        with open(f"{base}/best_model.pkl",      "rb") as f: model       = pickle.load(f)
        with open(f"{base}/scaler.pkl",          "rb") as f: scaler      = pickle.load(f)
        with open(f"{base}/label_encoders.pkl",  "rb") as f: le_dict     = pickle.load(f)
        with open(f"{base}/target_encoder.pkl",  "rb") as f: le_target   = pickle.load(f)
        with open(f"{base}/mlb_tools.pkl",       "rb") as f: mlb         = pickle.load(f)
        with open(f"{base}/feature_columns.pkl", "rb") as f: feat_cols   = pickle.load(f)
        return model, scaler, le_dict, le_target, mlb, feat_cols
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}\n\nMake sure the `models/` folder with all .pkl files is in the same directory as app.py")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC  (matches notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def predict(profile, model, scaler, le_dict, le_target, mlb, feat_cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    needs_scaling = isinstance(model, (LogisticRegression, SVC, KNeighborsClassifier))

    # Numeric features
    features = {}
    num_keys = ['CGPA','Python_Proficiency','CPP_Proficiency','Java_Proficiency',
                'MATLAB_Proficiency','DSA_Understanding','Database_SQL_Understanding',
                'OOP_Understanding','OS_Understanding','Confidence_Level']
    for k in num_keys:
        features[k] = profile[k]

    # Categorical features
    cat_cols = ['Engineering_Branch','Student_Status','Project_Count',
                'Project_Domain','Internship_Experience','Preparation_Domain']
    for col in cat_cols:
        le  = le_dict[col]
        val = profile[col]
        features[f'{col}_Encoded'] = int(le.transform([val])[0]) if val in le.classes_ else 0

    # Tools (MultiLabelBinarizer)
    tools_bin  = mlb.transform([profile['Technical_Tools']])
    tool_names = [f'Tool_{t.split("/")[0].split("(")[0].strip().replace(" ","_")}'
                  for t in mlb.classes_]
    for i, col in enumerate(tool_names):
        features[col] = int(tools_bin[0][i])

    # Align to training feature order
    input_df = pd.DataFrame([features])
    for col in feat_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feat_cols]

    input_data = scaler.transform(input_df) if needs_scaling else input_df.values

    proba    = model.predict_proba(input_data)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    top3     = [(le_target.classes_[i], round(proba[i]*100, 1)) for i in top3_idx]

    # Readiness score (formula from notebook)
    skill_vals  = [profile[s] for s in ['Python_Proficiency','CPP_Proficiency','Java_Proficiency',
                                        'DSA_Understanding','Database_SQL_Understanding',
                                        'OOP_Understanding','OS_Understanding']]
    skill_score = np.mean(skill_vals) / 5 * 40
    cgpa_score  = (profile['CGPA'] / 10) * 30
    conf_score  = (profile['Confidence_Level'] / 5) * 15
    proj_bonus  = {'0':0, '1-2':5, '3-4':10, '5+':15}
    proj_score  = proj_bonus.get(profile['Project_Count'], 0)
    readiness   = round(skill_score + cgpa_score + conf_score + proj_score, 1)

    return top3, readiness

def get_skill_gap(profile, role):
    reqs = SKILL_REQS.get(role, {})
    gaps, strengths = [], []
    for skill, req in reqs.items():
        cur = profile.get(skill, 1)
        if cur >= req:
            strengths.append((SKILL_DISPLAY.get(skill, skill), cur, req))
        else:
            gaps.append({'skill':SKILL_DISPLAY.get(skill,skill),'raw':skill,
                         'current':cur,'required':req,'gap':req-cur,
                         'tip':SKILL_TIPS.get(skill,'Practice more.')})
    return sorted(gaps, key=lambda x: x['gap'], reverse=True), strengths

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def radar_chart(profile, role):
    skills  = ['Python','C/C++','Java','MATLAB','DSA','SQL','OOP','OS']
    sk_keys = ['Python_Proficiency','CPP_Proficiency','Java_Proficiency','MATLAB_Proficiency',
               'DSA_Understanding','Database_SQL_Understanding','OOP_Understanding','OS_Understanding']
    ideal_p = SKILL_REQS.get(role, {})
    user_v  = [profile.get(k, 1) for k in sk_keys]
    ideal_v = [ideal_p.get(k, 1) for k in sk_keys]
    theta   = skills + [skills[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_v+[ideal_v[0]], theta=theta, fill='toself',
        name='Ideal', line_color='#3b82f6', fillcolor='rgba(59,130,246,0.15)', line_width=2))
    fig.add_trace(go.Scatterpolar(r=user_v+[user_v[0]],  theta=theta, fill='toself',
        name='You',   line_color='#f59e0b', fillcolor='rgba(245,158,11,0.25)', line_width=2))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])),
        showlegend=True, height=320, margin=dict(l=40,r=40,t=20,b=30),
        legend=dict(orientation='h',yanchor='bottom',y=-0.18,xanchor='center',x=0.5),
        paper_bgcolor='rgba(0,0,0,0)')
    return fig

def conf_bar(top3):
    colors = ['#f59e0b','#9ca3af','#b45309']
    fig = go.Figure(go.Bar(
        x=[c for _,c in top3], y=[r for r,_ in top3],
        orientation='h', marker_color=colors,
        text=[f"{c}%" for _,c in top3], textposition='auto',
        textfont=dict(color='white', size=13, family='Arial Black')
    ))
    fig.update_layout(xaxis=dict(range=[0,100],title='Confidence %',ticksuffix='%'),
        yaxis=dict(autorange='reversed'), height=170,
        margin=dict(l=10,r=20,t=10,b=30), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model, scaler, le_dict, le_target, mlb, feat_cols = load_artifacts()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 Your Profile")
        st.markdown("---")
        st.markdown("### 📚 Academic Info")
        branch      = st.selectbox("Engineering Branch", BRANCHES)
        status      = st.selectbox("Your Status",        STATUS_OPTIONS)
        cgpa        = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)

        st.markdown("---")
        st.markdown("### 💻 Programming Skills (1–5)")
        python  = st.slider("Python",  1, 5, 3)
        cpp     = st.slider("C / C++", 1, 5, 2)
        java    = st.slider("Java",    1, 5, 2)
        matlab  = st.slider("MATLAB",  1, 5, 1)

        st.markdown("---")
        st.markdown("### 🧠 Concept Understanding (1–5)")
        dsa     = st.slider("DSA",              1, 5, 3)
        sql     = st.slider("Database / SQL",   1, 5, 3)
        oop     = st.slider("OOP",              1, 5, 3)
        os_s    = st.slider("Operating Systems",1, 5, 2)

        st.markdown("---")
        st.markdown("### 🛠️ Technical Tools")
        tools_used = st.multiselect("Select all that apply", TOOLS_LIST)

        st.markdown("---")
        st.markdown("### 📁 Projects & Experience")
        proj_count  = st.selectbox("Projects Completed",    PROJECT_COUNTS)
        proj_domain = st.selectbox("Project Domain",        PROJECT_DOMAINS)
        internship  = st.selectbox("Internship Experience", INTERNSHIPS)

        st.markdown("---")
        st.markdown("### 🎯 Placement Prep")
        prep_domain = st.selectbox("Domain Prepared For", PREP_DOMAINS)
        confidence  = st.slider("Confidence Level (1–5)", 1, 5, 3)
        st.markdown("---")
        btn = st.button("🚀 Predict My Job Role", use_container_width=True, type="primary")

    # ── HEADER ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-banner">
        <h1>🎓 Placement Readiness &amp; Job Role Predictor</h1>
        <p>Fill in your profile on the left → Get Top 3 predicted job roles, skill gap analysis, personalised roadmap &amp; course recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.markdown('<div class="metric-box"><div style="font-size:1.5rem">🤖</div><div style="font-size:1.3rem;font-weight:800;color:#3b82f6">Trained</div><div style="font-size:0.72rem;color:#64748b">Model Loaded</div></div>', unsafe_allow_html=True)
    with m2: st.markdown('<div class="metric-box"><div style="font-size:1.5rem">🏢</div><div style="font-size:1.3rem;font-weight:800;color:#3b82f6">9</div><div style="font-size:0.72rem;color:#64748b">Job Roles</div></div>', unsafe_allow_html=True)
    with m3: st.markdown('<div class="metric-box"><div style="font-size:1.5rem">📊</div><div style="font-size:1.3rem;font-weight:800;color:#3b82f6">Top 3</div><div style="font-size:0.72rem;color:#64748b">Predictions</div></div>', unsafe_allow_html=True)
    with m4: st.markdown('<div class="metric-box"><div style="font-size:1.5rem">🗺️</div><div style="font-size:1.3rem;font-weight:800;color:#3b82f6">Roadmap</div><div style="font-size:0.72rem;color:#64748b">+ Courses</div></div>', unsafe_allow_html=True)

    if not btn:
        st.markdown("---")
        st.info("👈 Fill in your profile in the sidebar, then click **Predict My Job Role**.")
        return

    # Build profile dict
    profile = {
        'Engineering_Branch': branch, 'Student_Status': status, 'CGPA': cgpa,
        'Python_Proficiency': python, 'CPP_Proficiency': cpp,
        'Java_Proficiency': java, 'MATLAB_Proficiency': matlab,
        'DSA_Understanding': dsa, 'Database_SQL_Understanding': sql,
        'OOP_Understanding': oop, 'OS_Understanding': os_s,
        'Technical_Tools': tools_used if tools_used else [],
        'Project_Count': proj_count, 'Project_Domain': proj_domain,
        'Internship_Experience': internship,
        'Preparation_Domain': prep_domain, 'Confidence_Level': confidence
    }

    top3, readiness = predict(profile, model, scaler, le_dict, le_target, mlb, feat_cols)
    st.markdown("---")

    # ── READINESS + TOP 3 CARDS ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Your Results</div>', unsafe_allow_html=True)
    r_col, c1, c2, c3 = st.columns([1.1, 1, 1, 1])

    # Readiness
    if readiness >= 75:   r_color,r_label = "#16a34a","🌟 Excellent"
    elif readiness >= 60: r_color,r_label = "#f59e0b","👍 Good"
    elif readiness >= 45: r_color,r_label = "#ea580c","📚 Average"
    else:                 r_color,r_label = "#dc2626","⚠️ Needs Work"

    with r_col:
        st.markdown(f"""
        <div class="readiness-box">
            <div style="font-size:0.8rem;color:#64748b;text-transform:uppercase;letter-spacing:1px">Readiness Score</div>
            <div class="readiness-score" style="color:{r_color}">{readiness}<span style="font-size:1.2rem;color:#94a3b8">/100</span></div>
            <div style="font-size:0.85rem;font-weight:600;color:{r_color}">{r_label}</div>
        </div>
        """, unsafe_allow_html=True)

    cards = [("gold","🥇 Best Match"),(  "silver","🥈 2nd Match"),("bronze","🥉 3rd Match")]
    for col, (role, conf), (style, badge) in zip([c1,c2,c3], top3, cards):
        icon = ROLE_ICONS.get(role, '🎯')
        with col:
            st.markdown(f"""
            <div class="role-card {style}">
                <div style="font-size:2.2rem">{icon}</div>
                <div class="role-title">{role}</div>
                <div class="conf-pct">{conf}%</div>
                <div class="role-badge">{badge}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(conf_bar(top3), use_container_width=True)

    # ── SKILL GAP ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Skill Gap Analysis</div>', unsafe_allow_html=True)
    sel_role = st.selectbox("Analyse gap for:", [r for r,_ in top3], key="gap_sel")

    gaps, strengths = get_skill_gap(profile, sel_role)
    col_r, col_g = st.columns([1, 1])

    with col_r:
        st.markdown(f"**Radar — You vs Ideal {sel_role}**")
        st.plotly_chart(radar_chart(profile, sel_role), use_container_width=True)

    with col_g:
        st.markdown("**Skill Breakdown**")
        all_reqs = SKILL_REQS.get(sel_role, {})
        for raw_key, disp in SKILL_DISPLAY.items():
            if raw_key not in all_reqs:
                continue
            req = all_reqs[raw_key]
            cur = profile.get(raw_key, 1)
            pct = int((cur / 5) * 100)
            if cur >= req:  color, status_txt = "#16a34a", "✅ On track"
            elif req-cur==1:color, status_txt = "#f59e0b", "⚠️ Minor gap"
            else:           color, status_txt = "#dc2626", "❌ Needs work"
            st.markdown(f"""
            <div class="gap-row">
              <div class="gap-label">{disp} — {cur}/5 &nbsp;|&nbsp; Ideal: {req}/5 &nbsp;
                <span style="color:{color};font-size:0.73rem">{status_txt}</span></div>
              <div class="gap-bar-bg"><div class="gap-bar-fill" style="width:{pct}%;background:{color}"></div></div>
            </div>""", unsafe_allow_html=True)

        if gaps:
            st.markdown("**💡 Top Tips to Close Your Gaps:**")
            for g in gaps[:3]:
                with st.expander(f"↗ {g['skill']} (gap: {g['gap']} point{'s' if g['gap']>1 else ''})"):
                    st.markdown(g['tip'])

    # ── ROADMAP ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺️ Personalised 16-Week Roadmap</div>', unsafe_allow_html=True)
    tabs_r = st.tabs([f"{ROLE_ICONS.get(r,'🎯')} {r}" for r,_ in top3])
    for i,(role,conf) in enumerate(top3):
        with tabs_r[i]:
            st.markdown(f"#### {ROLE_ICONS.get(role,'🎯')} {role} — {conf}% match")
            for step in ROADMAPS.get(role,[]):
                st.markdown(f"""
                <div class="roadmap-step">
                    <div class="step-phase">{step['phase']}</div>
                    <div class="step-title">{step['title']}</div>
                    <div class="step-desc">{step['desc']}</div>
                </div>""", unsafe_allow_html=True)

    # ── COURSES ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Recommended Courses & Resources</div>', unsafe_allow_html=True)
    tabs_c = st.tabs([f"{ROLE_ICONS.get(r,'🎯')} {r}" for r,_ in top3])
    for i,(role,conf) in enumerate(top3):
        with tabs_c[i]:
            st.markdown(f"#### Best courses to become a **{role}**")
            for c in COURSES.get(role,[]):
                st.markdown(f"""
                <div class="course-card">
                    <span class="course-platform">{c['platform']}</span>&nbsp;
                    <span class="course-title"><a href="{c['url']}" target="_blank">{c['title']}</a></span><br>
                    <span class="course-desc">{c['desc']}</span>
                </div>""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""<div style="text-align:center;color:#94a3b8;font-size:0.78rem;padding:8px">
        🎓 Placement Readiness &amp; Job Role Predictor &nbsp;|&nbsp; Applied Machine Learning Project
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
