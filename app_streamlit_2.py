import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Dashboard Insécurité Alimentaire", layout="wide")

# ============================================================
# 🔹 Chargement des modèles
# ============================================================
rf_model = joblib.load("modele_food_insecurity.pkl")
xgb_model = joblib.load("modele_food_insecurity_1.pkl")

# ============================================================
# 🔹 Chargement des données test
# ============================================================
X_test_selected = pd.read_csv("X_test_selected.csv")
y_test_mapped = pd.read_csv("y_test_mapped.csv")["target"]

# ============================================================
# 🔹 Chargement des bases (brute/nettoyée + normalisée)
# ============================================================
@st.cache_data
def load_all_data():
    df_raw = pd.read_csv("data_encoded_1.csv")
    df_clean = pd.read_csv("data_clean.csv")
    df_norm = pd.read_csv("DDDIEMMME-D.xlsx")
    return df_raw, df_clean, df_norm

df, df_clean, df_norm = load_all_data()

# ============================================================
# 🔹 Merge pour calculer la prévalence par région
# ============================================================
st.title("📊 Dashboard Insécurité Alimentaire")

st.subheader("🗺️ Prévalence de l'insécurité alimentaire par région")

if "id" not in df_clean.columns or "id" not in df_norm.columns:
    st.error("❌ Les deux bases doivent contenir une colonne 'id' pour le merge.")
else:
    df_merged = df_clean.merge(df_norm[["id", "target"]], on="id", how="inner")
    df_merged["target_label"] = df_merged["target"].map({0: "Modérée", 1: "Sévère"})

    prevalence_region = (
        df_merged.groupby("region")["target"]
        .mean()
        .reset_index()
        .rename(columns={"target": "prevalence"})
    )
    prevalence_region["prevalence"] = (prevalence_region["prevalence"] * 100).round(2)

    st.write("### 📊 Tableau de prévalence par région (%)")
    st.dataframe(prevalence_region)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=prevalence_region, x="region", y="prevalence", palette="Reds")
    ax.set_title("Prévalence de l'insécurité alimentaire sévère par région (%)")
    ax.set_ylabel("Prévalence (%)")
    ax.set_xlabel("Région")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ============================================================
# 🔹 Analyse exploratoire
# ============================================================
st.subheader("📌 Statistiques descriptives")
st.dataframe(df.describe().round(2))

variables = [
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q603_sauter_un_repas",
    "q601_ne_pas_manger_nourriture_saine_nutritive"
]

st.subheader("📈 Matrice de corrélation")
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(df[variables].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ============================================================
# 🔹 Histogrammes
# ============================================================
st.sidebar.subheader("📊 Histogrammes")
vars_selectionnees = st.sidebar.multiselect("Choisissez les variables :", variables)
couleurs = sns.color_palette("husl", len(vars_selectionnees))

if vars_selectionnees:
    cols = st.columns(2)
    for i, (var, couleur) in enumerate(zip(vars_selectionnees, couleurs)):
        with cols[i % 2]:
            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=10, kde=True, color=couleur, ax=ax)
            ax.set_title(f"Distribution : {var}")
            st.pyplot(fig)

# ============================================================
# 🔹 Performances des modèles
# ============================================================
st.sidebar.subheader("⚙️ Choix du modèle")
modele = st.sidebar.selectbox("Sélectionnez un modèle :", ["Random Forest", "XGBoost"])

rf_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.996152, 0.986885, 0.973770],
    "Test": [0.994231, 0.981481, 0.962963]
})

xgb_perf = pd.DataFrame({
    "Métrique": ["Accuracy", "AUC", "Recall"],
    "Train": [0.996152, 0.986885, 0.973770],
    "Test": [0.994231, 0.981481, 0.962963]
})

if modele == "Random Forest":
    st.subheader("📋 Performance - Random Forest")
    st.dataframe(rf_perf)
else:
    st.subheader("📋 Performance - XGBoost")
    st.dataframe(xgb_perf)

# ============================================================
# 🔹 Matrice de confusion
# ============================================================
st.subheader("📌 Matrice de confusion")

rf_preds = rf_model.predict(X_test_selected)
xgb_preds = xgb_model.predict(X_test_selected)

modele_cm = st.selectbox("Modèle :", ["Random Forest", "XGBoost"], key="cm_selector")
y_pred_test = rf_preds if modele_cm == "Random Forest" else xgb_preds

label_mapping = {0: "Modérée", 1: "Sévère"}
conf_matrix = confusion_matrix(y_test_mapped, y_pred_test)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[label_mapping[x] for x in np.unique(y_test_mapped)],
    yticklabels=[label_mapping[x] for x in np.unique(y_test_mapped)],
    ax=ax
)
ax.set_title(f"Matrice de confusion - {modele_cm}")
st.pyplot(fig)

# ============================================================
# 🔹 Formulaire de prédiction API
# ============================================================
st.title("🧠 Prédiction d'insécurité alimentaire")

q606 = st.number_input("Faim sans manger ?", min_value=0, max_value=10, value=0)
q605 = st.number_input("Manque de nourriture ?", min_value=0, max_value=10, value=0)
q604 = st.number_input("Mangé moins que nécessaire ?", min_value=0, max_value=10, value=0)
q603 = st.number_input("Repas sautés ?", min_value=0, max_value=10, value=0)
q601 = st.number_input("Nourriture peu nutritive ?", min_value=0, max_value=10, value=0)

if st.button("🔍 Lancer la prédiction"):
    payload = {
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q603_sauter_un_repas": q603,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601
    }

    try:
        response = requests.post("https://fastapi-food-insecurity.onrender.com/predict", json=payload)
        result = response.json()

        niveau = result.get("niveau", "inconnu")
        score = result.get("score", 0.00)
        profil = result.get("profil", "inconnu")
        probabilites = result.get("probabilités", {})

        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        else:
            st.success("🟢 Aucun signe d'insécurité alimentaire")

        st.write("### 🔎 Score de risque")
        st.progress(score)

        st.write(f"Profil détecté : **{profil.capitalize()}**")

        fig, ax = plt.subplots()
        ax.pie(probabilites.values(), labels=probabilites.keys(), autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")
