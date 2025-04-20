import streamlit as st
import os
import pandas as pd
from utilis import preprocess_data, train_models, save_model

# Configuration de la page
st.set_page_config(page_title="Application ML", layout="wide")
st.markdown("<h1 style='color:#1f77b4;'>🧠 Projet 2 Oussama</h1>", unsafe_allow_html=True)

# Téléchargement de fichier
with st.container():
    st.markdown("### 📂 Importer un fichier CSV")
    fichier = st.file_uploader("Sélectionnez votre fichier", type=["csv"])

if fichier:
    try:
        data = pd.read_csv(fichier)

        with open(f"data/{fichier.name}", "wb") as f:
            f.write(fichier.getbuffer())

        st.success("✅ Fichier importé avec succès.")
        with st.expander("🔎 Aperçu des données (premières lignes)"):
            st.dataframe(data.head(), use_container_width=True)

        # Étape de configuration du modèle
        st.markdown("---")
        st.markdown("### ⚙️ Configuration de l'entraînement")
        col1, col2 = st.columns(2)

        with col1:
            type_tache = st.selectbox("📌 Type de tâche", ["Classification", "Régression"])

        with col2:
            variable_cible = st.selectbox("🎯 Variable cible", data.columns)

        if st.button("🚀 Entraîner les modèles", use_container_width=True):
            try:
                with st.spinner("⏳ Prétraitement et entraînement des modèles..."):
                    X, y = preprocess_data(data, variable_cible, type_tache)
                    models, metrics = train_models(X, y, type_tache)
                    save_model(models, f"{type_tache}_model")

                st.success("🎉 Entraînement terminé avec succès.")
                st.markdown("### 📊 Performances des modèles")
                st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)

                st.session_state["models"] = models
                st.session_state["features"] = list(X.columns)
                st.session_state["task_type"] = type_tache

            except ValueError as ve:
                st.error(f"❌ Erreur : {ve}")
            except Exception as e:
                st.error(f"❗ Problème inattendu : {e}")

        # Section prédiction
        if "models" in st.session_state:
            st.markdown("---")
            st.markdown("### 🔮 Prédiction à partir de nouvelles valeurs")

            modele_choisi = st.selectbox("🧠 Choisir un modèle", list(st.session_state["models"].keys()))

            valeurs = []
            with st.form("formulaire_prediction"):
                for feature in st.session_state["features"]:
                    valeur = st.text_input(f"🧾 {feature}", "0")
                    try:
                        valeurs.append(float(valeur))
                    except ValueError:
                        st.warning(f"Valeur non valide pour {feature} : '{valeur}'")

                bouton = st.form_submit_button("🔍 Lancer la prédiction")
                if bouton:
                    try:
                        model = st.session_state["models"][modele_choisi]
                        prediction = model.predict([valeurs])
                        st.success(f"🎯 Résultat : {prediction[0]}")
                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {e}")

    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
