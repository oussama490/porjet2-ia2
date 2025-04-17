import streamlit as st
from PIL import Image as im
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Configuration de la page
st.set_page_config(page_title="Analyse Beans DataSet", layout="wide")

# Menu barre latérale
st.sidebar.title('📊 Projet Beans DataSet')
menu = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "📈 Visualisation"])

if menu =="🏠 Accueil":
    st.markdown(
        """
        <div style='text-align:center;'>
        <h1>Beans and Pods<h1>
        </div>
        <h2>Introduction</h2>
        <p>Beans & Pods, une entreprise spécialisée dans la commercialisation de grains
          de café et de gousses, a récemment élargi son activité en lançant une plateforme en
         ligne avec le soutien d'Angeli VC. Ce rapport propose une analyse approfondie des ventes,
         en examinant leur répartition par canal (magasin physique et boutique en ligne), par type de
         produit et par région. Il présente également des recommandations stratégiques visant à optimiser
         les performances commerciales et à affiner le ciblage des clients pour maximiser les ventes.</p>
        """, unsafe_allow_html=True
    )

elif menu == "📈 Visualisation":
    try:
        fichier = 'BeansDataSet.csv'
        data = pd.read_csv(fichier)
        

        # Afficher le DataFrame
        st.title("🔍 Analyse du Beans DataSet")
        st.subheader("📋 Aperçu des données")
        st.dataframe(data.head())
# a changer 
    except FileNotFoundError:
        st.error("❌ Erreur : Le fichier 'BeansDataSet.csv' est introuvable.")
        st.stop()

    # Aperçu des données
    st.subheader("Aperçu des données")
    st.subheader("📊 Dimensions du dataset")
    st.write("Nombre de lignes : ", data.shape[0])
    st.write("Nombre de colonnes : ", data.shape[1])

    # Vérification des valeurs manquantes
    st.subheader("🚨 Valeurs manquantes")
    st.write(data.isnull().sum())
    st.write(f"Nombre total de valeurs manquantes : {data.isnull().sum().sum()}")

    # Comptage par 'Channel'   changer
    st.subheader("🛒 Analyse par Channel")
    if 'Channel' in data.columns:
        channel_count = data.groupby('Channel').size()
        st.write("Comptage des Channel :")
        st.bar_chart(channel_count)

    # Total des ventes par produit
    if {'Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino'}.issubset(data.columns):
        data['Total vente'] = data[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']].sum(axis=1)
        
        st.subheader("📦Total des ventes")
        total_vente = data[['Robusta', 'Arabica', 'Espresso', 'Lungo', 'Latte', 'Cappuccino']].sum()
        st.write(total_vente)

        # Ventes par région
        if 'Region' in data.columns:
            region_ventes = data.groupby('Region')['Total vente'].sum()
            st.subheader("Ventes par Région")
            st.bar_chart(region_ventes)

    # Statistiques descriptives
    st.subheader("📊 Statistiques descriptives")
    st.write(data.describe())

    # Histogrammes
    st.subheader("📊 Histogrammes")
    fig, ax = plt.subplots(figsize=(15, 10))
    data.hist(bins=15, ax=ax, layout=(3, 3), grid=True)
    st.pyplot(fig)

    try:
        fichier = 'BeansDataSet.csv'
        data = pd.read_csv(fichier)
#changer fillna
        data.fillna(0, inplace=True)

        st.title("📊Graphiques de densité pour chaque colonne")
#changer
        numeric_cols = data.select_dtypes(include=['number']).columns

        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
            axes = axes.flatten()
#changer
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    sns.kdeplot(data[col], ax=axes[i], fill=True)
                    axes[i].set_title(f"Densité de {col}")

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Aucune colonne numérique trouvée dans le dataset.")
    except Exception as e:
        st.error(f"Erreur lors de la génération des graphiques de densité : {e}")

    # Matrice de corrélation
    st.subheader("📌 Matrice de corrélation")
    data_num = data.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(15, 10))
    corr = data_num.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.subheader('Graphes de densité')
    data.plot(kind='density', subplots=True, sharex=False, figsize=(12,10))
    st.pyplot(plt.gcf()) #gcf changer

    # Boîtes à moustaches
    st.subheader("📊 Boîtes à moustaches")
    fig, ax = plt.subplots(figsize=(15, 15))
    data.plot(kind='box', layout=(3, 3), subplots=True, sharex=False, sharey=False, ax=ax)
    st.pyplot(fig)

    # Pairplot avec Seaborn
    if 'Cappuccino' in data.columns:
        st.subheader("📈 Pairplot (Cappuccino)")
        try:
            pairplot_fig = sns.pairplot(data, hue='Cappuccino', diag_kind="kde")
            st.pyplot(pairplot_fig.fig)
        except Exception as e:
            st.error(f"Erreur dans le pairplot (Cappuccino) : {e}")

        st.subheader("📈Pairplot (Arabica et Espresso)")
        try:
            pairplot_fig_2 = sns.pairplot(data, hue='Cappuccino', vars=['Arabica', 'Espresso'], diag_kind="kde")
            st.pyplot(pairplot_fig_2.fig)
        except Exception as e:
            st.error(f"Erreur dans le pairplot (Arabica et Espresso) : {e}")
            
    st.success("✅ Analyse terminée avec succès !")
