import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from tabulate import tabulate

# Page configuration
st.set_page_config(page_title="Customer Segmentation Analysis", layout="wide", page_icon=":label:")
# Set up the title of the app with custom style, FontAwesome icon, and background image
st.markdown("""
    <style>
        .title {
            color: #ff052b;
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            color: #0ca4eb;
            font-size: 17px;
            font-weight: bold;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            color: #1E90FF;
        }
        .metrics {
            color: #28A745;
            font-weight: bold;
        }
        .stApp {
        background-image: url("https://www.bounteous.com/sites/default/files/insights/2022-10/previews/screen_shot_2022-09-01_at_12.34.57_pm.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        }
    </style>
    <div class="title">
        <i class="fas fa-users"></i> Customer Segmentation Analysis
    </div>
""", unsafe_allow_html=True)

# Add a description for the app with a custom color and icon
st.markdown("""
    <div class='subtitle'>
        <i class="fas fa-info-circle"></i> Perform customer segmentation with various clustering techniques, such as K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models. Upload your dataset, select the algorithm, and explore the results.
        The app will display interactive visualizations and performance metrics to help you better understand the segmentation.
    </div>
""", unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader(":orange[Upload your CSV file]", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded dataset
    dataset = pd.read_csv(uploaded_file)
    st.write("### :orange[Dataset Overview:]")
    st.write(dataset.head())

    # 1. **Data Preprocessing**
    st.subheader(':orange[Data Preprocessing]')

    # Display column selection in a sidebar with custom title
    st.sidebar.markdown("<p class='sidebar-title'>Data Selection</p>", unsafe_allow_html=True)
    columns = st.sidebar.multiselect('Select Features', dataset.columns)

    if len(columns) < 2:
        st.error('Please select at least two features for clustering')
    else:
        # Selecting the features from the dataset
        X = dataset[columns].copy()  # Copy the selected columns to avoid modifying the original dataset

        # Handle missing values
        missing_values = dataset.isnull().sum().sum()
        st.write(f":orange[Missing values:] {missing_values}")
        
        if missing_values > 0:
            option = st.selectbox('How would you like to handle missing values?', ['Drop', 'Fill with mean', 'Fill with median'])
            if option == 'Drop':
                X = dataset.dropna(subset=columns).values
            elif option == 'Fill with mean':
                dataset[columns] = dataset[columns].fillna(dataset[columns].mean())
                X = dataset[columns].values
            elif option == 'Fill with median':
                dataset[columns] = dataset[columns].fillna(dataset[columns].median())
                X = dataset[columns].values

        # Encode categorical data (e.g., 'Gender') using Label Encoding
        if 'Gender' in columns:
            le = LabelEncoder()
            X[:, dataset.columns.get_loc('Gender')] = le.fit_transform(dataset['Gender'])

        # Remove outliers based on IQR
        if st.checkbox(":orange[Remove outliers]"):
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            outlier_indices = np.where((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
            X = np.delete(X, outlier_indices[0], axis=0)

        # Scaling Data
        st.write("### :orange[Scaling Data]")
        scaling_method = st.sidebar.selectbox('Scaling Method', ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
        
        if scaling_method == 'StandardScaler':
            scaler = StandardScaler()
        elif scaling_method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaling_method == 'RobustScaler':
            scaler = RobustScaler()
        
        X_scaled = scaler.fit_transform(X)

        # 2. **Clustering Algorithms**
        st.subheader(':orange[Select Clustering Algorithm]')

        clustering_algorithms = ['K-Means', 'Hierarchical Clustering', 'DBSCAN', 'Gaussian Mixture Models', 'Spectral Clustering']
        selected_algorithm = st.selectbox('Choose a Clustering Algorithm:', clustering_algorithms)

        # Initialize cluster labels variable
        y_clusters = None  # This will store the clustering results

        # Create a sidebar for hyperparameter selection
        st.sidebar.header('Hyperparameters')

        # K-Means
        if selected_algorithm == 'K-Means':
            st.write("### :orange[Using K-Means Clustering]")
            kmeans_clusters = st.sidebar.slider("Select number of clusters for K-Means", min_value=2, max_value=10, value=4)
            kmeans = KMeans(n_clusters=kmeans_clusters, init='k-means++', random_state=42)
            y_clusters = kmeans.fit_predict(X_scaled)

            # Performance Metrics
            silhouette_score_kmeans = silhouette_score(X_scaled, y_clusters)
            calinski_kmeans = calinski_harabasz_score(X_scaled, y_clusters)
            davies_kmeans = davies_bouldin_score(X_scaled, y_clusters)

            st.write(f":orange[Silhouette Score (K-Means): {silhouette_score_kmeans:.2f}]")
            st.write(f":orange[Calinski Harabasz Score (K-Means): {calinski_kmeans:.2f}]")
            st.write(f":orange[Davies-Bouldin Score (K-Means): {davies_kmeans:.2f}]")

            # Visualize Clusters
            df_kmeans = pd.DataFrame(X, columns=columns)
            df_kmeans['Cluster'] = y_clusters
            fig_kmeans = px.scatter_3d(df_kmeans, x=columns[0], y=columns[1], z=columns[2], color='Cluster', title="3D K-Means Clustering")
            st.plotly_chart(fig_kmeans)

        # Hierarchical Clustering
        elif selected_algorithm == 'Hierarchical Clustering':
            st.write("### :orange[Using Hierarchical Clustering]")
            hc_clusters = st.sidebar.slider("Select number of clusters for Hierarchical Clustering", min_value=2, max_value=10, value=3)
            hc = AgglomerativeClustering(n_clusters=hc_clusters, linkage='ward')
            y_clusters = hc.fit_predict(X_scaled)

            # Performance Metrics
            silhouette_score_hc = silhouette_score(X_scaled, y_clusters)
            calinski_hc = calinski_harabasz_score(X_scaled, y_clusters)
            davies_hc = davies_bouldin_score(X_scaled, y_clusters)

            st.write(f":orange[Silhouette Score (Hierarchical): {silhouette_score_hc:.2f}]")
            st.write(f":orange[Calinski Harabasz Score (Hierarchical): {calinski_hc:.2f}]")
            st.write(f":orange[Davies-Bouldin Score (Hierarchical): {davies_hc:.2f}]")

            # Visualize Clusters
            df_hc = pd.DataFrame(X, columns=columns)
            df_hc['Cluster'] = y_clusters
            fig_hc = px.scatter_3d(df_hc, x=columns[0], y=columns[1], z=columns[2], color='Cluster', title="3D Hierarchical Clustering")
            st.plotly_chart(fig_hc)

        # DBSCAN
        elif selected_algorithm == 'DBSCAN':
            st.write("### :orange[Using DBSCAN Clustering]")
            eps = st.sidebar.slider('Select epsilon value for DBSCAN', min_value=0.1, max_value=5.0, value=0.5)
            min_samples = st.sidebar.slider('Select minimum samples for DBSCAN', min_value=2, max_value=10, value=5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_clusters = dbscan.fit_predict(X_scaled)

            # Performance Metrics
            silhouette_score_dbscan = silhouette_score(X_scaled, y_clusters)
            calinski_dbscan = calinski_harabasz_score(X_scaled, y_clusters)
            davies_dbscan = davies_bouldin_score(X_scaled, y_clusters)

            st.write(f":orange[Silhouette Score (DBSCAN): {silhouette_score_dbscan:.2f}]")
            st.write(f":orange[Calinski Harabasz Score (DBSCAN): {calinski_dbscan:.2f}]")
            st.write(f":orange[Davies-Bouldin Score (DBSCAN): {davies_dbscan:.2f}]")

            # Visualize Clusters
            df_dbscan = pd.DataFrame(X, columns=columns)
            df_dbscan['Cluster'] = y_clusters
            fig_dbscan = px.scatter_3d(df_dbscan, x=columns[0], y=columns[1], z=columns[2], color='Cluster', title="3D DBSCAN Clustering")
            st.plotly_chart(fig_dbscan)

        # Gaussian Mixture Models
        elif selected_algorithm == 'Gaussian Mixture Models':
            st.write("### :orange[Using Gaussian Mixture Models (GMM)]")
            gmm_clusters = st.sidebar.slider("Select number of clusters for GMM", min_value=2, max_value=10, value=3)
            gmm = GaussianMixture(n_components=gmm_clusters)
            y_clusters = gmm.fit_predict(X_scaled)

            # Performance Metrics
            silhouette_score_gmm = silhouette_score(X_scaled, y_clusters)
            calinski_gmm = calinski_harabasz_score(X_scaled, y_clusters)
            davies_gmm = davies_bouldin_score(X_scaled, y_clusters)

            st.write(f":orange[Silhouette Score (GMM): {silhouette_score_gmm:.2f}]")
            st.write(f":orange[Calinski Harabasz Score (GMM): {calinski_gmm:.2f}]")
            st.write(f":orange[Davies-Bouldin Score (GMM): {davies_gmm:.2f}]")

            # Visualize Clusters
            df_gmm = pd.DataFrame(X, columns=columns)
            df_gmm['Cluster'] = y_clusters
            fig_gmm = px.scatter_3d(df_gmm, x=columns[0], y=columns[1], z=columns[2], color='Cluster', title="3D GMM Clustering")
            st.plotly_chart(fig_gmm)

        # 3. **PCA for Dimensionality Reduction and Cluster Visualization**
        st.subheader(':orange[PCA for Dimensionality Reduction]')
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        df_pca['Cluster'] = y_clusters  # This ensures the correct variable is used
        fig_pca = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', title="2D Cluster Visualization (PCA)")
        st.plotly_chart(fig_pca)

else:
    st.error(":orange[Please upload a CSV file to proceed.]")
