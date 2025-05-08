import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

def NormalSNV(Spectra):
    return Spectra.apply(zscore, axis=1)

def outlier_detection_multiple_datasets(data, PCA_components, contamination, random_state):
    # This function performs outlier detection automatically on multiple Raman spectral datasets using PCA and Isolation Forests.
    # data: Dictionary of pandas DataFrames, where each key is a dataset.
    # PCA_components(type: int): Number of principal components to retain. For practicality, this should be set to 2.
    # contamination (type: float): Estimated proportion of outliers in the data (i.e., from 0 to 1)
    # random_state (type: int): Random seed for reproducibility. Typically set to 42.

    fig = make_subplots(rows = len(data), cols = 1, subplot_titles = list(data.keys()))

    result = {}
    for key in data.keys():
        df = data[key]

        # Exclude the last two columns (if they are not spectral data; comment out if not needed)
        columns_to_select = df.columns[:-2] 
        df = df[columns_to_select]

        # Ensure the columns are numeric
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Apply SNV
        X = NormalSNV(df)

        # Standardise data
        X_mean = X.mean()
        X_std = X.std()
        Z = (X - X_mean) / X_std
        Z = pd.DataFrame(Z, index=X.index, columns=X.columns)

        # PCA transformation
        PCA_components = min(PCA_components, Z.shape[1])  # Ensure PCA components ≤ features
        pca = PCA(n_components=PCA_components)
        scores = pca.fit_transform(Z)
        df_pca = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(PCA_components)], index=X.index)

        explained_variance = pca.explained_variance_ratio_

        # Outlier detection using Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        iso_forest.fit(df_pca)
        labels = iso_forest.predict(df_pca)
        # -1 for outliers, 1 for inliers
        outliers = df_pca[labels == -1]
        inliers = df_pca[labels == 1]

        scores = iso_forest.decision_function(df_pca)

        # Add inliers
        fig.add_trace(go.Scatter(
            x=inliers.iloc[:, 0],
            y=inliers.iloc[:, 1],
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Inliers'
        ) , row = list(data.keys()).index(key) + 1, col = 1)

        # Add outliers
        fig.add_trace(go.Scatter(
            x=outliers.iloc[:, 0],
            y=outliers.iloc[:, 1],
            mode='markers',
            marker=dict(color='red', size=8, symbol='x'),
            name='Outliers'
        ) , row = list(data.keys()).index(key) + 1, col = 1)

        # Update layout
        fig.update_layout(
            autosize = False,
            width = 1000,
            height = 500 * len(data),
            title=f"Isolation Forest Outlier Detection",
            xaxis=dict(
                title=f'PC1 ({explained_variance[0] * 100:.2f}%)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title=f'PC2 ({explained_variance[1] * 100:.2f}%)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='black',
                showline=True,
                linecolor='black',
                gridcolor='lightgrey'
            ),
            font=dict(family='Times New Roman', size=14),
            plot_bgcolor='white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
        # Add vertical and horizontal lines at 0
        fig.add_shape(type="line", x0=0, x1=0, y0=df_pca.iloc[:, 1].min(), y1=df_pca.iloc[:, 1].max(),
                      line=dict(color="black", width=2), row = list(data.keys()).index(key) + 1, col = 1)
        fig.add_shape(type="line", x0=df_pca.iloc[:, 0].min(), x1=df_pca.iloc[:, 0].max(), y0=0, y1=0,
                      line=dict(color="black", width=2), row = list(data.keys()).index(key) + 1, col = 1)

        result[key] = Z

    fig.show()

    return result, fig