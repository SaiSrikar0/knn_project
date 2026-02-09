#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings('ignore')

#logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

#session state initialization
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None

#folder setup
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, "data", "raw")
clean_dir = os.path.join(base_dir, "data", "cleaned")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

log("application started")
log(f"raw_dir = {raw_dir}")
log(f"clean_dir = {clean_dir}")

#page config
st.set_page_config("End-to-End KNN", layout="wide")
st.title("End-to-End K-Nearest Neighbors (KNN) Application")

#sidebar: model settings
st.sidebar.header("KNN Settings")
problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 50, 5)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
metric = st.sidebar.selectbox("Distance Metric", ["minkowski", "euclidean", "manhattan", "chebyshev"])

if metric == "minkowski":
    p = st.sidebar.slider("p (for Minkowski)", 1, 5, 2)
else:
    p = 2

use_scaling = st.sidebar.checkbox("Use Feature Scaling", value=True)
use_grid_search = st.sidebar.checkbox("Use Grid Search CV")

log(f"KNN settings - problem_type: {problem_type}, n_neighbors: {n_neighbors}, weights: {weights}")

#step 1: Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1: Data Ingestion started")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])
df = None
raw_path = None

if option == "Download Dataset":
    dataset_choice = st.selectbox(
        "Select Dataset",
        ["Iris (Classification)", "Diabetes (Regression)", "Wine (Classification)"]
    )
    
    if st.button("Download Dataset"):
        log(f"Downloading {dataset_choice} dataset")
        
        if dataset_choice == "Iris (Classification)":
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            filename = "iris.csv"
        elif dataset_choice == "Diabetes (Regression)":
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/exercise.csv"
            filename = "exercise.csv"
        else:  # Wine
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
            filename = "penguins.csv"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            raw_path = os.path.join(raw_dir, filename)
            with open(raw_path, "wb") as f:
                f.write(response.content)

            df = pd.read_csv(raw_path)
            st.success(f"{dataset_choice} Dataset Downloaded successfully")
            log(f"{dataset_choice} dataset saved at {raw_path}")
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            log(f"Error downloading dataset: {e}")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(raw_dir, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded file saved at {raw_path}")

#step 2: EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2: EDA started")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape:**", df.shape)
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    with col2:
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Duplicate Rows:**", df.duplicated().sum())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap")

    # Distribution plots
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_features = st.multiselect(
            "Select features to visualize",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))]
        )
        
        if selected_features:
            n_cols = 2
            n_rows = (len(selected_features) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for idx, col in enumerate(selected_features):
                if idx < len(axes):
                    axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    axes[idx].set_title(f'Distribution of {col}')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
            
            # Hide extra subplots
            for idx in range(len(selected_features), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

    log("EDA completed")

#step 3: Data Cleaning
if df is not None:
    st.header("Step 3: Data Cleaning")
    log("Step 3: Data Cleaning started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Missing Value Handling Strategy",
            ["Mean", "Median", "Drop Rows"]
        )
    
    with col2:
        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
    
    df_clean = df.copy()
    
    # Handle missing values
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
        log("Dropped rows with missing values")
    else:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().sum() > 0:
                if strategy == "Mean":
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    log(f"Filled missing values in {col} with mean")
                elif strategy == "Median":
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    log(f"Filled missing values in {col} with median")
    
    # Remove duplicates
    if remove_duplicates:
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed = initial_len - len(df_clean)
        if removed > 0:
            st.info(f"Removed {removed} duplicate rows")
            log(f"Removed {removed} duplicate rows")
    
    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cleaned Data Preview:**")
        st.dataframe(df_clean.head())
    with col2:
        st.write("**Cleaning Summary:**")
        st.write(f"Original rows: {len(df)}")
        st.write(f"Cleaned rows: {len(df_clean)}")
        st.write(f"Missing values: {df_clean.isnull().sum().sum()}")
    
    log("Data cleaning completed")
else:
    st.info("Please complete Step 1 to proceed.")

#step 4: Save cleaned dataset
st.header("Step 4: Save Cleaned Dataset")
if st.button("Save Cleaned Dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data to save. Please complete Step 3.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_data_{timestamp}.csv"
        clean_path = os.path.join(clean_dir, clean_filename)

        st.session_state.df_clean.to_csv(clean_path, index=False)
        st.success("Cleaned dataset saved successfully")
        st.info(f"Cleaned dataset saved at {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")
        st.session_state.cleaned_saved = True

#step 5: Load cleaned dataset
st.header("Step 5: Load Cleaned Dataset")
clean_files = os.listdir(clean_dir)
if not clean_files:
    st.warning("No cleaned datasets found. Please save one in Step 4")
    log("No cleaned datasets found")
    df_model = None
else:
    selected = st.selectbox("Select cleaned dataset", clean_files)
    df_model = pd.read_csv(os.path.join(clean_dir, selected))
    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")
    
    st.dataframe(df_model.head())

#step 6: Train KNN Model
if df_model is not None:
    st.header(f"Step 6: Train KNN {problem_type} Model")
    log(f"Step 6: Train KNN {problem_type} started")

    target = st.selectbox("Select target variable", df_model.columns)
    
    if st.button("Train Model"):
        y = df_model[target].copy()
        
        # Check if it's classification or regression based on target
        is_classification = y.dtype == 'object' or y.nunique() < 20
        
        if problem_type == "Classification":
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                log("Target column encoded")
                st.info(f"Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            elif not is_classification:
                st.warning("Target appears to be continuous. Consider using Regression mode.")

        # Select numerical features only
        x = df_model.drop(columns=[target])
        x = x.select_dtypes(include=[np.number])
        
        if x.empty:
            st.error("No numerical features available for training.")
            st.stop()

        log(f"Features selected: {list(x.columns)}")
        st.write(f"**Features used:** {list(x.columns)}")
        st.write(f"**Number of samples:** {len(x)}")
        st.write(f"**Number of features:** {len(x.columns)}")

        # Train-test split
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )
        log(f"Train size: {len(x_train)}, Test size: {len(x_test)}")

        # Feature Scaling
        if use_scaling:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            st.info("Features scaled using StandardScaler")
            log("Features scaled")

        if use_grid_search:
            st.info("Using Grid Search CV for hyperparameter tuning...")
            log("Using Grid Search CV")
            
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'metric': ['minkowski', 'euclidean', 'manhattan']
            }
            
            if problem_type == "Classification":
                base_model = KNeighborsClassifier()
                scoring = 'accuracy'
            else:
                base_model = KNeighborsRegressor()
                scoring = 'neg_mean_squared_error'
            
            grid = GridSearchCV(
                base_model, 
                param_grid=param_grid, 
                scoring=scoring,
                cv=5,
                n_jobs=-1
            )
            
            with st.spinner("Training with Grid Search... This may take a while..."):
                grid.fit(x_train, y_train)
            
            st.success("Grid Search completed!")
            st.write("**Best Parameters:**")
            st.json(grid.best_params_)
            st.write(f"**Best Cross-Validation Score:** {grid.best_score_:.4f}")
            
            model = grid.best_estimator_
            log(f"Best params: {grid.best_params_}")
            log(f"Best CV score: {grid.best_score_:.4f}")
        else:
            # Model initialization with user settings
            if problem_type == "Classification":
                model = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    metric=metric,
                    p=p
                )
            else:
                model = KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    algorithm=algorithm,
                    metric=metric,
                    p=p
                )
            
            model.fit(x_train, y_train)
            st.success(f"KNN {problem_type} model trained successfully")
            log(f"KNN {problem_type} model trained successfully")

        # Predictions
        y_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)

        if problem_type == "Classification":
            # Classification Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_pred)
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_acc:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{test_acc:.4f}")
            
            # Check for overfitting
            if train_acc - test_acc > 0.1:
                st.warning("⚠️ Potential overfitting detected! Consider increasing k or using regularization.")
            
            log(f"Train Accuracy: {train_acc:.4f}")
            log(f"Test Accuracy: {test_acc:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            log("Confusion matrix displayed")

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            log("Classification report displayed")

        else:
            # Regression Metrics
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test R² Score", f"{test_r2:.4f}")
                st.metric("Train R² Score", f"{train_r2:.4f}")
            with col2:
                st.metric("Test RMSE", f"{test_rmse:.4f}")
                st.metric("Train RMSE", f"{train_rmse:.4f}")
            with col3:
                st.metric("Test MAE", f"{test_mae:.4f}")
                st.metric("Train MAE", f"{train_mae:.4f}")
            
            log(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

            # Prediction vs Actual Plot
            st.subheader("Predictions vs Actual Values")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Predicted vs Actual')
            ax1.grid(True, alpha=0.3)
            
            # Residual plot
            residuals = y_test - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Predicted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            log("Regression plots displayed")

        # Cross-validation scores
        st.subheader("Cross-Validation Analysis")
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, 
                                    scoring='accuracy' if problem_type == "Classification" else 'r2')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
        with col2:
            st.metric("Std CV Score", f"{cv_scores.std():.4f}")
        with col3:
            st.metric("Min/Max", f"{cv_scores.min():.4f} / {cv_scores.max():.4f}")
        
        # Plot CV scores
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, 6), cv_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        log(f"CV scores: {cv_scores}")

        # K-value analysis
        if not use_grid_search:
            st.subheader("K-Value Analysis")
            st.info("Testing different values of k to find optimal performance...")
            
            k_range = range(1, min(51, len(x_train)))
            k_scores = []
            
            for k in k_range:
                if problem_type == "Classification":
                    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm)
                else:
                    knn = KNeighborsRegressor(n_neighbors=k, weights=weights, algorithm=algorithm)
                
                knn.fit(x_train, y_train)
                score = knn.score(x_test, y_test)
                k_scores.append(score)
            
            # Plot k vs score
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(k_range, k_scores, marker='o', linestyle='-', linewidth=2)
            ax.axvline(x=n_neighbors, color='r', linestyle='--', label=f'Current k={n_neighbors}')
            ax.set_xlabel('Number of Neighbors (k)')
            ax.set_ylabel('Accuracy' if problem_type == "Classification" else 'R² Score')
            ax.set_title('Model Performance vs K-Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            optimal_k = k_range[np.argmax(k_scores)]
            st.write(f"**Optimal k-value:** {optimal_k} (Score: {max(k_scores):.4f})")
            log(f"Optimal k: {optimal_k} with score {max(k_scores):.4f}")

        # Model Information
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Model Type:** {type(model).__name__}")
            st.write(f"**Number of Neighbors (k):** {model.n_neighbors}")
            st.write(f"**Weights:** {model.weights}")
        with col2:
            st.write(f"**Algorithm:** {model.algorithm}")
            st.write(f"**Distance Metric:** {model.metric}")
            st.write(f"**Number of Training Samples:** {len(x_train)}")

        log("Model training and evaluation completed")

st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit 🎈")
log("Application running")
