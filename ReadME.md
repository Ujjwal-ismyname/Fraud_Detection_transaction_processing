# Real-Time Fraud Detection System

## 📌 Project Overview
This project is a **Real-Time Fraud Detection System** designed to identify suspicious banking transactions using streaming analytics. Built with **LightGBM** for high-efficiency classification and **Flask** for web deployment, the system allows users to upload transaction datasets and receive immediate risk assessments, fraud probability scores, and visual analytics.

## 🏗️ Project Architecture

The system follows a modular machine learning pipeline:

```mermaid
graph TD
    A[Raw Data (CSV)] -->|Ingest| B(preprocess.py)
    B -->|Clean & Encode| C{Mode}
    C -->|Training| D[train.py]
    D -->|Train LightGBM| E[model/fraud_lgbm.pkl]
    D -->|Generate Plots| F[static/plots/]
    
    C -->|Inference| G[app.py]
    H[User / Browser] -->|Upload CSV| G
    G -->|Load Model| E
    G -->|Predict| I[Risk Scores & Visuals]
    I -->|Render| H