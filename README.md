# Federated Learning for Postpartum Depression Screening

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Flower](https://img.shields.io/badge/backend-Flower-orange)
![XGBoost](https://img.shields.io/badge/model-XGBoost-green)

## 📌 Project Overview
This repository contains the implementation of a **Federated Learning (FL) framework** designed to screen for Postpartum Depression (PPD) in a privacy-preserving manner. 

Using **Federated XGBoost (Horizontal FL)**, this system allows multiple medical institutions (simulated clients) to collaboratively train a robust screening model without sharing sensitive patient data. The system features:
- **Privacy Preservation:** Patient data never leaves the local client.
- **Imbalance Handling:** Automated class weighting and binary mapping (Healthy vs. At-Risk).
- **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) analysis to identify key risk factors like Anxiety and Sleep Quality.

## 🏗️ Architecture
The system is built on the **Flower (flwr)** framework and utilizes a **Federated Bagging** strategy.

| Phase | Description |
| :--- | :--- |
| **I. Feature Engineering** | Automated cleaning, binary target mapping |
| **II. Partitioning** | Simulates heterogeneous data distribution across 4 virtual clients. |
| **III. Client Training** | Independent XGBoost training on edge nodes using Hist-based trees. |
| **IV. Server Aggregation** | Federated Bagging of trees, global evaluation, and XAI visualization. |

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YourUsername]/ppd-federated-learning.git
cd ppd-federated-learning
```
### 2. Install Dependencies
Install the required dependencies in a virtual environment.
```bash
pip install -r requirements.txt
```
### 3. Run Simulation
```bash
flwr run . --run-config "num-server-rounds=15"
```

## Author
[Sabah Ummie] 
