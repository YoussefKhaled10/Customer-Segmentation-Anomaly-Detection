# 🛍️ Customer Segmentation & Anomaly Detection System

An end-to-end **unsupervised for deployment and production-style usage.An end-to-end **unsupervised machine learning project** for analyzing customer behavior in an online retail business using real transactional data.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- KMeans
- DBSCAN
- Hierarchical Clustering
- Isolation Forest
- FastAPI
- Gradio
- Joblib
- Matplotlib
- Seaborn

---

## Project Workflow

### 1. Data Cleaning

The raw transactional data was cleaned by handling several real-world data issues:

- Removed transactions with missing `Customer ID`.
- Removed returned items where `Quantity <= 0`.
- Removed invalid transactions where `Price <= 0`.
- Cleaned non-numeric product stock codes.
- Calculated total purchase value for each transaction.

---

### 2. RFM Feature Engineering

The dataset was transformed from transaction-level data into customer-level data.

Final customer features:

- **Recency**: Number of days since the customer's last purchase.
- **Frequency**: Number of purchases made by the customer.
- **Monetary**: Total amount spent by the customer.
- **AvgOrderValue**: Average value per order.

---

### 3. Customer Segmentation Using KMeans

KMeans clustering was applied to the RFM features after applying log transformation to monetary-based features.

The number of clusters was selected as:

text
k = 4

The project builds RFM-based customer features, segments customers using KMeans clustering, detects unusual customer behavior within each segment using Isolation Forest, and deploys the final system using FastAPI and Gradio.

---

## Project Overview

The goal of this project is to transform raw transactional data into a practical machine learning system that helps businesses:

- Segment customers into meaningful groups.
- Understand customer behavior patterns.
- Detect unusual or unexpected customer behavior.
- Convert model outputs into actionable business decisions such as:
  - Reviewing high-value customers.
  - Monitoring regular customers.
  - Identifying growth opportunities among occasional buyers.
  - Ignoring new or inactive customers when appropriate.

---

## Project Objectives

- Clean and preprocess real-world retail transaction data.
- Build customer-level RFM features.
- Apply KMeans clustering for customer segmentation.
- Compare clustering behavior using DBSCAN and Hierarchical Clustering.
- Train segment-specific Isolation Forest models for anomaly detection.
- Build a FastAPI backend for model inference.
- Build a Gradio UI for user interaction.


---
## Author
Youssef Khaled

Machine Learning Engineer
