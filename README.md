# 🛡️ Online Consumer Complaint Classification System

> **AI-powered web application for automated consumer complaint classification using NLP & Machine Learning**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Cloud Deployment](#cloud-deployment)
- [Usage](#usage)
- [Web Data Mining Concepts](#web-data-mining-concepts)
- [License](#license)
- [Author](#author)

---

## 🔭 Overview

The **Online Consumer Complaint Classification System** is an end-to-end machine learning web application that automatically classifies consumer financial complaints into predefined product categories. Built as a college mini-project for the **Web Data Mining** lab, this system leverages state-of-the-art Natural Language Processing (NLP) techniques and classical machine learning algorithms to process and categorize real-world complaint data sourced from the **Consumer Financial Protection Bureau (CFPB)**.

### Who Is It For?

| Audience | Use Case |
|----------|----------|
| **Consumers** | Understand how their complaints are categorized |
| **Financial Regulators** | Monitor complaint trends across financial products |
| **Financial Institutions** | Quickly route incoming complaints to the right department |
| **Researchers / Students** | Study NLP, text classification, and consumer behaviour patterns |
| **Data Analysts** | Explore geographic and temporal complaint trends |

The system accepts raw, unstructured complaint text as input, preprocesses it using an NLP pipeline, transforms it using TF-IDF vectorization, and predicts the financial product category — all within seconds, through a clean and interactive Streamlit dashboard.

---

## ✨ Features

- 🤖 **Automated Classification** — Instantly classifies complaint narratives into 8 financial product categories using trained ML models
- 📊 **Interactive Analytics Dashboard** — Visualize complaint distributions, monthly trends, company-wise analysis, and geographic heatmaps
- 🗺️ **US State Choropleth Map** — Interactive map showing complaint density by state across the United States
- 📈 **Model Performance Comparison** — Side-by-side comparison of Logistic Regression vs Naive Bayes with confusion matrices and classification reports
- 🔍 **Dataset Explorer** — Browse, filter, and search the raw complaint dataset with pagination
- 🧹 **NLP Preprocessing Pipeline** — Automated text cleaning including URL removal, stopword filtering, and tokenization
- ☁️ **Cloud-Ready** — Deployable on Streamlit Community Cloud with zero infrastructure management
- 📋 **Confidence Scores** — View per-category prediction probabilities for every classified complaint

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Interactive web application framework |
| **scikit-learn** | Machine learning — TF-IDF, Logistic Regression, Naive Bayes |
| **NLTK** | Natural Language Processing — tokenization, stopwords |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Plotly** | Interactive charts and visualizations |
| **Joblib** | Model serialization and persistence |
| **WordCloud** | Word frequency visualization |
| **Matplotlib / Seaborn** | Supporting static visualizations |

---

## 📂 Dataset

### Source
**Consumer Financial Protection Bureau (CFPB)** — the official U.S. government agency that collects consumer complaints about financial products.

🔗 **Download Link:** [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)

### Dataset Details

| Property | Value |
|----------|-------|
| **Provider** | Consumer Financial Protection Bureau (CFPB) |
| **Format** | CSV |
| **Size** | ~3–5 GB (full dataset, 3M+ records) |
| **Update Frequency** | Daily |
| **License** | Public Domain (U.S. Government Open Data) |

### Columns Used

| Column Name | Description | Role |
|-------------|-------------|------|
| `Product` | Financial product category (e.g., Mortgage, Credit card) | **Target / Label** |
| `Consumer complaint narrative` | Free-text complaint written by consumer | **Feature (input text)** |
| `Date received` | Date the complaint was submitted | Analytics |
| `Company` | Financial company the complaint is about | Analytics |
| `State` | US state of the complainant | Analytics |
| `Company response to consumer` | How the company responded | Analytics |

### Target Classes (8 Categories)

1. Mortgage
2. Debt collection
3. Credit reporting, credit repair services, or other personal consumer reports
4. Credit card or prepaid card
5. Student loan
6. Checking or savings account
7. Vehicle loan or lease
8. Money transfer, virtual currency, or money service

### Download Instructions

1. Visit [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
2. Click the **"Download all complaint data"** button
3. Select **CSV** as the download format
4. Wait for the download to complete (the file is large — expect several minutes)
5. Rename the downloaded file to **`complaints.csv`**
6. Place it inside the `data/` folder of this project: `online-consumer-complaint-classifier/data/complaints.csv`

> **💡 Tip for Deployment:** The full dataset is very large. For local development and cloud deployment, use only the first 50,000 rows:
> ```python
> df = pd.read_csv('data/complaints.csv', nrows=50000)
> ```

---

## 🗂️ Project Structure

```
online-consumer-complaint-classifier/
│
├── app.py                    # Main Streamlit web application
├── train_model.py            # ML model training script
├── preprocessing.py          # NLP text preprocessing module
├── analytics.py              # Analytics and visualization functions
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)
├── viva_preparation.md       # Viva Q&A and exam preparation guide
│
├── data/
│   └── complaints.csv        # CFPB dataset (download separately)
│
└── model/
    ├── trained_model.joblib      # Saved best ML model
    ├── tfidf_vectorizer.joblib   # Saved TF-IDF vectorizer
    └── model_metadata.joblib     # Model info, metrics, label classes
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git (optional, for cloning)

### Step-by-Step Setup

#### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/online-consumer-complaint-classifier.git
cd online-consumer-complaint-classifier
```

> Or download and extract the ZIP file from GitHub.

#### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

#### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4 — Download the CFPB Dataset

1. Visit [https://www.consumerfinance.gov/data-research/consumer-complaints/](https://www.consumerfinance.gov/data-research/consumer-complaints/)
2. Download the full CSV dataset
3. Create the `data/` directory:
   ```bash
   mkdir data
   ```
4. Save the file as `data/complaints.csv`

#### Step 5 — Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train Logistic Regression and Naive Bayes models
- Evaluate both on the test set
- Save the best model and vectorizer to the `model/` directory

Expected output:
```
Loading data...
Preprocessing text... (this may take a few minutes)
Training Logistic Regression...
Training Naive Bayes...
Best model: Logistic Regression (F1: 0.87)
Models saved to model/
```

#### Step 6 — Launch the Web Application

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

---

## ☁️ Cloud Deployment

Deploy this project for free on **Streamlit Community Cloud** so anyone with the URL can access it — no server management needed.

### STEP 1 — Create a GitHub Account
Go to [https://github.com](https://github.com) and sign up for a free account (if you don't have one already).

### STEP 2 — Create a New Repository
1. Click the **"+"** icon → **"New repository"**
2. Name it: `online-consumer-complaint-classifier`
3. Set visibility to **Public**
4. Click **"Create repository"**

### STEP 3 — Upload Project Files
1. Open your new repository on GitHub
2. Click **"uploading an existing file"** (or drag and drop files)
3. Upload all project files: `app.py`, `train_model.py`, `preprocessing.py`, `analytics.py`, `requirements.txt`
4. Upload the `model/` folder with all `.joblib` files (train locally first)
5. **Do NOT upload `data/complaints.csv`** — it's too large for GitHub (100 MB limit per file)
6. Click **"Commit changes"**

> ⚠️ **Important:** Since `complaints.csv` cannot be uploaded to GitHub (file too large), the Analytics and Dataset Explorer pages will not work in cloud deployment unless you use a sample. Create a `data/sample_complaints.csv` with the first 50,000 rows and upload that instead:
> ```python
> import pandas as pd
> df = pd.read_csv('data/complaints.csv', nrows=50000)
> df.to_csv('data/sample_complaints.csv', index=False)
> ```

### STEP 4 — Go to Streamlit Community Cloud
Visit [https://share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

### STEP 5 — Create a New App
Click the **"New app"** button.

### STEP 6 — Configure the App
- **Repository:** Select `YOUR_USERNAME/online-consumer-complaint-classifier`
- **Branch:** `main`
- **Main file path:** `app.py`

### STEP 7 — Click Deploy
Click the **"Deploy!"** button.

### STEP 8 — Wait and Copy Your URL
Wait 2–5 minutes for the build to complete. Once deployed, you'll get a public URL like:
```
https://your-username-online-consumer-complaint-classifier-app-xxxx.streamlit.app
```

Share this URL with anyone — it's your live web application!

> 📌 **NLTK Data on Cloud:** Add the following at the top of `app.py` to ensure NLTK data is available on Streamlit Cloud:
> ```python
> import nltk
> nltk.download('stopwords', quiet=True)
> nltk.download('punkt', quiet=True)
> ```

---

## 💻 Usage

### 1. Classify a Complaint
- Navigate to **🔍 Complaint Classification** in the sidebar
- Paste or type a consumer complaint narrative
- Click **"Classify Complaint"**
- View the predicted product category and confidence scores

### 2. Explore Analytics
- Navigate to **📊 Complaint Analytics**
- Explore interactive charts: category distribution, monthly trends, company analysis, US state map

### 3. Browse the Dataset
- Navigate to **📋 Dataset Explorer**
- Use the search bar and filters to explore raw complaint data

### 4. Compare Model Performance
- Navigate to **📈 Model Performance**
- View accuracy, precision, recall, F1-score for both models
- Examine confusion matrices side by side

> 📸 **Screenshots**
>
> *(Add screenshots of the application here after deployment)*
>
> | Home Dashboard | Complaint Classifier |
> |---|---|
> | `[Screenshot 1]` | `[Screenshot 2]` |
>
> | Analytics Dashboard | Model Performance |
> |---|---|
> | `[Screenshot 3]` | `[Screenshot 4]` |

---

## 🌐 Web Data Mining Concepts Demonstrated

This project is designed as a Web Data Mining lab exercise and demonstrates the following core concepts:

1. **Web Data Collection** — Utilizing publicly available government web data (CFPB open API / CSV download) as the data source
2. **Text Mining** — Extracting meaningful patterns and insights from unstructured consumer complaint narratives
3. **Data Preprocessing** — Cleaning raw text: lowercasing, punctuation removal, URL stripping, digit removal, and whitespace normalization
4. **Stop-Word Removal** — Filtering out high-frequency, low-information words (e.g., "the", "is", "and") using NLTK
5. **Tokenization** — Breaking complaint text into individual word tokens for further processing
6. **Feature Extraction (TF-IDF)** — Converting text into numerical feature vectors using Term Frequency–Inverse Document Frequency
7. **N-gram Analysis** — Capturing bigrams (two-word combinations) alongside unigrams for richer feature representation
8. **Multi-Class Text Classification** — Applying supervised ML algorithms (Logistic Regression, Naive Bayes) to classify text into 8 categories
9. **Model Evaluation** — Assessing model quality using accuracy, precision, recall, F1-score, and confusion matrices
10. **Data Visualization** — Communicating data insights through interactive Plotly charts, choropleth maps, and heatmaps
11. **Pattern Identification** — Identifying complaint trends by time, geography, company, and product category
12. **Web Application Deployment** — Packaging the entire pipeline into an interactive browser-based application

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👤 Author

**College Mini-Project — Web Data Mining Lab**

| Field | Detail |
|-------|--------|
| **Project Title** | Online Consumer Complaint Classification System |
| **Subject** | Web Data Mining |
| **Dataset** | CFPB Consumer Complaints (Public Domain) |
| **GitHub** | [https://github.com/YOUR_USERNAME/online-consumer-complaint-classifier](https://github.com/YOUR_USERNAME/online-consumer-complaint-classifier) |

> *Built with ❤️ using Python, Streamlit, scikit-learn, and NLTK*

---

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Powered%20by-Streamlit-FF4B4B?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/Data-CFPB%20Open%20Data-green?style=flat-square" />
</p>
