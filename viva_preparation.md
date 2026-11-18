# 📚 Viva Preparation Guide
## Online Consumer Complaint Classification System
### Web Data Mining Lab — College Mini-Project

---

> **How to use this guide:** Read each section carefully before your viva. Practice speaking the answers out loud. Focus especially on Sections 1, 6, 7, 10, and 12.

---

## 1. Complete Project Explanation (Simple English)

### What is this project?

This project is called the **Online Consumer Complaint Classification System**. It is a web application that uses **Artificial Intelligence (AI)** to automatically read a consumer's complaint text and tell us which financial product category the complaint belongs to.

For example, if a customer writes: *"My credit card company charged me extra fees without informing me"* — our system will automatically classify this as a **"Credit card or prepaid card"** complaint.

### Why is it needed?

Every year, millions of people submit complaints about financial companies to the government — complaints about mortgages, credit cards, loans, bank accounts, and more. Manually reading and categorizing each complaint would take thousands of human hours. Our AI system can classify a complaint **in under a second**, saving enormous time and effort.

The U.S. government body called the **Consumer Financial Protection Bureau (CFPB)** collects these complaints and makes the data publicly available. We use this real, government-collected data to train our system.

### How does it work — step by step?

**Step 1: Data Collection**
We download complaint data from the CFPB website. This data contains over 3 million consumer complaints, each with a free-text narrative (the customer's own words) and a product label (which financial product it's about).

**Step 2: Data Cleaning (Preprocessing)**
Raw complaint text is messy. It may contain URLs, numbers, special characters, unnecessary punctuation, and very common English words like "the", "is", "a" that carry no meaningful information. Our `preprocessing.py` module cleans the text by removing all of these.

**Step 3: Feature Extraction (TF-IDF)**
Computers cannot understand words directly — they understand numbers. So we convert each complaint text into a numerical vector using a technique called **TF-IDF (Term Frequency–Inverse Document Frequency)**. This gives higher weight to words that are unique and important to a specific complaint, and lower weight to very common words.

**Step 4: Model Training**
We feed these numerical vectors into two machine learning algorithms:
- **Logistic Regression** — a powerful linear classifier
- **Multinomial Naive Bayes** — a probabilistic classifier well-suited for text

Both models learn patterns from thousands of labeled complaints and can then predict the category of new, unseen complaints.

**Step 5: Model Evaluation**
We test both models on data they have never seen before and measure their accuracy, precision, recall, and F1-score. We automatically save the better-performing model.

**Step 6: Web Application**
We build a user-friendly web dashboard using **Streamlit** — a Python library for creating interactive web apps. The dashboard has 6 pages:
- A **Home** page with project overview
- A **Classifier** page where you type a complaint and get a prediction
- An **Analytics** page with charts and maps
- A **Dataset Explorer** to browse raw data
- A **Model Performance** page comparing both models
- An **About** page explaining the project

**Step 7: Deployment**
The application is deployed on **Streamlit Community Cloud**, making it accessible to anyone with the URL — no installation required.

### Who benefits from this project?

- **Financial regulators** can track which products generate the most complaints
- **Financial companies** can automatically route incoming complaints to the right department
- **Consumers** can understand how their complaint will be categorized
- **Researchers** can analyze trends in consumer financial grievances over time and across geographic regions

### What makes this project special?

This project is a complete, end-to-end pipeline — from raw government data, through NLP preprocessing, machine learning training, model evaluation, all the way to a live web application. It demonstrates every major concept taught in the Web Data Mining course in a single, working system.

---

## 2. File-by-File Explanation

### `app.py` — Main Web Application
This is the **entry point** of the application. It is a Streamlit script that creates the entire web dashboard. It imports the trained model and vectorizer using `joblib`, loads the dataset, and renders all 6 pages of the dashboard based on the user's selection in the sidebar. It calls functions from `analytics.py` for charts and uses the loaded model to make real-time predictions.

**Key responsibilities:**
- Page routing (sidebar navigation)
- Loading and caching model, vectorizer, and dataset
- Running the prediction pipeline on user-entered text
- Rendering all visualizations

### `train_model.py` — ML Training Script
This script is run **once** to train and save the machine learning model. It:
1. Loads `data/complaints.csv`
2. Drops rows with missing narratives or product labels
3. Keeps only the top 8 product categories (to avoid class imbalance)
4. Samples up to 2,000 rows per category
5. Applies `preprocess_pipeline()` from `preprocessing.py`
6. Fits a **TF-IDF vectorizer** (10,000 features, unigrams and bigrams)
7. Splits data into 80% training and 20% testing (stratified)
8. Trains Logistic Regression and Multinomial Naive Bayes
9. Evaluates both and selects the better model by F1-score
10. Saves the best model, vectorizer, and metadata to `model/`

### `preprocessing.py` — NLP Preprocessing Module
This module handles all text cleaning. It is used by both `train_model.py` (for training data) and `app.py` (for live input text before classification).

**Functions:**
- `clean_text(text)` — Converts to lowercase, removes URLs, removes special characters and digits, strips extra whitespace
- `remove_stopwords(text)` — Removes common English stopwords using NLTK's stopword list
- `preprocess_pipeline(text)` — Combines `clean_text` + `remove_stopwords` into one call
- `get_text_stats(df, text_col)` — Returns average text length, vocabulary size, and total word count for a DataFrame column

### `analytics.py` — Analytics & Visualization Module
This module contains functions that generate **Plotly interactive charts**. It is imported by `app.py` for the Analytics dashboard page.

**Functions:**
- `get_category_distribution(df)` — Bar chart of top 15 product categories
- `get_monthly_trend(df)` — Line chart of complaints per month over time
- `get_company_analysis(df)` — Horizontal bar chart of top N companies by complaints
- `get_status_distribution(df)` — Pie chart of company response types
- `get_complaint_heatmap(df)` — Heatmap of products vs companies (pivot table)
- `get_state_distribution(df)` — US choropleth map of complaints by state
- `get_yearly_trend(df)` — Bar chart of complaints per year

### `requirements.txt` — Python Dependencies
Lists all external Python packages needed to run the project. Used by `pip install -r requirements.txt` during setup. Key packages: `streamlit`, `pandas`, `scikit-learn`, `nltk`, `plotly`, `joblib`, `wordcloud`.

---

## 3. Dashboard Module Explanations

### Page 1: 🏠 Home Dashboard
The landing page of the application. Displays:
- Project title and description
- Key statistics: total complaints loaded, number of categories, model accuracy, number of features
- A quick overview chart (category distribution)
- Quick navigation cards to other sections

**Purpose:** Gives the user an immediate bird's-eye view of the system's capabilities and data.

### Page 2: 🔍 Complaint Classification
The core functionality of the application. Contains:
- A large text area for entering a complaint narrative
- A "Classify Complaint" button
- Output: predicted category displayed prominently
- A bar chart showing confidence (probability) for all 8 categories
- Sample complaints the user can try

**How it works:** The input text is passed through `preprocess_pipeline()`, then `tfidf_vectorizer.transform()`, and finally `model.predict()` and `model.predict_proba()`.

### Page 3: 📊 Complaint Analytics
An interactive analytics dashboard with multiple visualizations:
- Category distribution bar chart
- Monthly complaint trend line chart
- Top companies horizontal bar chart
- US state choropleth map
- Product vs Company heatmap
- Company response pie chart
- Yearly complaint bar chart

**Purpose:** Answers data-driven questions like "Which state has the most complaints?" or "Which company is most complained about?"

### Page 4: 📋 Dataset Explorer
A data browsing interface that allows users to:
- View the raw dataset in a paginated table
- Filter by product category, state, or company
- Search for keywords in complaint text
- Download filtered data as CSV

**Purpose:** Enables data exploration and verification without writing code.

### Page 5: 📈 Model Performance
A model comparison and evaluation page showing:
- Accuracy, Precision, Recall, and F1-score for both models in a table
- Confusion matrices visualized as heatmaps
- Classification reports per category
- Feature importance (top words contributing to each category for Logistic Regression)

**Purpose:** Demonstrates rigorous model evaluation and allows academic review of model quality.

### Page 6: ℹ️ About Project
An informational page containing:
- Project description and motivation
- Team/author information
- Dataset source and citation
- Tech stack overview
- Links to GitHub repository

**Purpose:** Provides context, attribution, and documentation for examiners.

---

## 4. Dataset Explanation

### What is CFPB?
The **Consumer Financial Protection Bureau (CFPB)** is a U.S. government agency created in 2011 by the Dodd-Frank Act. Its mission is to protect consumers from unfair, deceptive, or abusive financial practices. It acts as a watchdog for banks, credit card companies, lenders, and other financial institutions.

### What Data Does It Contain?
The CFPB maintains a public database of consumer complaints submitted about financial products and services. Each record in the database represents one complaint and includes:
- The consumer's narrative (their words describing the problem)
- The financial product involved
- The company being complained about
- The date the complaint was received
- The state the consumer lives in
- How the company responded to the complaint
- Whether the consumer provided consent to publish the narrative

The dataset contains **over 3 million complaints** dating back to 2011, updated daily.

### Which Columns Are Used and Why?

| Column | Why Used |
|--------|----------|
| `Consumer complaint narrative` | The actual text — this is what we classify |
| `Product` | The label — this is what we're trying to predict |
| `Date received` | For temporal trend analysis |
| `Company` | For company-level analysis |
| `State` | For geographic visualization |
| `Company response to consumer` | For response pattern analysis |

### How Is the Data Cleaned?
1. **Drop NaN rows:** Rows where either the narrative or the product label is missing are removed
2. **Category filtering:** Only the top 8 most common product categories are kept (others are dropped)
3. **Sampling:** A maximum of 2,000 rows per category are sampled to ensure balanced training (avoids one category dominating)
4. **Text cleaning:** The narrative text is lowercased, URLs removed, special characters stripped, digits removed, stopwords filtered out

### Why Did We Choose This Dataset?
- It is **real government data** — not synthetic or toy data
- It is **freely available** as open data (public domain)
- It is **large and diverse** — millions of records across many categories and companies
- It has **natural language text** — ideal for NLP tasks
- It is **well-labeled** — each complaint already has a verified product category
- It is directly relevant to **Web Data Mining** as it comes from a web-accessible API

---

## 5. NLP Explanation (for a Production Engineering Student)

### What is NLP?
**Natural Language Processing (NLP)** is a branch of Artificial Intelligence that deals with enabling computers to understand, interpret, and generate human language (like English). Just as a human reads and understands text, NLP techniques allow machines to do the same.

Think of it like teaching a calculator — which only knows numbers — how to understand a sentence.

### Why Do We Need Text Preprocessing?
Raw complaint text is **noisy and inconsistent**. Consider these problems:
- "CREDIT CARD" and "credit card" are the same thing but look different to a computer
- Punctuation like "!" or "..." adds no meaning to classification
- URLs like "http://company.com" in a complaint are irrelevant noise
- Words like "the", "a", "is", "and" appear in every document and carry no discriminating information
- Numbers like "1234567" are specific identifiers, not meaningful patterns

Preprocessing solves all of these problems, giving the ML model clean, consistent input.

### Tokenization — Explained Simply
**Tokenization** means breaking a sentence into individual words (called "tokens").

**Example:**
- Input: `"My credit card was charged without notice"`
- Tokens: `["My", "credit", "card", "was", "charged", "without", "notice"]`

This is the first step in analyzing text — you can't analyze a sentence as a single blob; you need to work with individual words.

### Stop-Word Removal — Explained Simply
**Stop words** are extremely common English words that appear in nearly every sentence but carry almost no meaningful information. Examples: *the, is, it, a, an, and, or, but, in, on, at, was, were, be, been*.

If we included these words in our model, they would appear in complaints about every product equally, making it impossible to distinguish between categories.

**Example:**
- Before: `["my", "credit", "card", "was", "charged", "without", "notice"]`
- After removing stopwords: `["credit", "card", "charged", "without", "notice"]`

The remaining words are much more informative for classification.

---

## 6. TF-IDF Explanation

### What Does TF-IDF Stand For?
**TF-IDF** stands for **Term Frequency–Inverse Document Frequency**. It is a numerical technique to represent text documents as vectors of numbers, giving higher scores to words that are important and distinctive.

### TF — Term Frequency
**TF** measures how often a word appears in a single document, normalized by the total number of words.

**Formula:**
```
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

**Example:**
- Document: `"credit card credit fraud credit report"` (6 words)
- TF("credit") = 3/6 = **0.5**
- TF("fraud") = 1/6 = **0.167**

### IDF — Inverse Document Frequency
**IDF** measures how rare a word is across all documents in the dataset. Words that appear in almost every document (like "the") get a very low IDF score.

**Formula:**
```
IDF(t) = log(N / df(t))
```
Where:
- `N` = total number of documents
- `df(t)` = number of documents that contain term `t`

**Example:**
- Total documents (N) = 1,000
- "credit" appears in 800 documents → IDF = log(1000/800) = log(1.25) ≈ **0.097**
- "fraud" appears in 50 documents → IDF = log(1000/50) = log(20) ≈ **1.301**

### TF-IDF = TF × IDF

**Example (continued):**

| Word | TF | IDF | TF-IDF |
|------|----|-----|--------|
| credit | 0.5 | 0.097 | **0.049** |
| fraud | 0.167 | 1.301 | **0.217** |

The word **"fraud"** gets a much higher TF-IDF score, correctly indicating it is more distinctive and informative for this document.

### Why Is TF-IDF Better Than Simple Word Count?

| Approach | Problem |
|----------|---------|
| **Word Count** | Common words like "the" get the highest counts and dominate the model |
| **TF-IDF** | Penalizes common words, rewards rare and distinctive words |

TF-IDF solves the "common word problem" automatically. It's why words like "mortgage", "foreclosure", and "escrow" will score highly in mortgage complaints but low in credit card complaints.

---

## 7. Logistic Regression Explanation

### What Is It?
Despite its name containing "regression", **Logistic Regression is a classification algorithm**. The "regression" in the name refers to the mathematical technique used internally — but the output is a category (class label), not a continuous number.

### How Does It Work?
Logistic Regression learns a set of **weights** (one per feature/word) that represent the importance of each word for predicting each category. 

For a given input, it computes a **score** for each class by multiplying the TF-IDF values by the weights and summing them. This score is then passed through the **sigmoid function** (or softmax for multi-class), which converts it to a probability between 0 and 1.

**Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```
- If z is very large → σ(z) ≈ 1 (high probability)
- If z is very small → σ(z) ≈ 0 (low probability)
- If z = 0 → σ(z) = 0.5

### Why Does It Work Well for Text?
- Text features (TF-IDF) are **linear in nature** — the importance of words adds up
- Logistic Regression is a **linear model** — it perfectly matches this structure
- It handles **high-dimensional data** (10,000 features) very efficiently
- It provides **probability estimates** (confidence scores), not just class labels
- It is **fast to train and predict**, even with large vocabularies

### Hyperparameters Used

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `C` | 5 | Regularization strength (higher C = less regularization, model fits training data more closely) |
| `max_iter` | 1000 | Maximum iterations for the optimization algorithm to converge |
| `random_state` | 42 | Ensures reproducibility of results |

---

## 8. Naive Bayes Explanation

### What Is It?
**Naive Bayes** is a **probabilistic classifier** based on Bayes' Theorem. Given a complaint text, it calculates the probability that the complaint belongs to each category and picks the category with the highest probability.

### Why "Naive"?
It is called "naive" because it makes a **simplifying assumption**: it assumes that all words in a complaint are **independent of each other**. In reality, words are related (e.g., "credit card" is a phrase where the words depend on each other), but this assumption makes the math much simpler and the algorithm much faster — and it still works surprisingly well in practice.

### Bayes' Theorem

```
P(Category | Text) = P(Text | Category) × P(Category) / P(Text)
```

In plain English:
- **P(Category | Text)** — What is the probability this complaint belongs to "Mortgage", given the words in it?
- **P(Text | Category)** — How likely are these specific words to appear in a "Mortgage" complaint?
- **P(Category)** — How common are "Mortgage" complaints overall?
- **P(Text)** — This is a constant normalizer (same for all categories, so it can be ignored)

### Why Multinomial NB for Text?
**Multinomial Naive Bayes** is designed specifically for data where features represent **counts or frequencies** — exactly what TF-IDF vectors represent. It models the distribution of word occurrences and is one of the best classical algorithms for document classification.

### Hyperparameter: Alpha (Laplace Smoothing)

| Hyperparameter | Value | Meaning |
|----------------|-------|---------|
| `alpha` | 0.1 | Laplace smoothing — prevents zero probabilities when a word was never seen during training for a specific category. A word unseen in training would give P=0, breaking the whole multiplication. Alpha adds a small count to avoid this. |

---

## 9. Evaluation Metrics Explanation

### Accuracy
**Definition:** The percentage of predictions that were correct.

**Formula:**
```
Accuracy = (Number of Correct Predictions) / (Total Predictions) × 100%
```

**Example:** 850 correct out of 1000 → Accuracy = 85%

**Limitation:** Misleading when classes are imbalanced. If 90% of complaints are "Mortgage", a model that always predicts "Mortgage" gets 90% accuracy but is useless.

---

### Precision
**Definition:** Of all the complaints we predicted as category X, how many actually were category X?

**Formula:**
```
Precision = True Positives / (True Positives + False Positives)
```

**Example:** We predicted "Mortgage" 100 times. 90 were actually Mortgage → Precision = 90%

**When it matters:** When false positives are costly (e.g., misrouting a complaint to the wrong department).

---

### Recall
**Definition:** Of all the actual category X complaints, how many did we correctly identify?

**Formula:**
```
Recall = True Positives / (True Positives + False Negatives)
```

**Example:** There were 100 real "Mortgage" complaints. We correctly caught 85 → Recall = 85%

**When it matters:** When false negatives are costly (e.g., missing a fraud complaint).

---

### F1-Score
**Definition:** The harmonic mean of Precision and Recall. Balances both metrics into one number.

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:** Precision = 90%, Recall = 85% → F1 = 2 × (0.90 × 0.85) / (0.90 + 0.85) = **0.874 (87.4%)**

**We use weighted F1** (averages F1 across all classes, weighted by class size) to select the best model.

---

### Confusion Matrix
**Definition:** A table that shows, for each actual category, how many predictions fell into each predicted category. The diagonal shows correct predictions; off-diagonal shows errors.

**Simple 2×2 Example (Binary: Mortgage vs Not-Mortgage):**

```
                  Predicted: Mortgage  |  Predicted: Not-Mortgage
Actual: Mortgage          85 (TP)      |          15 (FN)
Actual: Not-Mortgage      10 (FP)      |         890 (TN)
```

| Cell | Name | Meaning |
|------|------|---------|
| 85 | True Positive (TP) | Actually Mortgage, predicted Mortgage ✅ |
| 890 | True Negative (TN) | Not Mortgage, predicted Not-Mortgage ✅ |
| 10 | False Positive (FP) | Not Mortgage, but predicted Mortgage ❌ |
| 15 | False Negative (FN) | Actually Mortgage, but missed it ❌ |

In a **multi-class** confusion matrix (8×8), each row is an actual class and each column is a predicted class. A perfect model would have all numbers on the diagonal and zeros elsewhere.

---

## 10. Viva Questions & Answers

### 🎯 Project Purpose and Overview

**Q1. What is the main objective of your project?**
> The main objective is to automatically classify consumer financial complaint narratives into predefined product categories using NLP and machine learning. Instead of manually reading complaints, our system reads the complaint text and instantly identifies which financial product — mortgage, credit card, student loan, etc. — the complaint is about.

**Q2. Why did you choose this particular project?**
> This project covers all the core concepts of Web Data Mining in a single application: web data collection, text mining, NLP preprocessing, feature extraction with TF-IDF, multi-class classification, model evaluation, and interactive data visualization. It uses real, publicly available government data, making it practical and meaningful.

**Q3. What is the real-world application of this system?**
> Financial companies receive thousands of complaints daily. Our system can automatically route each complaint to the correct department based on its product category — saving human effort, reducing routing errors, and speeding up response times. Government regulators can also use it to automatically analyze complaint patterns and detect emerging issues.

**Q4. How many product categories does your system classify?**
> Our system classifies complaints into 8 categories: Mortgage, Debt collection, Credit reporting, Credit card or prepaid card, Student loan, Checking or savings account, Vehicle loan or lease, and Money transfer or virtual currency services.

**Q5. What is the overall workflow of your project?**
> The workflow is: (1) Download CFPB complaint data → (2) Clean and preprocess the text using NLP → (3) Convert text to TF-IDF vectors → (4) Train Logistic Regression and Naive Bayes models → (5) Evaluate both models → (6) Save the best model → (7) Build a Streamlit web app that uses the saved model for real-time predictions and analytics.

---

### 📂 Dataset Questions

**Q6. What is the CFPB?**
> CFPB stands for Consumer Financial Protection Bureau. It is a U.S. government agency that protects consumers from unfair financial practices. It collects and publicly publishes consumer complaints about financial products, making the dataset freely available for research and analysis.

**Q7. How large is the CFPB dataset?**
> The full CFPB dataset contains over 3 million complaints dating back to 2011, and it is updated daily. The raw CSV file is approximately 3–5 GB. For our project, we sample up to 2,000 rows per category (maximum 16,000 rows) for efficient training and cloud deployment.

**Q8. Why do you sample only 2000 rows per category?**
> We sample to: (1) prevent class imbalance — some categories have far more complaints than others; (2) reduce training time; (3) make the model deployable in cloud environments with memory constraints. Training on 16,000 balanced samples still gives excellent model performance.

**Q9. What preprocessing do you apply to the dataset?**
> We drop rows where the complaint narrative or product label is null, keep only the top 8 product categories, sample up to 2,000 rows per category, then apply the full NLP preprocessing pipeline: lowercase conversion, URL removal, special character and digit removal, whitespace normalization, and stopword removal.

**Q10. Why did you choose this dataset specifically?**
> It is real government data (not synthetic), freely available as open data, contains rich natural language text with verified labels, is diverse in categories and companies, and directly relevant to Web Data Mining since it comes from a web-accessible government portal with a public API.

---

### 🔤 NLP and Preprocessing

**Q11. What is NLP?**
> NLP stands for Natural Language Processing. It is a branch of AI that enables computers to understand, process, and generate human language. In our project, NLP techniques allow the computer to process raw complaint text and extract meaningful features for classification.

**Q12. Why do we need to preprocess text before classification?**
> Raw text is inconsistent and contains noise — uppercase/lowercase variations, punctuation, URLs, numbers, and very common words like "the" or "is" that appear everywhere. Without preprocessing, the model would learn meaningless patterns from this noise instead of the actual content. Preprocessing gives the model clean, consistent input, improving accuracy significantly.

**Q13. What does your clean_text() function do?**
> `clean_text()` performs these steps in order: (1) converts text to lowercase, (2) removes URLs using regex, (3) removes all characters except letters and spaces using regex, (4) removes standalone single letters, (5) removes extra whitespace. The result is a clean, lowercase string of meaningful words.

**Q14. What are stop words? Give examples.**
> Stop words are extremely common English words that carry minimal meaning in the context of text classification. Examples include: "the", "is", "it", "a", "an", "and", "or", "but", "in", "on", "at", "to", "was", "were", "has", "have", "be", "been". We remove them because they appear equally in complaints about all product categories and would confuse the model.

**Q15. What NLTK functions do you use?**
> We use `nltk.corpus.stopwords.words('english')` to get the list of English stop words, and `nltk.download('stopwords', quiet=True)` and `nltk.download('punkt', quiet=True)` to download the required NLTK data files. The downloaded stopword list contains 179 common English words.

---

### 📊 TF-IDF Questions

**Q16. What does TF-IDF stand for and what does it do?**
> TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text documents into numerical vectors by assigning a weight to each word. Words that appear frequently in a specific document but rarely across all documents get high weights. Words that appear everywhere get low weights. This representation captures the most discriminative words in each complaint.

**Q17. What is the difference between TF and IDF?**
> TF (Term Frequency) measures how often a word appears in a single document. IDF (Inverse Document Frequency) measures how rare that word is across all documents. TF rewards frequently used words; IDF penalizes words that appear in too many documents. Together, TF-IDF rewards words that are both frequent in a document AND rare across the corpus.

**Q18. What parameters did you use for TF-IDF vectorizer?**
> We used: `max_features=10000` (keep only the 10,000 most important words to control dimensionality), `ngram_range=(1,2)` (use both single words and two-word pairs to capture phrases), and `sublinear_tf=True` (apply log normalization to term frequency to reduce the impact of extremely common words).

**Q19. Why use ngram_range=(1,2)?**
> Including bigrams (two-word combinations) captures meaningful phrases that single words miss. For example, "credit card" as a bigram is much more informative than "credit" and "card" as separate words. Without bigrams, "student loan" and "home loan" would share the word "loan" and be harder to distinguish. Bigrams greatly improve classification accuracy for financial text.

---

### 🤖 Machine Learning Models

**Q20. Why did you choose Logistic Regression for text classification?**
> Logistic Regression works extremely well for high-dimensional, sparse text data (TF-IDF vectors). It is linear, fast, interpretable (you can see which words most influence each prediction), and provides probability estimates. Research consistently shows it performs among the best classical algorithms for text classification tasks.

**Q21. Why did you choose Naive Bayes?**
> Multinomial Naive Bayes is specifically designed for text classification with count-based features. It is very fast to train, works well with limited data, handles high-dimensional feature spaces efficiently, and has a solid theoretical basis for text. It is a natural baseline for any text classification problem.

**Q22. How do you choose between the two models?**
> We evaluate both models on the same held-out test set and compare their weighted F1-scores. The model with the higher weighted F1-score is selected and saved. F1-score is chosen over accuracy because our dataset, even after sampling, may have minor class imbalances.

**Q23. What is the train-test split ratio?**
> We use an 80/20 split with `stratified=True` and `random_state=42`. Stratified ensures each class is represented proportionally in both training and test sets. Random state 42 ensures the same split every time the code is run (reproducibility).

**Q24. What is regularization in Logistic Regression? (C parameter)**
> Regularization prevents overfitting by penalizing model complexity. The `C` parameter in scikit-learn is the **inverse** of regularization strength: higher C = less regularization (model can fit training data more closely); lower C = more regularization (simpler, more generalizable model). We use C=5, allowing moderate flexibility.

---

### 📈 Evaluation Metrics

**Q25. What is the difference between precision and recall?**
> Precision answers: "Of everything we labeled as X, how many were actually X?" (quality of positive predictions). Recall answers: "Of everything that was actually X, how many did we find?" (completeness of detection). There is typically a trade-off: increasing precision often decreases recall and vice versa. F1-score balances both.

**Q26. Why use F1-score instead of accuracy?**
> Accuracy can be misleading with imbalanced classes. If one category has 60% of all complaints, a model that always predicts that category gets 60% accuracy but is completely useless. F1-score considers both precision and recall for each class, giving a more honest picture of model performance, especially for multi-class problems.

**Q27. What does "weighted" F1-score mean?**
> Weighted F1-score calculates the F1-score for each class separately, then averages them, with each class weighted by its number of instances. So larger classes contribute more to the final score. This is appropriate for our problem because we want overall performance to reflect the natural distribution of complaint categories.

**Q28. How do you read a confusion matrix?**
> Each row represents the actual class; each column represents the predicted class. The diagonal (top-left to bottom-right) shows correct predictions. Off-diagonal cells show errors — specifically, what a complaint was wrongly classified as. A good model has large diagonal numbers and small off-diagonal numbers.

---

### 🌐 Streamlit and Web App

**Q29. What is Streamlit?**
> Streamlit is an open-source Python library that allows you to create interactive web applications using only Python code — no HTML, CSS, or JavaScript needed. You write Python scripts, and Streamlit automatically renders them as web apps with interactive widgets like buttons, sliders, text inputs, and charts.

**Q30. How does the user classify a complaint using your app?**
> The user navigates to the "🔍 Complaint Classification" page, types or pastes their complaint narrative into the text area, and clicks "Classify Complaint". The app runs the text through the preprocessing pipeline, converts it to a TF-IDF vector, and passes it through the trained model. The predicted category and confidence scores for all 8 categories are displayed.

**Q31. How does the app load the trained model?**
> The app uses `joblib.load()` to load three files from the `model/` directory: `trained_model.joblib` (the ML model), `tfidf_vectorizer.joblib` (the fitted TF-IDF vectorizer), and `model_metadata.joblib` (metrics, category names). These are loaded once and cached using `@st.cache_resource` to avoid reloading on every user interaction.

**Q32. What is `@st.cache_resource` and why do you use it?**
> `@st.cache_resource` is a Streamlit decorator that caches the return value of a function so it is computed only once per app session, not on every re-run. We use it for loading the model and dataset because these operations are expensive (reading large files from disk). Without caching, the app would reload the model every time the user clicks anything, making it very slow.

---

### 🌍 Web Data Mining Concepts

**Q33. How does your project relate to Web Data Mining?**
> Our project uses data sourced from the web (CFPB government portal), applies text mining techniques to unstructured web data (complaint narratives), performs all stages of a data mining pipeline (collection, preprocessing, feature extraction, classification, evaluation, visualization), and delivers results through a web application — covering the full spectrum of Web Data Mining.

**Q34. What is text mining?**
> Text mining is the process of extracting useful, structured information from unstructured text data. It involves techniques like tokenization, stopword removal, stemming, TF-IDF vectorization, and classification. Our project applies text mining to convert raw complaint narratives into classified, actionable information.

**Q35. What patterns does your project identify?**
> Our project identifies: (1) which financial products generate the most complaints, (2) which companies are most complained about, (3) how complaint volumes change over time (monthly/yearly trends), (4) which US states generate the most complaints, (5) how companies respond to different types of complaints, (6) which words and phrases are most strongly associated with each product category.

**Q36. How is feature extraction done in your project?**
> Feature extraction is done using TF-IDF vectorization. The raw complaint text, after preprocessing, is passed through a fitted `TfidfVectorizer` from scikit-learn. This converts each complaint into a numerical vector of 10,000 features, where each dimension corresponds to a word or bigram, and the value is its TF-IDF weight. These vectors are what the ML models actually learn from.

---

### 🛠️ Practical / Demo Questions

**Q37. What happens if someone enters a very short complaint?**
> Short complaints (e.g., one or two words) will still be classified, but confidence scores may be lower since there are fewer words for the model to work with. The model will still return the most probable category based on whatever words are present. We handle edge cases gracefully — the app shows a warning if the input is too short.

**Q38. Can the model classify complaints it has never seen before?**
> Yes — that is the purpose of machine learning. The model learns patterns from the training data, not specific complaints. As long as the new complaint contains words that the model has seen during training (which is very likely given our 10,000-feature vocabulary), it can classify it. The model generalizes from examples to unseen data.

**Q39. How long does training take?**
> With our sampled dataset (maximum 16,000 rows), training typically takes 2–5 minutes on a standard laptop. The most time-consuming step is the NLP preprocessing pipeline — cleaning 16,000 texts with NLTK. The actual model fitting (TF-IDF + Logistic Regression + Naive Bayes) takes under 30 seconds after preprocessing.

---

## 11. Important Examiner "Trick" Questions

These are commonly asked by examiners to test depth of understanding:

---

**Trick Q1: "If Logistic Regression is a 'regression' algorithm, how can it be used for classification?"**

> **Tip:** Don't panic! Explain that the name is historical — Logistic Regression uses the logistic (sigmoid) function to map continuous outputs to probabilities (0 to 1), and then applies a threshold (0.5) or softmax to assign class labels. The *regression* refers to the internal mathematical process, not the output type. The output is always a class label, making it a classifier.

---

**Trick Q2: "Why is your model 'naive'? Isn't that bad?"**

> **Tip:** Explain that "naive" is a technical term for the conditional independence assumption — the model assumes each word occurs independently of others. In practice, this assumption is violated (words are related), but the model still performs remarkably well for text classification. The naivety is a deliberate simplification that makes the math tractable and training extremely fast, not a design flaw.

---

**Trick Q3: "What if two words have the same TF-IDF score? How does the model differentiate?"**

> **Tip:** Each document is a *vector* of thousands of TF-IDF values — not a single score. The classifier learns weights for all 10,000 features *jointly*, so it considers the full combination of words simultaneously, not just individual words. Two documents might share one word with equal TF-IDF but differ across hundreds of other words.

---

**Trick Q4: "Why do you use weighted F1 instead of macro F1?"**

> **Tip:** Macro F1 gives equal weight to all classes regardless of size, so a rare class contributes as much as a common class. Weighted F1 weights each class by its frequency, reflecting real-world importance. Since our categories naturally vary in frequency (even after sampling), weighted F1 gives a more realistic measure of overall performance.

---

**Trick Q5: "Your model was trained on 2000 samples per class. How do you know it generalizes to real data?"**

> **Tip:** We use a stratified train-test split, so the test set contains data the model has never seen. The metrics reported (accuracy, F1) are measured on this unseen test data, not training data. The consistent performance on unseen test data demonstrates generalization. Additionally, the CFPB data is real user-written text, so the model is trained on authentic language patterns.

---

**Trick Q6: "What would happen if you removed TF-IDF and used raw word counts instead?"**

> **Tip:** Raw word counts would give very high scores to common words like "the", "and", "I" — words that appear in every category equally. These words have no discriminating power for classification. The model would likely perform significantly worse because the most "important" features would actually be uninformative. TF-IDF's IDF component specifically solves this problem.

---

**Trick Q7: "Could you use a neural network instead of Logistic Regression?"**

> **Tip:** Yes! Deep learning models like LSTM, BERT, or other transformers would likely achieve higher accuracy, especially for understanding context (e.g., negation: "not a credit card issue" vs "a credit card issue"). However, classical ML with TF-IDF is a more appropriate choice for a college mini-project because it: (1) requires far less computational resources, (2) is interpretable and explainable, (3) covers the core Web Data Mining concepts, and (4) achieves very good performance (~85-90% F1).

---

**Trick Q8: "What is overfitting? How do you prevent it in your model?"**

> **Tip:** Overfitting is when the model memorizes training data perfectly but fails on new data. We prevent it through: (1) **Regularization** in Logistic Regression (C=5 parameter), (2) **Train-test split** — we evaluate on data the model has never seen, (3) **Sampling** — we use at most 2,000 examples per class, preventing the model from seeing the same example many times, (4) **TF-IDF with max_features=10,000** — limiting vocabulary prevents memorizing rare, one-off words.

---

**Trick Q9: "What is Laplace smoothing in Naive Bayes? Why do you need it?"**

> **Tip:** When a word appears in a test complaint but was never seen during training for a particular category, the probability P(word | category) = 0. Since Naive Bayes multiplies all word probabilities together, one zero probability makes the entire product zero — the category is considered impossible no matter what other words are present. Laplace smoothing (alpha=0.1) adds a small count to every word-category combination, ensuring no probability is exactly zero. It "smooths" the probability distribution.

---

**Trick Q10: "How would you improve this project further?"**

> **Tip:** Great question to demonstrate depth! Mention: (1) Using a larger, balanced dataset without sampling, (2) Trying transformer models like BERT or DistilBERT for higher accuracy, (3) Adding active learning — letting the model flag uncertain predictions for human review, (4) Real-time complaint ingestion via CFPB's live API instead of a static CSV, (5) Adding multilingual support for non-English complaints, (6) Adding an explainability layer (e.g., SHAP values) to show *why* the model made a prediction.

---

## 12. Practical Demonstration Script

### Before the Viva: Preparation
1. Open your laptop and have these tabs ready in your browser:
   - The running Streamlit app (`http://localhost:8501` or cloud URL)
   - The CFPB website: `https://www.consumerfinance.gov/data-research/consumer-complaints/`
   - Your GitHub repository
2. Have VS Code open with `app.py` and `train_model.py` visible
3. Have the `model/` directory open in file explorer to show the saved model files
4. Prepare 3–4 sample complaints of different types (copy from the CFPB website)

---

### The 15-Minute Demo Script

**[0:00–1:00] Introduction (1 min)**
> *"Good morning sir/ma'am. My project is the Online Consumer Complaint Classification System. It's a machine learning web application that automatically classifies consumer financial complaints into product categories using NLP and machine learning. Let me start by showing you the live application."*

**[1:00–3:00] Home Dashboard (2 min)**
> *"This is the home dashboard. You can see on the top row: we have [X] complaints loaded from the CFPB dataset, 8 product categories, our best model achieves [X]% accuracy, and we're using 10,000 TF-IDF features. This chart shows the distribution of complaints across all categories."*

**[3:00–6:00] Complaint Classifier — Core Demo (3 min)**
> *"Now let me show you the core functionality — the classifier."*
> 
> Navigate to 🔍 Complaint Classification.
> 
> Type this sample: *"I have been having issues with my mortgage company. They have been charging me extra fees every month and refused to explain the charges. I tried calling them multiple times but no one could help me understand why my payment amount keeps increasing."*
> 
> Click "Classify Complaint."
> 
> *"As you can see, the system correctly identified this as a Mortgage complaint with [X]% confidence. The bar chart shows the probability scores for all 8 categories — Mortgage has the highest score, which is why it was selected."*

**[6:00–8:00] Analytics Dashboard (2 min)**
> Navigate to 📊 Complaint Analytics.
> 
> *"This is the analytics dashboard. This bar chart shows which financial products get the most complaints — debt collection and credit reporting are historically the most common. This line chart shows monthly complaint trends over time. This choropleth map shows complaint density by US state — California, Texas, and Florida typically have the most complaints, which correlates with their large populations."*

**[8:00–9:30] Dataset Explorer (1.5 min)**
> Navigate to 📋 Dataset Explorer.
> 
> *"Here we can browse the raw CFPB data. I can filter by product category, state, or company. For example, let me filter for Mortgage complaints from California..."* (apply filter). *"You can see the actual complaint text, the company involved, and when it was submitted."*

**[9:30–11:30] Model Performance (2 min)**
> Navigate to 📈 Model Performance.
> 
> *"This page compares our two models. Logistic Regression achieved [X]% accuracy and [X]% weighted F1-score. Naive Bayes achieved [X]% accuracy and [X]% F1-score. The confusion matrix shows that the model correctly classifies most categories — the diagonal is brightly colored, indicating high correct predictions. The main confusion is between [mention two commonly confused categories], which makes sense because these complaints often use similar vocabulary."*

**[11:30–13:00] Code Explanation (1.5 min)**
> Switch to VS Code. Open `train_model.py`.
> 
> *"Let me quickly show you the training code. Here on line [X] we load the CFPB data. On line [X] we apply our preprocessing pipeline. Here we fit the TF-IDF vectorizer with 10,000 features and bigrams. Here we train Logistic Regression with C=5. Here we compare F1-scores and save the best model using joblib."*

**[13:00–14:30] Deployment Explanation (1.5 min)**
> Switch to browser tab with GitHub repo (or Streamlit cloud).
> 
> *"The project is deployed on GitHub and hosted live on Streamlit Community Cloud. Anyone with the URL can access it without installing anything. The model files are committed to the repository, so they load instantly on the cloud."*

**[14:30–15:00] Conclusion (30 sec)**
> *"To summarize: this project demonstrates a complete Web Data Mining pipeline — from real government web data, through NLP preprocessing and TF-IDF feature extraction, through machine learning classification and evaluation, to an interactive web application. It covers all the core concepts of the Web Data Mining course in a single end-to-end system. Thank you."*

---

### How to Handle Questions You Don't Know

If you don't know the answer to a question:

1. **Buy time:** Say *"That's a great question. Let me think for a moment..."*
2. **Relate to what you know:** *"I'm not fully sure of the exact formula, but I know that [related concept]..."*
3. **Be honest:** *"I haven't studied that specific aspect in depth, but in our implementation we approached it by..."*
4. **Redirect to code:** *"Let me show you how we handled that in the code..."* (opens relevant file)
5. **Never bluff a formula** — if you don't know it, say you'd need to look it up. Examiners respect honesty over incorrect answers.

---

## 13. Web Data Mining Lab Requirements

This project satisfies all standard requirements of the Web Data Mining laboratory course:

### ✅ 1. Web Data Collection
**Requirement:** Collect data from web sources.

**How we satisfy it:** We collect data from the Consumer Financial Protection Bureau (CFPB) — a U.S. government web portal at `https://www.consumerfinance.gov`. The dataset is downloaded directly from the web in CSV format. The CFPB also provides a live REST API (`https://api.consumerfinance.gov/data/complaints`), demonstrating real web data collection capabilities.

---

### ✅ 2. Text Mining
**Requirement:** Apply text mining techniques to unstructured data.

**How we satisfy it:** The core of our project is text mining on consumer complaint narratives — raw, unstructured text written by consumers in their own words. We extract patterns, features, and categories from this text using NLP, TF-IDF, and ML classification. The word cloud visualization further demonstrates text mining by visually representing the most common terms in each complaint category.

---

### ✅ 3. Data Preprocessing
**Requirement:** Demonstrate data cleaning and transformation techniques.

**How we satisfy it:** Our `preprocessing.py` module implements a comprehensive preprocessing pipeline:
- **Lowercase conversion** — normalization
- **URL removal** — noise reduction using regex
- **Special character removal** — removing punctuation and symbols
- **Digit removal** — removing irrelevant numerical tokens
- **Whitespace normalization** — cleaning formatting artifacts
- **Stop-word removal** — filtering 179 common English words using NLTK
- **Missing value handling** — dropping rows with null narratives or labels
- **Class balancing** — sampling to prevent dominant class bias

---

### ✅ 4. Feature Extraction
**Requirement:** Convert raw data into features suitable for machine learning.

**How we satisfy it:** We use **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization to convert raw complaint text into numerical feature vectors of 10,000 dimensions. We use `ngram_range=(1,2)` to capture both unigrams and bigrams, `sublinear_tf=True` for log-normalization, and `max_features=10000` for dimensionality control. This is the central feature extraction technique in text mining.

---

### ✅ 5. Classification
**Requirement:** Apply machine learning classification algorithms.

**How we satisfy it:** We implement and compare two classification algorithms:
- **Logistic Regression** (C=5, max_iter=1000) — a linear classifier with regularization
- **Multinomial Naive Bayes** (alpha=0.1) — a probabilistic classifier with Laplace smoothing

Both are trained on the same TF-IDF features with an 80/20 stratified train-test split. The best model is selected based on weighted F1-score and deployed in the web application.

---

### ✅ 6. Visualization
**Requirement:** Visualize data and results using appropriate charts.

**How we satisfy it:** Our `analytics.py` module generates 7+ interactive Plotly visualizations:
- **Bar charts** — category distribution, yearly trends, company analysis
- **Line charts** — monthly complaint trends
- **Pie charts** — company response distribution
- **Heatmaps** — product vs company confusion matrix, ML confusion matrices
- **Choropleth map** — US state-level complaint geographic distribution
- **Word clouds** — most common words per complaint category
- **Confidence bar charts** — prediction probabilities for the classifier

---

### ✅ 7. Pattern Identification
**Requirement:** Identify meaningful patterns in the data.

**How we satisfy it:** Our analytics dashboard identifies multiple meaningful patterns:
- **Temporal patterns:** How complaint volumes change over months and years (e.g., spikes during economic events)
- **Geographic patterns:** Which US states generate the most complaints (correlating with population density and financial activity)
- **Company patterns:** Which financial companies attract the most complaints across different product categories
- **Response patterns:** How response rates and types differ across companies and categories
- **Linguistic patterns:** Which words and phrases are most strongly associated with each product category (visible in word clouds and feature importance)
- **Category patterns:** The natural distribution and imbalance of complaint types in the financial sector

---

*End of Viva Preparation Guide*

---

> **📌 Final Reminder:** Read this guide at least twice before your viva. Practice explaining TF-IDF, the difference between Precision and Recall, and the demo script out loud. Confidence comes from preparation. Good luck! 🎓
