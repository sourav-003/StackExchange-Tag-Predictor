# 📌 Auto Tagging System (GRU)

## 🔎 Project Overview
This project implements an **automatic tag prediction system** for StackExchange questions. Given the **title + body text** of a question, the system predicts the most relevant tags using a **deep learning model** based on **Gated Recurrent Units (GRU)**.

The workflow covers the **entire ML lifecycle** — from **data preprocessing → feature engineering → model training → evaluation → saving artifacts → deployment with Gradio**.

---

## 📂 Dataset
- **Source**: Stack Overflow Questions Dataset (Kaggle)
- **Files Used**:
  - `Questions.csv` → Contains `Id`, `Title`, `Body`, etc.
  - `Tags.csv` → Maps each `Id` to its corresponding tags.
- **Size**: ~85,000 questions
- **Tag Distribution**: Multi-label problem (a question can have multiple tags)

---

## ⚙️ Workflow

### 1. Data Preprocessing
- Merged `Questions.csv` and `Tags.csv` on `Id`.
- Grouped tags per question into lists.
- Combined **Title + Body** into a single `Text` column.
- Cleaned text using:
  - **BeautifulSoup** → removed HTML tags
  - **Regex rules** → removed punctuation, special characters
  - Converted all text to lowercase

### 2. Multi-label Encoding
- Used **MultiLabelBinarizer** to convert tag lists into binary vectors
- Each question is represented by a **multi-hot encoded vector** of tags

### 3. Text Representation
- Tokenized text using **Keras Tokenizer**
- Built a vocabulary (ignoring words appearing <3 times)
- Converted text into padded sequences (`max_len = 100`)

### 4. Model Architecture (GRU)
- **Embedding Layer** → Dense word embeddings
- **GRU Layer** → Captures sequential dependencies in text
- **Dense + Dropout** → For dimensionality reduction & regularization
- **Output Layer (Sigmoid)** → Multi-label probability distribution across tags

### 5. Training
- **Loss Function**: Binary Cross-Entropy (multi-label)
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy and validation loss curves

### 6. Saving Artifacts
- Saved **Tokenizer** and **MultiLabelBinarizer** using Pickle
- Saved **trained GRU model** using Keras `model.save()`

### 7. Deployment (Gradio App)
- Built a **Gradio interface** (`app.py`) to allow:
  - **Input** → Any text query (question)
  - **Output** → Predicted tags with probabilities

---

## 📊 Results
- Successfully predicts relevant tags for unseen questions
- Validation accuracy improves steadily with training
- Provides a functional interactive demo through Gradio

---

## 🚀 Future Improvements
- Incorporate **pre-trained embeddings** (GloVe / Word2Vec / FastText)
- Experiment with **LSTM** and **Transformer-based models (BERT, DistilBERT)**
- Improve multi-label accuracy using **attention mechanisms**
- Deploy as a **full-stack web app** (Flask/Streamlit + Heroku/Render)
- Add explainability features (e.g., SHAP, LIME for tag prediction insights)

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras, BeautifulSoup, Regex
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Gradio, Pickle
- **Tools**: Jupyter Notebook, Git, VS Code

---

## 📌 Author
**Sourav Kumar**  
- 📧 Email: souravmail003@gmail.com  
- 🔗 [LinkedIn](https://linkedin.com/in/sourav-kumar-5814341b8)  
