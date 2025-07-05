# Sentiment Classification Project

This project implements a sentiment analysis classifier that predicts whether a given text expresses a **positive-1** or **negative-0** sentiment. The main approach uses a deep learning model based on a pre-trained DistilBERT transformer.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Setup Instructions](#setup-instructions)  
- [Running the Code](#running-the-code)  
<!-- - [LLM-based Approach](#llm-based-approach)   -->
- [External Libraries](#external-libraries)  

---

## Project Structure
│
├── classifier/
│ ├── dataset.py # Dataset class for loading and tokenizing data
│ ├── train.py # Training functions
│ ├── evaluate.py # Evaluation functions
│ ├── predict.py # Prediction and saving results
│ ├── sentiment_classifier.py # Main script to run training, evaluation, and prediction
│ └── init.py
│
├── data/
│ └── imdb.csv # Dataset file (not included)
│
├── requirements.txt # Python dependencies
├── README.md # This documentation file


## SetUp Instruction

1. **Clone the repository**

```bash
md sentiment-classification
cd sentiment-classification
git clone https://github.com/Asini-Jayakody/sentiment-classification.git
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate         # On Linux/macOS
# OR
venv\Scripts\activate            # On Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

## Running the Code

Full model training, evaluating and predicting can done by running the following command.

```bash
python sentiment_classifier.py
```

## External Libraries

- PyTorch 
- Transformers (Hugging Face)
- scikit-learn
- pandas
- tqdm
