import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your OpenAI API key
load_dotenv()
api_key = os.getenv("OPEN_AI_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Choose the embeddings model
model_id = "text-embedding-ada-002"


def get_embeddings(text):
    logging.info(f"Generating embeddings for text: {text[:50]}...")
    response = client.embeddings.create(
        input=text,
        model=model_id
    )
    embeddings = response.data[0].embedding
    return embeddings


# Load the dataset
logging.info("Loading dataset...")
df = pd.read_csv('data/phishing_dataset.csv')
logging.info(f"Dataset loaded with {len(df)} entries.")

# Generate embeddings for each text message
df['embeddings'] = df['TEXT'].apply(get_embeddings)
logging.info("Embeddings generation completed.")

# Prepare the feature matrix and labels
X = list(df['embeddings'])
y = df['LABEL']

# Encode the labels
logging.info("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
logging.info("Labels encoded.")

# Split the data into training and testing sets
logging.info("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
logging.info(f"Training set: {len(X_train)} samples, Testing set: {len(X_test)} samples.")


# Train a logistic regression classifier
logging.info("Training the logistic regression classifier...")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
logging.info("Classifier trained successfully.")

# Save the trained model and the label encoder
classifier_path = os.path.join('data', 'classifier.pkl')
label_encoder_path = os.path.join('data', 'label_encoder.pkl')

logging.info("Saving the trained model and label encoder...")
with open(classifier_path, 'wb') as f:
    pickle.dump(classifier, f)
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
logging.info("Model and label encoder saved.")

# Evaluate the model
logging.info("Evaluating the model...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model accuracy: {accuracy}")

