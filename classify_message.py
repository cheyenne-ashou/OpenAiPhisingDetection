import os
from openai import OpenAI
import pickle
import logging
from dotenv import load_dotenv

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


# Load the trained classifier and the label encoder
classifier_path = os.path.join('data', 'classifier.pkl')
label_encoder_path = os.path.join('data', 'label_encoder.pkl')

with open(classifier_path, 'rb') as f:
    classifier = pickle.load(f)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)


def classify_text_message(text_message):
    embeddings = get_embeddings(text_message)
    prediction = classifier.predict([embeddings])[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return interpret_label(label)


def interpret_label(label):
    label = label.lower()
    if label == "spam" or label == "smishing":
        return "This message seems suspicious. " \
               "If you do not recognize the sender, please report the message to your provider."
    return "This message seems safe. If you do not recognize the sender, " \
           "please report the message to your provider."


if __name__ == "__main__":
    user_text_message = input("Enter the text message: ")
    while user_text_message != 'exit':
        result = classify_text_message(user_text_message)
        print(f'result: {result}')
        user_text_message = input("Enter the text message: ")
