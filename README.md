# Suspicious Message Detection using OpenAI Embeddings
## Overview
This project aims to detect suspicious text messages provided by a user.
It uses OpenAI's embeddings model to convert message text into numerical representations, which are then used to train a machine learning classifier. 
The classifier can then predict the category of new, unseen messages based on their embeddings.

## Results
The classifier has mixed results due to the size limitations of the data set.
It pretty accurately detects safe messages; however, sometimes it miscategorizes malicious messages as legitimate.
The image below shows sample results. In the first three messages, the classifier was accurate.
The last message provides an example of the classifiers's innaccuracies. You can see that the the link seems malicious, but the classifer still classified it as safe.
<img width="833" alt="image" src="https://github.com/cheyenne-ashou/OpenAiPhisingDetection/assets/54869764/c7fe008d-e0cf-4585-891c-63ed24f295bf">

## How can we improve accuracy?
1. Data Quality and Quantity
  - **Increase Dataset Size:** Collect more labeled examples from diverse sources.
  - **Data Augmentation:** Create synthetic emails to increase data diversity.
  - **Data Cleaning:** Remove noise and ensure a balanced dataset across categories.

2. Feature Engineering
  - **Additional Features:** Include metadata (e.g., sender info, subject) and handcrafted features (e.g., presence of keywords).

3. Active Learning
  - **Model Improvement:** Use active learning to focus on uncertain samples, label them, and improve the training set iteratively.

ALthough I am not well-versed in Machine learning, these are simple ways to improve the accuracy of the classifier without going deep into Maachine Learning.
