# Arabic Autocomplete System - Documentation
## 1. Introduction
The Arabic Autocomplete System uses a **`Statistical`** machine learning-based language model aimed at predicting and suggesting the next possible word(s) in a sequence of Arabic text input. It uses sequential modeling techniques to understand language patterns and provide accurate word predictions.

## 2. Demo of a continuous Auto Completed Sentence
| image 3 | image 2 | image 1 |
|--------|--------|--------|
|![alt1](demo_images/3.png)|![alt1](demo_images/2.png)|![alt1](demo_images/1.png)|
|![alt1](demo_images/6.png)|![alt1](demo_images/5.png)|![alt1](demo_images/4.png)|
|![alt1](demo_images/9.png)|![alt1](demo_images/8.png)|![alt1](demo_images/7.png)|
|![alt1](demo_images/12.png)|![alt1](demo_images/11.png)|![alt1](demo_images/10.png)|
|![alt1](demo_images/15.png)|![alt1](demo_images/14.png)|![alt1](demo_images/13.png)|
|![alt1](demo_images/18.png)|![alt1](demo_images/17.png)|![alt1](demo_images/16.png)|
|![alt1](demo_images/21.png)|![alt1](demo_images/20.png)|![alt1](demo_images/19.png)|

## 3. Dataset Information
The dataset used for training is a corpus of Arabic text collected from various sources such as Arabic news websites, Wikipedia articles, and literature.  
The dataset consists of approximately `1 million` **sentences** and over `10 million` **words**.  
Preprocessing steps include:
- Tokenization.
- Normalization (removing diacritics, unifying different forms of letters).
- Removing punctuation and non-Arabic characters.
- Padding and truncating sequences to a fixed length.


## 4. The Trained Model file can be downloaded from Hugging Face either:
### a) Manually from this link: [Arabic Auto Completion Model](https://huggingface.co/michaelHenry1/Ngrams_Arabic_AutoCompletion/blob/main/arabic_ngram_model_full.pkl)
 - ### OR
### b) Using the `huggingface_hub` library:
```python
from huggingface_hub import hf_hub_download
import pickle

# Download the file to local cache
file_path = hf_hub_download(repo_id="michaelHenry1/Ngrams_Arabic_AutoCompletion", filename="arabic_ngram_model_full.pkl")

with open(file_path, "rb") as f:
    model = pickle.load(f)
```
