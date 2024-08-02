import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_string
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import pandas as pd
import torch
import joblib
import warnings
import time
import random

warnings.filterwarnings('ignore')
    

# Descargar stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_question_type(question):
    if question is None:
        return ''
    # Convertir a minúsculas
    question = question.strip()
    question = ' '.join(question.split())
    question = question.lower()
    # Eliminar etiquetas HTML (si las hubiera)
    question = re.sub(r'<.*?>', '', question)
    # Eliminar caracteres especiales
    question = re.sub(r'&amp;quot;', "", question)
    question = re.sub(r'&apos;', "'", question)
    question = re.sub(r'&quot;', "", question)
    question = re.sub(r'&amp;/', "", question)
    question = re.sub(r'&amp;', "", question)
    question = re.sub(r'&gt;', "'", question)
    question = re.sub(r'[^a-zA-Z0-9\s]', '', question)
    # Tokenizar el questiono
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(question)
    # Eliminar stop words
    question_tokens = [word for word in tokens if word not in stopwords.words('english')]
    question = ' '.join([word for word in question_tokens if word not in stop_words])
    return question

def vectorize_question(question):
    vector_size = model_w2v.vector_size
    # Inicializar un vector de ceros
    vector = np.zeros(vector_size)
    # Contar las palabras que están en el vocabulario de Word2Vec
    count = 0
    for word in question:
        if word in model_w2v.wv.key_to_index:
            vector += model_w2v.wv[word]
            count += 1
    if count != 0:
        vector /= count
    return vector

def detection_type(question):
    question = preprocess_question_type(question)
    question = vectorize_question(preprocess_string(question))
    type_pred = model_type_loaded.predict(question.reshape(1, -1))
    return type_pred

def preprocess_question_focus(question):
    if question is None:
        return ''
    # Convertir a minúsculas
    question = question.strip()
    question = ' '.join(question.split())
    question = question.lower()
    # Eliminar etiquetas HTML (si las hubiera)
    question = re.sub(r'<.*?>', '', question)
    # Eliminar caracteres especiales
    question = re.sub(r'&amp;quot;', "", question)
    question = re.sub(r'&apos;', "'", question)
    question = re.sub(r'&quot;', "", question)
    question = re.sub(r'&amp;/', "", question)
    question = re.sub(r'&amp;', "", question)
    question = re.sub(r'&gt;', "'", question)
    question = re.sub(r'[^a-zA-Z0-9\s]', '', question)
    task_prefix = "Keywords: "
    input_question = task_prefix + question
    tokenized_question = tokenizer_focus_loaded(input_question, return_tensors="pt", truncation=True, max_length=512).input_ids
    return tokenized_question

def extract_focus(question):
    tokenized_question = preprocess_question_focus(question)
    output = model_focus_loaded.generate(tokenized_question, no_repeat_ngram_size=3, num_beams=4, max_length=50)
    focus = tokenizer_focus_loaded.decode(output[0], skip_special_tokens=True)
    return focus

def generation_QA_question(type_pred, focus_pred):
    final_question = f'Can you tell me {type_pred} of {focus_pred}?'
    return final_question

def answer_question(question, max_length=256, temperature=0.7, top_k=50, top_p=0.95):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer_QA_loaded(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = model_QA_loaded.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer_QA_loaded.eos_token_id,
        )
    
    answer = tokenizer_QA_loaded.decode(outputs[0], skip_special_tokens=True)
    answer = answer[len(prompt):]  # Eliminar el prompt de la respuesta
    return clean_answer(answer)

def clean_answer(answer):
    # Limitar a la última oración completa
    end_marks = ['.', '!', '?']
    for mark in end_marks:
        pos = answer.rfind(mark)
        if pos != -1:
            answer = answer[:pos + 1]
            break

    # Eliminar enlaces usando una expresión regular
    answer = re.sub(r'http\S+', '', answer)
    answer = re.sub(r'www\.\S+', '', answer)
    return answer

if __name__ == "__main__":
    # Models loading
    w2vec_path = '../models/word2vec.model'
    model_w2v = Word2Vec.load(w2vec_path)
    model_type_path = '../models/model_classification_balanced.joblib'
    model_type_loaded = joblib.load(model_type_path)
    model_focus_path = '../models/model_focus'
    model_focus_loaded = T5ForConditionalGeneration.from_pretrained(model_focus_path)
    tokenizer_focus_loaded = T5Tokenizer.from_pretrained(model_focus_path)
    model_QA_path = '../models/model_QA'
    model_QA_loaded = GPT2LMHeadModel.from_pretrained(model_QA_path)
    tokenizer_QA_loaded = GPT2Tokenizer.from_pretrained(model_QA_path)

    cont = 1
    # Ask a question
    while cont:
        input_question = str(input('You (ctrl+c to exit): '))
        t0 = time.time()

        # Classification of the question
        type_pred = detection_type(input_question)
        t_type = time.time() - t0

        # Extraction of the focus
        focus_pred = extract_focus(input_question)
        t_focus = time.time() - t0 - t_type

        # Preparation of the question given to the QA model
        final_question = generation_QA_question(type_pred, focus_pred)
        # Final answer to the question
        print(f'Answer: {answer_question(final_question)}')
        t_answer = time.time() - t0 - t_type - t_focus
        t_total = t_type + t_focus + t_answer
        print(f'Time for type: {t_type:.0f} s')
        print(f'Time for focus: {t_focus:.0f} s')
        print(f'Time for answer: {t_answer:.0f} s')
        print(f'Time elapsed: {t_total:.0f} s')
        random_numb = random.randint(1, 10)
        if random_numb==1:
            input_feedback = str(input('Are you happy with this answer (y/n)? '))