import spacy
from transformers import pipeline

# Load language models for preprocessing
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")

# Initialize pre-trained transformers for extractive QA
extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Initialize GPT-2 or another generative model for generative QA
generative_qa = pipeline("text-generation", model="gpt2")

def preprocess_text(text, language="en"):
    """
    Preprocess text using SpaCy NLP model for tokenization and language-specific cleanup.
    """
    if language == "en":
        doc = nlp_en(text)
    else:
        doc = nlp_es(text)
    return " ".join([token.text for token in doc])

def extractive_answer(context, question, language="en"):
    """
    Use the extractive QA model to find the answer from the given context.
    """
    return extractive_qa(question=question, context=context)['answer']

def generative_answer(context, question, language="en"):
    """
    Use the generative model to generate more creative and complex answers.
    """
    return generative_qa(context + question, max_new_tokens=50)[0]['generated_text']

def detect_causality(context, question, language="en"):
    """
    Detect causal relationships in financial disclosures using hybrid QA.
    """
    preprocessed_context = preprocess_text(context, language)
    preprocessed_question = preprocess_text(question, language)

    # First, use extractive QA for exact answers
    exact_answer = extractive_answer(preprocessed_context, preprocessed_question, language)

    # Then, use generative QA for more complex relationships
    generated_answer = generative_answer(preprocessed_context, preprocessed_question, language)

    return exact_answer, generated_answer
