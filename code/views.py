from django.shortcuts import render
from .qa_model import detect_causality

def index(request):
    """
    Renders the index page where users can upload documents and ask questions.
    """
    if request.method == "POST":
        context_text = request.POST['context']
        question_text = request.POST['question']
        language = request.POST.get('language')

        # Get answers using the QA models
        extractive_ans, generative_ans = detect_causality(context_text, question_text, language)

        # Pass the results to the template
        context = {
            'extractive_answer': extractive_ans,
            'generative_answer': generative_ans
        }
        return render(request, 'qa_app/results.html', context)

    return render(request, 'qa_app/index.html')
