import streamlit as st
from transformers import pipeline
import psutil

# Check system resources
def check_resources():
    mem = psutil.virtual_memory()
    if mem.available < 500 * 1024 * 1024:  # Less than 500 MB available
        st.error("Not enough memory to run this application. Available memory: {} MB".format(mem.available / 1024 / 1024))
        return False
    return True

if check_resources():
    summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6")
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    examples = [
        [   
            'Question-Answer',
            '',
            'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.',
            'Why is model conversion important?'
        ],
        [   
            'Question-Answer',
            '',
            "The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America", 
            "Which continent is the Amazon rainforest in?"
        ],
        [   
            'Question-Answer',
            '',
            'I am a Programmer.',
            'Who am I?' 
        ]
    ]

    def summarize_text(text):
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        summary = summary[0]['summary_text']
        return summary

    def question_answer(context, question):
        QA_input = {
            'context': context,
            'question': question
        }
        res = nlp(QA_input)
        return res['answer']

    st.title("Text Summarizer and Question-Answering App")

    model_choice = st.selectbox("Choose the model", ["Text Summarizer", "Question-Answer"])

    if model_choice == "Text Summarizer":
        summ_text = st.text_area("Enter the text to be summarized", height=150)
        if st.button("Summarize"):
            if summ_text.strip() == "":
                st.error("Please enter some text to summarize.")
            else:
                summary = summarize_text(summ_text)
                st.write("Summary:")
                st.write(summary)

    elif model_choice == "Question-Answer":
        qa_context = st.text_area("Enter the context", height=150)
        qa_question = st.text_area("Enter the question", height=50)
        if st.button("Get Answer"):
            if qa_context.strip() == "" or qa_question.strip() == "":
                st.error("Please enter both context and question.")
            else:
                answer = question_answer(qa_context, qa_question)
                st.write("Answer:")
                st.write(answer)

    st.sidebar.title("Examples")
    example_index = st.sidebar.selectbox("Choose an example", range(len(examples)), format_func=lambda x: examples[x][0])

    if example_index is not None:
        st.sidebar.write("Context:")
        st.sidebar.write(examples[example_index][2])
        st.sidebar.write("Question:")
        st.sidebar.write(examples[example_index][3])
