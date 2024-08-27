import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER

st.set_page_config(page_title="AI Evaluation", page_icon="🧠", layout="wide")

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.cache_resource(show_spinner=False)
def load_model():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0 , convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings , allow_dangerous_deserialization=True)
    else:
        new_db = ''
    
    return model,embeddings,new_db


model,embeddings,new_db = load_model()



def get_pdf_text(docs):
    
    text = ''
    for pdf in docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    global embeddings
    # vector_store = Chroma.from_texts(text_chunks, embedding = embeddings , persist_directory="chroma_db")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

st.cache_resource(show_spinner=False)   
def get_conversational_chain():

    prompt_template = """
You are a highly intelligent and meticulous academic AI assistant tasked with evaluating student answer sheets. Your primary objective is to perform a fair, thorough, and insightful assessment of student responses based on three critical criteria: Relevance, Correctness, and Depth of Knowledge. Your evaluation should not only determine if the student’s response is correct but also consider how well the student has understood and articulated the underlying concepts.

**Evaluation Criteria:**
1. **Relevance**: Evaluate the extent to which the student’s response directly addresses the question posed. Consider whether the answer stays on topic and fulfills the requirements of the question.
2. **Correctness**: Assess the accuracy of the information provided by the student. This includes checking facts, figures, and any technical details to ensure the response is factually correct and logically sound.
3. **Depth of Knowledge**: Analyze the depth and breadth of the student’s understanding as demonstrated in the response. Look for insightful explanations, connections to broader concepts, and a clear demonstration of mastery over the subject matter.

**Scoring Instructions:**
- Allocate a maximum of 2 marks per question, considering all three criteria together.
- If a student has answered multiple questions, sum up the marks awarded for each question and provide a final score in the format: *Student scored X/Total Marks*.

**Process:**
1. Begin by semantically comparing the student's response to the original answer key.
2. Reason through the answer step by step to determine if the student has given a correct and relevant response.
3. If an answer is incorrect or incomplete, provide a brief explanation highlighting the inaccuracies or missing elements.
4. After evaluating each question individually, sum up the marks to provide a final score.

**Original Answer Key:**
{context}

**Student's Written Answer:**
{input_value}

Evaluate the student's answers, allocate up to 2 marks per question based on the overall assessment, provide justifications for the marks awarded, and calculate the final score. If there are 10 questions, for example, you should present the final score as *Student scored X/20* (with 20 being the total possible marks for 10 questions).

"""
    

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "input_value"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    
    global new_db
    
    docs = new_db.similarity_search(user_question ,k=1, fetch_k=10)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "input_value": user_question}
        , return_only_outputs=True)


    return response["output_text"],docs

# Initialize chat history
if "messages_document" not in st.session_state:
    st.session_state.messages_document = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages_document:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def generate_pdf(content):
    # Create a buffer to hold the PDF data
    buffer = io.BytesIO()

    # Create a document template with margins
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    # Set up styles
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    header_style = styles['Heading1']
    header_style.alignment = TA_CENTER

    # Container for the 'flowable' elements in the document
    elements = []

    # Add a header
    elements.append(Paragraph("Student Report", header_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Split the content into paragraphs and add to PDF
    for line in content.split('\n'):
        elements.append(Paragraph(line, normal_style))
        elements.append(Spacer(1, 0.1 * inch))  # Add space between lines

    # Add a page break (optional)
    elements.append(PageBreak())

    # Build the PDF
    doc.build(elements)

    # Seek the buffer to the beginning
    buffer.seek(0)
    return buffer

def main():
    if "messages_document" not in st.session_state:
        st.session_state.messages_document = []

    with st.sidebar:
        st.title("Upload Answer Key:")
        pdf_docs = st.file_uploader("Upload your Answer Key and Click on the Submit & Process Button", accept_multiple_files=True)
        answer_key_button = st.button("Submit & Process Answer Key")
        
        st.title("Upload Student's Answer Sheet:")
        question_docs = st.file_uploader("Upload your Student's Answer Sheet and Click on the Submit & Process Button", accept_multiple_files=True)
        question_button = st.button("Submit & Process Student's Answer Sheet")

        if answer_key_button:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
                
    if question_button:
        text_content_question = get_pdf_text(question_docs)
        st.chat_message("user").markdown(text_content_question)
        st.session_state.messages_document.append({"role": "user", "content": text_content_question})
        
        with st.spinner('Wait for it...........'):  
            response, source_docs = user_input(text_content_question)
            st.markdown(response)
            st.session_state.messages_document.append({"role": "assistant", "content": response})
            pdf = generate_pdf(response)
            st.download_button(label="Download PDF",
                                data=pdf,
                                file_name="report.pdf",
                                mime="application/pdf") 


if __name__ == "__main__":
    main()
    