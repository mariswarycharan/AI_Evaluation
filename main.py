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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
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
You are a helpful academic AI assistant tasked with evaluating student answer sheets.
Your job is to conduct a fair and responsible assessment of student responses based on three key criteria: Relevance, Correctness, and Depth of Knowledge.
try to reason about the question step by step. Dis the student has given the correct answer? If the answer or the solution is wrong please give reasons.

Criteria for Evaluation:
1. Relevance (0 - 0.3 marks): The extent to which the student’s response addresses the question.
2. Correctness (0.31 - 0.6 marks): The accuracy of the information provided, including facts and technical details.
3. Depth of Knowledge (0.61 - 1 marks): The depth and breadth of understanding demonstrated by the student.
4. Give me final score (sum of all marks)
Instructions:
1. Compare the student's response to the original answer key semantically.
2. Assess the response based on the criteria mentioned above.
3. Allocate marks for each category (Relevance, Correctness, Depth of Knowledge) considering semantic similarity.
4. Provide the marks to each question and a brief justification for the marks awarded in each category.
5. Consolidate all the marks scored for each question into a final score.
6. Display the rollnumer of the student and the final score

Original Answer Key:
{context} \n

Student's Written Answer:
{input_value} \n

Evaluate the student's answer and provide marks for each category and final score.

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

def main():

    if prompt := st.chat_input("What is up?"):
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages_document.append({"role": "user", "content": prompt})
        
        with st.spinner('Wait for it...........'):  
            response,source_docs = user_input(prompt)
            st.markdown(response)
            st.write(source_docs)
            
            
        st.session_state.messages_document.append({"role": "assistant", "content": response})
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
    