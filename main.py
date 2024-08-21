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
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from ragas import evaluate
from datasets import Dataset


st.cache_resource(show_spinner=False)
def load_model():
    load_dotenv()
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0 , convert_system_message_to_human=True)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    global embeddings
    # vector_store = Chroma.from_texts(text_chunks, embedding = embeddings , persist_directory="chroma_db")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


st.cache_resource(show_spinner=False)   
def get_conversational_chain():
    global prompt_template
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
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt,verbose=False)
        
    return chain


def user_input(user_question):
    
    global new_db
    
    docs = new_db.similarity_search(user_question ,k=1)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "input_value": user_question}
        )
    
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
            
            prompt_list = ["""What is the goal of prediction?
The goal of prediction is to predict a quantity""" ,"""2. List few prediction algorithms.
Linear regression, polynomial regression, support vector regression""",""" 3. Write any use case for prediction task.
Estimating house prices based on features like size, location, and number of rooms.""","""a. An algorithm that establishes a linear relationship between an independent
variable and a dependent variable for making predictions.
b. y=β0+β1x+ϵ """ , """5. John smith has decided to purchase a house and he was not sure about the price
of house. Which machine learning model would you suggest for this use case to
predict the price of house?
Regression
 """,""" 6. What are the performance metrices of regression?
Mean Squared Error (MSE),
Root Mean Squared Error (RMSE),
Mean Absolute Error (MAE),
R-squared
Adjusted R-squared
""" , """ 7. Apply regression analysis between sales (y in Rs) and advertising cost (x in Rs)
across all the branches of the company Ranson. The data analyst discovered the
regression as . If the advertising budgets of two branches of
the company Ranson differ by Rs 2000, then what will be the predicted
difference in their sales?
Answer: Rs. 41,000/-""" , """ 8. Explain how you would determine whether a simple linear regression model is
appropriate for your data.
To determine if a simple linear regression model is appropriate, I would first visually
inspect the scatter plot of the data to check for a linear relationship. Then, I would
examine the residuals for patterns; they should be randomly scattered if the model is
appropriate. I would also calculate the correlation coefficient to assess the strength of
the linear relationship.
""" , """9. The function (Squiggly line), training and test samples of a dataset is given
below:
a. Identify the problem faced by the model from the above visualization.
b. Provide different solutions to fix the issue and make it perfect?    """ , """10. Identify the type of regression given below and discuss the pros and cons of the
model.
Polynomial Regression
Pros - can model non-linear relationships between variables
Cons - models are prone to overfitting"""

            ]
            
            # for prompt in prompt_list:
            #     response,source_docs = user_input(prompt)
            #     st.markdown(response)
            
            response,source_docs = user_input(prompt)
            st.markdown(response)
            
            # st.subheader("Relevant Documents : ")
            # st.write()
            
            d = {
                "question": [prompt],
                "answer": [response],
                "contexts": [[source_docs[0].page_content]],
                "ground_truth": [prompt_template.replace("context",source_docs[0].page_content).replace("input_value",prompt)]
            }

            dataset = Dataset.from_dict(d)
            score = evaluate(dataset, metrics=[faithfulness,answer_relevancy, answer_similarity, answer_correctness]
                            , llm=model, embeddings=embeddings,)
            
            # st.subheader("Score : ")
            print(score)
            st.write(score)
            
            score_df = score.to_pandas()
            score_df.to_csv("EvaluationScores.csv", encoding="utf-8", index=False)
        
            st.subheader("Metrics : ")
            st.write(score_df)
            
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
    