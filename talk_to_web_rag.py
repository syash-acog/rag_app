from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def ask_question(link, question, llm_model):

    loader = WebBaseLoader(link)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = Chroma.from_documents(document_chunks, embedding_model)

    prompt_template = """
    Use the context only to answer the question at the end. Answer the following question in 1-2 sentences. 
    Be concise and to the point. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

    answer = qa_chain.invoke(question)

    return answer


if __name__ == "__main__":
    webpage_link = "https://en.wikipedia.org/wiki/Sigmund_Freud"
    question = "what was freud's realtionship with Wilhelm Fliess?"
    llm_model = ChatGroq(model_name="llama-3.3-70b-versatile")

    answer = ask_question(webpage_link, question, llm_model)
    print("Answer:", answer['result'])
