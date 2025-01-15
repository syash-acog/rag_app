from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os
import argparse
from dotenv import load_dotenv


load_dotenv()
hf_api_key: str = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def ask_question(link: str, question: str, model_name: str) :

    loader = WebBaseLoader(link)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = Chroma.from_documents(document_chunks, embedding_model)

    prompt_template = """
    Use the context only to answer the question at the end.Be concise and to the point. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    llm_model = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=hf_api_key,
        max_new_tokens= 250,
        temperature= 0.5,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

    answer: dict[str, any] = qa_chain.invoke(question)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Ask questions using talk_to_web_rag.py.")
    parser.add_argument("--link", required=True, help="The link to the webpage.")
    parser.add_argument("--question", required=True, help="The question to ask.")
    parser.add_argument("--model", required=True, help="The name of the LLM model to use.")
    args = parser.parse_args()

    answer = ask_question(args.link, args.question, args.model)
    print("Answer:", answer["result"])
    
if __name__ == "__main__":
    main()