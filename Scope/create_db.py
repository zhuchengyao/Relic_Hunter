from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
import chromadb
import os


chroma_client = chromadb.Client()
client = chromadb.PersistentClient(path="./db_history_data")


embedding = DashScopeEmbeddings(
    model="text-embedding-v2", dashscope_api_key="sk-a55c969f708b43429ec601d536f9efac")


dir_path = []
dir_name = []
folder_path = './history/'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        dir_path.append(folder_path+file)
        dir_name.append(file)

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "。", " ", ""],
                                               chunk_size=128,
                                               chunk_overlap=0)

# text chunk
for i, file in enumerate(dir_path):
    Loader = TextLoader(file)
    document = Loader.load()
    file_chunks = text_splitter.split_documents(document)
    db = Chroma.from_documents(file_chunks, embedding, persist_directory=f"db_{dir_name[i].rstrip('.txt')}")

