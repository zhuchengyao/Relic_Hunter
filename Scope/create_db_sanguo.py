from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain, SimpleSequentialChain, RetrievalQA, ConversationalRetrievalChain
import chromadb
import os


chroma_client = chromadb.Client()
client = chromadb.PersistentClient(path="./db_history")


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
                                               chunk_size=256,
                                               chunk_overlap=32)
select_dir = []
select_dir.append(dir_path[23])
select_dir.append(dir_path[25])
select_dir.append(dir_path[22])
for i,j in enumerate(select_dir):
    print(i)
    print(":")
    print(j)
# print(dir_path[25])
# # text chunk
for i in select_dir:
    file = i
    Loader = TextLoader(file)
    document = Loader.load()
    file_chunks = text_splitter.split_documents(document)
    db = Chroma.from_documents(file_chunks, embedding, persist_directory=f"db_history")

