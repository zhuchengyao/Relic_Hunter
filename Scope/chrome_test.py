import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

with open("./history/baihuabeiqishu.txt", "r") as f:
    file_content = f.read()

collection.add(
    documents=["This is a document about engineer", "This is a document about steak", file_content],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}, {"source": "baihuabeiqishu.txt"}],
    ids=["id1", "id2", 'id3']
)

results = collection.query(
    query_texts=["北齐皇帝是谁"],
    n_results=2
)

print(results)

db = Chroma.from_documents(texts_chunks, embedding, persist_directory="db")