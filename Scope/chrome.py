from langchain_community.vectorstores import Chroma
from utility import embedding_method, book_name



db = Chroma(persist_directory="./db_history", embedding_function=embedding_method())
query = input("")
res = db.similarity_search(query, k=6)
# print(res)
contents = [i.page_content for i in res]
sources = [i.metadata['source'] for i in res]
print(contents)