import json
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

class DataHandler:
    def __init__(self, data_file):
        self.data_file = data_file
        self.vectorstore = None

    def load_data(self):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        return data
    
    def build_vectorstore(self, data):
        docs = [{"content": entry["answer"], "metadata": {"topic": entry["category"]}} for entry in data]
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(docs, embeddings)

    def query_data(self, query, top_k=1):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not built. Load and build it first.")
        return self.vectorstore.similarity_search(query, k=top_k)
    
if __name__ == "__main__":
    handler = DataHandler("data here")
    data = handler.load_data()
    handler.build_vectorstore(data)
    result = handler.query_data("What is a good diet for diabetes?")
    print(result[0].page_content)
