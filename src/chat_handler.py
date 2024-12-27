import openai
from data_handler import DataHandler

class Chatbot:
    def __init__(self, gpt_model="gpt-3.5-turbo", data_file="the data"):
        self.gpt_model = gpt_model
        self.data_handler = DataHandler(data_file)
        self.data_handler.build_vectorstore(self.data_handler.load_data())

    def get_response(self, user_query):
        relevant_docs = self.data_handler.query_data(user_query, top_k=1)
        context = relevant_docs[0].page_context if relevant_docs else ""

        prompt = (
            "You are a diabetes assistant. Use the following information to answer the user's query. "
            "If the information is not definitive, provide suggestions or explain that outcomes may vary. "
            "Context:\n"
            f"{context}\n\n"
            "User Query:\n"
            f"{user_query}"
        )

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing health advice."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]