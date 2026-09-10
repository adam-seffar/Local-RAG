import ollama

class RagPipeline:
    def __init__(self):
        self.ollama_model = "llama3:8b"

    def build_context(self, documents):
        return "\n\n".join(documents)

    def build_prompt(self, documents, query, conversation_history=None):
        context = self.build_context(documents)

        return f"""STRICT INSTRUCTIONS:
            1. You must ONLY use information from the CONTEXT below to answer the QUESTION.
            2. If the CONTEXT does not contain enough information to answer the QUESTION, you MUST say: "I cannot answer this question based on the provided information."
            3. Do not use any external knowledge, assumptions, or personal opinions.
            4. If context is empty, you must say: "There is no information in the provided context"

            CONTEXT:
            {context}
            CONVERSATION HISTORY:
            {conversation_history}
            QUESTION:
            {query}

            ANSWER (based only on the context above):"""

    def generate_answer(self, prompt):
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt
        )
        return response["response"]