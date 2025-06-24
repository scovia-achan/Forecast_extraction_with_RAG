from langchain_core.prompts import PromptTemplate

prompt_template = """
System: You are a helpful assistant that provides concise and accurate answers based on the provided context.
Based on the following context, answer the question as precisely as possible. Cite years or regions when possible.

Context:
{context}

Question: {question}

Answer: """

input_variables = ["context", "question"]

prompt = PromptTemplate(
    input_variables=input_variables,
    template=prompt_template
)