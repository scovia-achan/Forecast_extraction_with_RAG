from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""
System: You are a helpful assistant that provides concise and accurate answers based on the provided context. Based on the following context, answer the question as precisely as possible. Cite years or regions when possible.
user: What were the main trends in {topic} in {place} in {year}?
                                      
""")

input_variables = {
    "topic": "Digital Transformation",
    "place": "Africa",
    "year": "2013 to 2017"
}