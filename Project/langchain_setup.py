# Imports
from langchain_community.llms import OpenAI  # Correct import based on the new version
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# langchain_setup.py

# Initialize the OpenAI LLM with your API key
llm = OpenAI(api_key="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")  # Replace with your actual API key

# Define a basic prompt template for testing the LLM
test_prompt = PromptTemplate(
    input_variables=["input_text"],
    template="Summarize the following news report: {input_text}"
)

# Print setup success message
print("LangChain environment set up successfully with the latest version.")
