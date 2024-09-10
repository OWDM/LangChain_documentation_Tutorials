# summarization_chain.py

from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize the LLM with the correct import
llm = OpenAI(api_key="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")  # Replace with your actual API key


# Define the prompt template for stricter verification
verification_prompt = PromptTemplate(
    input_variables=["news_text", "summary"],
    template="""
    Verify the accuracy and completeness of the following Arabic news summary based on the original news report.

    Original News: {news_text}

    Summary: {summary}

    Check if the summary accurately reflects the main points of the original news, and ensure it meets the following criteria:
    1. The summary must contain a title that begins with a noun or entity relevant to the news.
    2. The summary should have exactly four sentences, each serving a specific purpose as follows:
       - First Sentence: Describes what was developed or achieved, including specific names and their nationalities.
       - Second Sentence: Explains the functionality or purpose of the development.
       - Third Sentence: Mentions the key results or findings.
       - Fourth Sentence: Provides any future plans or goals related to the development.
    3. The summary should be concise, clear, and not exceed 85 words.
    
    If any of these criteria are not met, list the specific issues with detailed feedback on what is wrong and how it can be improved. If the summary meets all criteria, state that it is accurate and complete.
    """
)

# Create a chain that uses the LLM and the enhanced verification prompt template
verification_chain = LLMChain(
    llm=llm,
    prompt=verification_prompt
)

def verify_summary(news_text, summary):
    """
    Function to verify the accuracy and completeness of the generated summary against the original news text.

    :param news_text: The input news report text.
    :param summary: The generated summary text.
    :return: Feedback on the summary's accuracy and completeness.
    """
    # Run the verification chain with the news text and summary
    result = verification_chain.run({"news_text": news_text, "summary": summary})
    return result

# Test verification with a sample input
input_file_path = 'D:\\owd1\\Documents\\GitHub-REPO\\LangChain_documentation_Tutorials\\Project\\article.txt'  # Path to the input file
output_file_path = 'D:\\owd1\\Documents\\GitHub-REPO\\LangChain_documentation_Tutorials\\summary_output.txt'  # Path to the output file

# Function to read input from file
def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to write verification feedback to a file
def write_verification_feedback(output_path, feedback):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write("Verification Feedback:\n")
        file.write(feedback)

# Read the input news text and summary
news_text = read_input_file(input_file_path)
summary = read_input_file(output_file_path)

# Verify the summary
verification_feedback = verify_summary(news_text, summary)

# Print and save the verification feedback
print("Verification Feedback:")
print(verification_feedback)

# Save feedback to a file
verification_feedback_path = 'verification_feedback.txt'  # Path to save the feedback
write_verification_feedback(verification_feedback_path, verification_feedback)

print(f"Verification feedback has been saved to {verification_feedback_path}.")