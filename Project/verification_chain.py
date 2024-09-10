from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

api_key="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"  # Replace with your actual API key

# Initialize the GPT-4 chat model using the correct class
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)




# Define the prompt template for summarization
summarization_prompt = PromptTemplate(
    input_variables=["news_text"],
    template="""
    Write a news summary in Arabic with the following guidelines:

    Title: Begin with a noun or entity relevant to the news.
    Summary:
    - First Sentence: Describe what was developed or achieved, including any specific names and their nationalities.
    - Second Sentence: Explain the functionality or purpose of the development in a brief manner.
    - Third Sentence: Mention the key results or findings.
    - Fourth Sentence: Provide any future plans or goals related to the development.
    - Length: Ensure the summary does not exceed 85 words, excluding the title.
    Make sure to focus on clarity and conciseness.

    News: {news_text}
    """
)

# Create a summarization chain
summarization_chain = LLMChain(
    llm=llm,
    prompt=summarization_prompt
)

# Define an improved verification prompt
verification_prompt = PromptTemplate(
    input_variables=["news_text", "summary"],
    template="""
    You are tasked with verifying the accuracy and completeness of the following Arabic news summary based on the original news report. Assess the summary step-by-step according to each criterion below and provide specific feedback for improvements if necessary.

    Original News: {news_text}

    Summary: {summary}

    1. **Title Check**: Does the summary contain a title that begins with a noun or entity relevant to the news? Provide feedback if missing or incorrect.
    
    2. **Sentence Structure Check**: Check if the summary contains exactly four sentences, each serving the following purposes:
       - **First Sentence**: Describes what was developed or achieved, including specific names and their nationalities.
       - **Second Sentence**: Explains the functionality or purpose of the development.
       - **Third Sentence**: Mentions the key results or findings.
       - **Fourth Sentence**: Provides any future plans or goals related to the development.
       Provide specific feedback for each sentence if it does not meet these requirements.

    3. **Conciseness Check**: Ensure the summary does not exceed 85 words. Provide a revised version if it is too long.

    4. **Completeness Check**: Check if the summary is complete and not cut off. Highlight missing information if any.

    5. **Accuracy Check**: Does the summary accurately reflect the key points of the original news? Highlight inaccuracies.

    6. **Suggestions**: Provide a corrected version of the summary that addresses all issues noted above.

    Your feedback should be structured, concise, and actionable, enabling direct improvements to the summary.
    """
)

# Create a verification chain
verification_chain = LLMChain(
    llm=llm,
    prompt=verification_prompt
)

def summarize_news(news_text):
    """Generates a summary based on the input news text."""
    return summarization_chain.run({"news_text": news_text})

def verify_summary(news_text, summary):
    """Verifies the summary against the original news text and provides feedback."""
    return verification_chain.run({"news_text": news_text, "summary": summary})

def refine_summary(news_text):
    """Refines the summary until it meets the required criteria based on verification feedback."""
    summary = summarize_news(news_text)
    print("Generated Summary:")
    print(summary)
    
    verification_feedback = verify_summary(news_text, summary)
    print("Verification Feedback:")
    print(verification_feedback)
    
    # Check if the summary meets all criteria
    if "meets all criteria" in verification_feedback:
        return summary
    
    # Adjust summarization based on feedback
    print("Refining summary based on feedback...")
    
    # Adjusting prompt to refine based on the feedback from verification
    adjusted_prompt = PromptTemplate(
        input_variables=["news_text", "feedback"],
        template="""
        Based on the feedback provided, refine the summary to ensure it meets all criteria.

        Feedback: {feedback}

        News: {news_text}

        Write a refined summary following the above feedback and the same guidelines as before.
        """
    )
    
    # Create a refinement chain
    refinement_chain = LLMChain(
        llm=llm,
        prompt=adjusted_prompt
    )
    
    # Run the refinement with feedback
    refined_summary = refinement_chain.run({"news_text": news_text, "feedback": verification_feedback})
    print("Refined Summary:")
    print(refined_summary)
    
    return refined_summary

# Read the input news text
input_file_path = 'D:\\owd1\\Documents\\GitHub-REPO\\LangChain_documentation_Tutorials\\Project\\article.txt'  # Path to the input file

# Function to read input from file
def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Read the input news text
news_text = read_input_file(input_file_path)

# Refine the summary until it meets all criteria
final_summary = refine_summary(news_text)

# Save the final summary to a file
output_file_path = 'final_summary.txt'  # Path to save the final summary
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("Final Summary:\n")
    file.write(final_summary)

print(f"Final summary has been saved to {output_file_path}.")