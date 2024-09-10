# "D:\\owd1\\Documents\\GitHub-REPO\\LangChain_documentation_Tutorials\\Project\\article.txt"




# structured_feedback_loop_corrected.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize your API key here
api_key = "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"  # Replace with your actual GPT-4 API key

# Initialize the GPT-4 chat model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# Define the prompt templates for different validation steps
title_check_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
    Check if the summary contains a title that begins with a noun or entity relevant to the news. If missing, suggest a suitable title based on the content.
    
    Summary: {summary}
    """
)

sentence_structure_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
    Check the sentence structure of the summary to ensure it has exactly four sentences:
    1. Describes what was developed or achieved, including names and nationalities.
    2. Explains the functionality or purpose of the development.
    3. Mentions the key results or findings.
    4. Provides any future plans or goals related to the development.

    Identify which sentences are missing or incorrect and suggest specific corrections.

    Summary: {summary}
    """
)

accuracy_check_prompt = PromptTemplate(
    input_variables=["summary", "news_text"],
    template="""
    Verify the accuracy of the summary against the original news text. Highlight inaccuracies or missing critical information.

    News Text: {news_text}

    Summary: {summary}
    """
)

conciseness_check_prompt = PromptTemplate(
    input_variables=["summary"],
    template="""
    Check the length of the summary to ensure it does not exceed 85 words. If it does, suggest specific edits to reduce the word count while retaining key information.

    Summary: {summary}
    """
)

def validate_title(summary):
    """Validates and suggests corrections for the title."""
    formatted_prompt = title_check_prompt.format(summary=summary)
    feedback = llm.invoke(formatted_prompt)
    logging.info(f"Title Check Feedback: {feedback}")
    return feedback

def validate_sentence_structure(summary):
    """Validates and suggests corrections for sentence structure."""
    formatted_prompt = sentence_structure_prompt.format(summary=summary)
    feedback = llm.invoke(formatted_prompt)
    logging.info(f"Sentence Structure Feedback: {feedback}")
    return feedback

def validate_accuracy(news_text, summary):
    """Validates and suggests corrections for accuracy."""
    formatted_prompt = accuracy_check_prompt.format(news_text=news_text, summary=summary)
    feedback = llm.invoke(formatted_prompt)
    logging.info(f"Accuracy Feedback: {feedback}")
    return feedback

def validate_conciseness(summary):
    """Validates and suggests corrections for conciseness."""
    formatted_prompt = conciseness_check_prompt.format(summary=summary)
    feedback = llm.invoke(formatted_prompt)
    logging.info(f"Conciseness Feedback: {feedback}")
    return feedback

def apply_feedback(summary, feedback):
    """
    Applies targeted feedback to the summary without rewriting it completely.

    :param summary: The current summary to be refined.
    :param feedback: Feedback containing specific corrections.
    :return: The refined summary with targeted edits.
    """
    # Simplified example of applying feedback, needs adjustment based on feedback format
    if "missing title" in feedback.lower():
        title_suggestion = feedback.split("suggested title:")[-1].strip()
        summary = title_suggestion + "\n" + summary
    
    # More detailed parsing can be added here based on feedback content
    # Apply direct edits instead of broad changes
    return summary

def refine_summary(news_text, initial_summary, max_iterations=5):
    """
    Validates and refines the summary step-by-step using a structured approach.

    :param news_text: The input news report text.
    :param initial_summary: The initial generated summary.
    :param max_iterations: Maximum number of iterations for refinement.
    :return: The refined summary after validation steps.
    """
    summary = initial_summary
    for iteration in range(max_iterations):
        logging.info(f"Iteration {iteration + 1} - Current Summary: {summary}")

        # Validate title
        title_feedback = validate_title(summary)
        summary = apply_feedback(summary, title_feedback)

        # Validate sentence structure
        structure_feedback = validate_sentence_structure(summary)
        summary = apply_feedback(summary, structure_feedback)

        # Validate accuracy
        accuracy_feedback = validate_accuracy(news_text, summary)
        summary = apply_feedback(summary, accuracy_feedback)

        # Validate conciseness
        conciseness_feedback = validate_conciseness(summary)
        summary = apply_feedback(summary, conciseness_feedback)

        # Log the refined summary
        logging.info(f"Refined Summary After Iteration {iteration + 1}: {summary}")

        # Check if the summary meets all criteria
        if "meets all criteria" in accuracy_feedback.lower():
            logging.info("Summary meets all criteria. Stopping refinement.")
            break

    return summary

# Function to read input from file
def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Main flow
input_file_path = "D:\\owd1\\Documents\\GitHub-REPO\\LangChain_documentation_Tutorials\\Project\\article.txt"  # Path to the input file
news_text = read_input_file(input_file_path)

# Generate the initial summary (simplified placeholder step, integrate with your summarization method)
initial_summary = "Initial generated summary based on the news text."

# Validate and refine the summary using the structured approach
final_summary = refine_summary(news_text, initial_summary)

# Save the final refined summary
output_file_path = 'final_summary.txt'
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("Final Summary:\n")
    file.write(final_summary)

logging.info(f"Final summary has been saved to {output_file_path}.")
