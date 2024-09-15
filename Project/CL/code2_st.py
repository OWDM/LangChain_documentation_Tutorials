import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import os
from deep_translator import GoogleTranslator

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Streamlit app title
st.title("News Summarization and Verification System")

# Function to translate text to Arabic
def translate_to_arabic(text):
    try:
        translator = GoogleTranslator(source='auto', target='ar')
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

# The rest of your code remains the same...

# Step 1: Create the summarization chain (in English)
summarize_template = """
Summarize the following news article in English, adhering to these guidelines:
1. Title: Start with a noun or an entity relevant to the news.
2. Summary:
   - First Sentence: Describe what was developed or achieved, mentioning specific names and their nationalities.
   - Second Sentence: Briefly explain the functionality or purpose of the development.
   - Third Sentence: Mention key results or findings.
   - Fourth Sentence: Provide any future plans or goals related to the development.
3. Keep the summary under 85 words, excluding the title.
4. Focus on clarity and conciseness.

News Article:
{article}

Summary:
"""

summarize_prompt = PromptTemplate(input_variables=["article"], template=summarize_template)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt, output_key="summary")

# Step 2: Create the verification chain
verify_template = """
Verify the accuracy of the following summary based on the original news article. 
If there are any inaccuracies, please point them out and suggest corrections.

Original Article:
{article}

Summary:
{summary}

Verification:
"""

verify_prompt = PromptTemplate(input_variables=["article", "summary"], template=verify_template)
verify_chain = LLMChain(llm=llm, prompt=verify_prompt, output_key="verification")

# Step 3: Create the overall chain
overall_chain = SequentialChain(
    chains=[summarize_chain, verify_chain],
    input_variables=["article"],
    output_variables=["summary", "verification"],
    verbose=True
)

def generate_verify_and_translate_summary(article):
    max_attempts = 3
    best_summary = None
    best_verification = None
    
    for attempt in range(max_attempts):
        with get_openai_callback() as cb:
            result = overall_chain({"article": article})
        
        summary = result["summary"]
        verification = result["verification"]
        
        st.write(f"\n{'='*50}")
        st.write(f"Attempt {attempt + 1}:")
        st.write(f"{'='*50}")
        st.write("English Summary:")
        st.write(f"{'='*50}")
        st.write(summary)
        st.write(f"\n{'='*50}")
        st.write("Verification:")
        st.write(f"{'='*50}")
        st.write(verification)
        st.write(f"\n{'='*50}")
        st.write(f"Total Tokens: {cb.total_tokens}")
        st.write(f"Total Cost (USD): ${cb.total_cost:.4f}")
        st.write(f"{'='*50}")
        
        if "inaccuracies" not in verification.lower() and "incorrect" not in verification.lower():
            st.write("\nSummary verified successfully!")
            best_summary = summary
            best_verification = verification
            break
        else:
            st.write("\nRefining summary based on verification feedback...\n")
            if best_summary is None or len(verification) < len(best_verification):
                best_summary = summary
                best_verification = verification
    
    if best_summary is None:
        st.write("Max attempts reached. Using the last generated summary.")
        best_summary = summary

    # Translate the best summary to Arabic
    st.write("\nTranslating to Arabic...")
    arabic_summary = translate_to_arabic(best_summary)
    if arabic_summary:
        st.write("Arabic translation completed.")
    else:
        st.write("Arabic translation failed.")
    
    return best_summary, arabic_summary

# Streamlit input
# Streamlit input
article = st.text_area("Enter the news article:", height=300)

if st.button("Generate Summary"):
    if article:
        final_english_summary, final_arabic_summary = generate_verify_and_translate_summary(article)

        # Display the English summary
        st.markdown("## English Summary")
        st.text_area("", value=final_english_summary, height=200, disabled=True)

        if final_arabic_summary:
            # Display the Arabic summary with right alignment
            st.markdown("## Arabic Summary")
            st.markdown(
                f"""
                <div dir="rtl" lang="ar" style="text-align: right; direction: rtl; unicode-bidi: embed;">
                    <textarea readonly style="width: 100%; height: 200px; direction: rtl; text-align: right; unicode-bidi: embed;">
                        {final_arabic_summary}
                    </textarea>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("Failed to generate Arabic summary.")
    else:
        st.write("Please enter a news article.")