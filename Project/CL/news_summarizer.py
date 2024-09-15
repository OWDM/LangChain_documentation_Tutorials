from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import os
import io

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

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

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)
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

# Step 3: Create the translation chain
translate_template = """
Translate the following English summary to Arabic. Maintain the structure and formatting of the original summary.

English Summary:
{summary}

Arabic Translation:
"""

translate_prompt = PromptTemplate(input_variables=["summary"], template=translate_template)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="arabic_summary")

# Step 4: Create the overall chain
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
        
        print(f"\n{'='*50}")
        print(f"Attempt {attempt + 1}:")
        print(f"{'='*50}")
        print("English Summary:")
        print(f"{'='*50}")
        print(summary)
        print(f"\n{'='*50}")
        print("Verification:")
        print(f"{'='*50}")
        print(verification)
        print(f"\n{'='*50}")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost:.4f}")
        print(f"{'='*50}")
        
        if "inaccuracies" not in verification.lower() and "incorrect" not in verification.lower():
            print("\nSummary verified successfully!")
            best_summary = summary
            best_verification = verification
            break
        else:
            print("\nRefining summary based on verification feedback...\n")
            if best_summary is None or len(verification) < len(best_verification):
                best_summary = summary
                best_verification = verification
    
    if best_summary is None:
        print("Max attempts reached. Using the last generated summary.")
        best_summary = summary

    # Translate the best summary to Arabic
    with get_openai_callback() as cb:
        arabic_result = translate_chain({"summary": best_summary})
    
    arabic_summary = arabic_result["arabic_summary"]
    print("\nArabic translation completed.")
    print(f"\n{'='*50}")
    print(f"Translation Tokens: {cb.total_tokens}")
    print(f"Translation Cost (USD): ${cb.total_cost:.4f}")
    print(f"{'='*50}")
    
    return best_summary, arabic_summary

def save_arabic_to_file(arabic_text, filename="arabic_summary.txt"):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(arabic_text)
    print(f"Arabic summary saved to {filename}")

# Example usage
article = """
MIT researchers use large language models to flag problems in complex systems
The approach can detect anomalies in data recorded over time, without the need for any training.

Identifying one faulty turbine in a wind farm, which can involve looking at hundreds of signals and millions of data points, is akin to finding a needle in a haystack.

Engineers often streamline this complex problem using deep-learning models that can detect anomalies in measurements taken repeatedly over time by each turbine, known as time-series data.

But with hundreds of wind turbines recording dozens of signals each hour, training a deep-learning model to analyze time-series data is costly and cumbersome. This is compounded by the fact that the model may need to be retrained after deployment, and wind farm operators may lack the necessary machine-learning expertise.

In a new study, MIT researchers found that large language models (LLMs) hold the potential to be more efficient anomaly detectors for time-series data. Importantly, these pretrained models can be deployed right out of the box.

The researchers developed a framework, called SigLLM, which includes a component that converts time-series data into text-based inputs an LLM can process. A user can feed these prepared data to the model and ask it to start identifying anomalies. The LLM can also be used to forecast future time-series data points as part of an anomaly detection pipeline.

While LLMs could not beat state-of-the-art deep learning models at anomaly detection, they did perform as well as some other AI approaches. If researchers can improve the performance of LLMs, this framework could help technicians flag potential problems in equipment like heavy machinery or satellites before they occur, without the need to train an expensive deep-learning model.

“Since this is just the first iteration, we didn’t expect to get there from the first go, but these results show that there’s an opportunity here to leverage LLMs for complex anomaly detection tasks,” says Sarah Alnegheimish, an electrical engineering and computer science (EECS) graduate student and lead author of a paper on SigLLM.

Her co-authors include Linh Nguyen, an EECS graduate student; Laure Berti-Equille, a research director at the French National Research Institute for Sustainable Development; and senior author Kalyan Veeramachaneni, a principal research scientist in the Laboratory for Information and Decision Systems. The research will be presented at the IEEE Conference on Data Science and Advanced Analytics.

An off-the-shelf solution

Large language models are autoregressive, which means they can understand that the newest values in sequential data depend on previous values. For instance, models like GPT-4 can predict the next word in a sentence using the words that precede it.

Since time-series data are sequential, the researchers thought the autoregressive nature of LLMs might make them well-suited for detecting anomalies in this type of data.

However, they wanted to develop a technique that avoids fine-tuning, a process in which engineers retrain a general-purpose LLM on a small amount of task-specific data to make it an expert at one task. Instead, the researchers deploy an LLM off the shelf, with no additional training steps.

But before they could deploy it, they had to convert time-series data into text-based inputs the language model could handle.

They accomplished this through a sequence of transformations that capture the most important parts of the time series while representing data with the fewest number of tokens. Tokens are the basic inputs for an LLM, and more tokens require more computation.

“If you don’t handle these steps very carefully, you might end up chopping off some part of your data that does matter, losing that information,” Alnegheimish says.

Once they had figured out how to transform time-series data, the researchers developed two anomaly detection approaches.

Approaches for anomaly detection

For the first, which they call Prompter, they feed the prepared data into the model and prompt it to locate anomalous values.

“We had to iterate a number of times to figure out the right prompts for one specific time series. It is not easy to understand how these LLMs ingest and process the data,” Alnegheimish adds.

For the second approach, called Detector, they use the LLM as a forecaster to predict the next value from a time series. The researchers compare the predicted value to the actual value. A large discrepancy suggests that the real value is likely an anomaly.

With Detector, the LLM would be part of an anomaly detection pipeline, while Prompter would complete the task on its own. In practice, Detector performed better than Prompter, which generated many false positives.

“I think, with the Prompter approach, we were asking the LLM to jump through too many hoops. We were giving it a harder problem to solve,” says Veeramachaneni.

When they compared both approaches to current techniques, Detector outperformed transformer-based AI models on seven of the 11 datasets they evaluated, even though the LLM required no training or fine-tuning.

In the future, an LLM may also be able to provide plain language explanations with its predictions, so an operator could be better able to understand why an LLM identified a certain data point as anomalous.

However, state-of-the-art deep learning models outperformed LLMs by a wide margin, showing that there is still work to do before an LLM could be used for anomaly detection.

“What will it take to get to the point where it is doing as well as these state-of-the-art models? That is the million-dollar question staring at us right now. An LLM-based anomaly detector needs to be a game-changer for us to justify this sort of effort,” Veeramachaneni says.

Moving forward, the researchers want to see if finetuning can improve performance, though that would require additional time, cost, and expertise for training.

Their LLM approaches also take between 30 minutes and two hours to produce results, so increasing the speed is a key area of future work. The researchers also want to probe LLMs to understand how they perform anomaly detection, in the hopes of finding a way to boost their performance.

“When it comes to complex tasks like anomaly detection in time series, LLMs really are a contender. Maybe other complex tasks can be addressed with LLMs, as well?” says Alnegheimish.

This research was supported by SES S.A., Iberdrola and ScottishPower Renewables, and Hyundai Motor Company.


"""

final_english_summary, final_arabic_summary = generate_verify_and_translate_summary(article)

print(f"\n{'='*50}")
print("Final English Summary:")
print(f"{'='*50}")
print(final_english_summary)

if final_arabic_summary:
    save_arabic_to_file(final_arabic_summary)
    print("Arabic summary has been saved to 'arabic_summary.txt'")
else:
    print("Failed to generate Arabic summary.")