import os
from typing import Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.schema import SystemMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set API key
os.environ["OPENAI_API_KEY"] = "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"

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_vectorstore(text: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

def extract_key_info(article: str) -> Dict[str, str]:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    vectorstore = create_vectorstore(article)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    questions = [
        "What are the main technical concepts discussed in this article?",
        "What are the key findings or advancements mentioned?",
        "Are there any specific companies or researchers mentioned?",
        "What potential impacts or applications are discussed?"
    ]

    return {question: qa_chain.run(question) for question in questions}

def generate_summary(article: str, key_info: Dict[str, str]) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert in summarizing technical news articles. Your task is to create a concise and structured summary of the given article, focusing on the key technical information."),
        HumanMessagePromptTemplate.from_template("""
Article: {article}

Key Information:
{key_info}

Please provide a structured summary of this article, following these guidelines:

1. List all statistics mentioned in the article with their descriptions.
2. Provide a structured summary of the article using the following guidelines:
   a. Start with a title that begins with a noun or an entity relevant to the news.
   b. Keep the summary under 85 words, excluding the title.
""")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(article=article, key_info=str(key_info))

def translate_to_arabic(text: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an expert translator specializing in technical translations from English to Arabic. Your task is to provide an accurate and fluent translation that preserves the technical nuances and structure of the original text."),
        HumanMessagePromptTemplate.from_template("""
Translate the following text to Arabic, but transliterate all proper nouns and brand names, ensuring they are written in Arabic script based on their pronunciation, without translating their meaning. If you do, please mention the English noun or brand between two brackets for example جوجل (Google)

{text}

Please ensure that the translation follows the same format as the original, with a title on the first line, followed by the summary. Do not add any labels in Arabic for "Title" or "Summary".
""")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text=text)

def save_to_file(content: str, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def main():
    try:
        input_path = r"D:\owd1\Documents\GitHub-REPO\LangChain_documentation_Tutorials\Project\CL\articles\article_3.txt"
        output_path = r"D:\owd1\Documents\GitHub-REPO\LangChain_documentation_Tutorials\Project\CL\output\arabic_summary.txt"

        article = read_file(input_path)
        key_info = extract_key_info(article)
        summary = generate_summary(article, key_info)
        arabic_summary = translate_to_arabic(summary)
        
        save_to_file(arabic_summary, output_path)
        print(f"Arabic summary has been saved to: {output_path}")

        print("\nOriginal Article:")
        print(article)
        print("\nExtracted Key Information:")
        for question, answer in key_info.items():
            print(f"{question}\n{answer}\n")
        print("\nStructured English Summary:")
        print(summary)
        print("\nArabic Translation:")
        print(arabic_summary)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the input file exists and the path is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()