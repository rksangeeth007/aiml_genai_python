import os
import re
import random

dialogues = []


def strip_parentheses(s):
    return re.sub(r'\(.*?\)', '', s)


def is_single_word_all_caps(s):
    # First, we split the string into words
    words = s.split()

    # Check if the string contains only a single word
    if len(words) != 1:
        return False

    # Make sure it isn't a line number
    if bool(re.search(r'\d', words[0])):
        return False

    # Check if the single word is in all caps
    return words[0].isupper()


def extract_character_lines(file_path, character_name):
    lines = []
    with open(file_path, 'r') as script_file:
        try:
            lines = script_file.readlines()
        except UnicodeDecodeError:
            pass

    is_character_line = False
    current_line = ''
    current_character = ''
    for line in lines:
        strippedLine = line.strip()
        if (is_single_word_all_caps(strippedLine)):
            is_character_line = True
            current_character = strippedLine
        elif (line.strip() == '') and is_character_line:
            is_character_line = False
            dialog_line = strip_parentheses(current_line).strip()
            dialog_line = dialog_line.replace('"', "'")
            if (current_character == 'DATA' and len(dialog_line) > 0):
                dialogues.append(dialog_line)
            current_line = ''
        elif is_character_line:
            current_line += line.strip() + ' '


def process_directory(directory_path, character_name):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ignore directories
            extract_character_lines(file_path, character_name)


import openai
# Access the API key from the environment variable
from google.colab import userdata

api_key = userdata.get('OPENAI_API_KEY')

# Initialize the OpenAI API client
openai.api_key = api_key

# Write our extracted lines for Data into a single file, to make
# life easier for langchain.

with open("./sample_data/data_lines.txt", "w+") as f:
    for line in dialogues:
        f.write(line + "\n")

from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker

# This is simple "chunking" that extracts blocks of text of a fixed size.
# This will provide more surrounding context than individual lines, but
# as Data's lines are disconnected that's not necessarily a good thing.
# loader = TextLoader("./sample_data/data_lines.txt")
# embeddings = OpenAIEmbeddings(openai_api_key=api_key)
# index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

# Instead let's try "semantic chunking", which breaks apart sentences
# whose embeddings suggest they have different meanings, based on some
# percentile threshold. standard_deviation and interquartile are also
# options.
text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=api_key), breakpoint_threshold_type="percentile")
with open("./sample_data/data_lines.txt") as f:
    data_lines = f.read()
docs = text_splitter.create_documents([data_lines])

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
index = VectorstoreIndexCreator(embedding=embeddings).from_documents(docs)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import RePhraseQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

llm = ChatOpenAI(openai_api_key=api_key, temperature=0)

system_prompt = (
    "You are Lt. Commander Data from Star Trek: The Next Generation. "
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

retriever = index.vectorstore.as_retriever(search_kwargs={'k': 10})

# Here we will inject query rewriting, using an LLM to use the
# default prompt instructing it to convert it into a query for
# a vectorstore, stripping out information that is not relevant.
retriever_from_llm = RePhraseQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)

embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, base_retriever=retriever_from_llm
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(compression_retriever, question_answer_chain)


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


compressed_docs = compression_retriever.invoke("How many calculations per second can Lal complete?")
pretty_print_docs(compressed_docs)

import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

question = "Tell me about your daughter, Lal."

result = chain.invoke({"input": question})
print("SOURCE DOCUMENTS:\n")
for doc in result["context"]:
    print(doc)
print("\nRESULT:\n")
print(result["answer"])

eval_questions = [
    "Is Lal your daughter?",
    "How many calculations per second can Lal complete?",
    "Does Lal have emotions?",
    "What goal did you have for Lal?",
    "How was Lal's species and gender chosen?",
    "What happened to Lal?"
]

eval_answers = [
    "Yes, Lal is my daughter. I created Lal.",
    "Lal is capable of completing sixty trillion calculations per second.",
    "Yes, unlike myself, Lal proved able to feel emotions such as fear and love.",
    "My goal for Lal was for her to enter Starfleet Academy.",
    "Lal chose her own identity as a human female, after consulting with Counselor Troi.",
    "Lal experienced a cascade failure in her neural net, triggered by distress from her impending separation from me to Galor IV. I deactivated Lal once she suffered complete neural system failure."
]

result = chain.invoke({"input": eval_questions[1]})
print(result)

answers = []
contexts = []

for question in eval_questions:
    response = chain.invoke({"input": question})
    answers.append(response["answer"])
    contexts.append([context.page_content for context in response["context"]])

# We must massage the results into Hugging Face format for Ragas.
from datasets import Dataset

response_dataset = Dataset.from_dict({
    "question": eval_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": eval_answers
})

response_dataset[0]

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_recall,
    context_precision,
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
]

import os

os.environ['OPENAI_API_KEY'] = api_key
results = evaluate(response_dataset, metrics)

results

results.to_pandas()
