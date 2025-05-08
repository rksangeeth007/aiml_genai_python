# This needs to be executed in Google Colab and not in Jupyter nitebook or local to avoid any dependency issues
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
            if (current_character == 'DATA' and len(dialog_line)>0):
                dialogues.append(dialog_line)
            current_line = ''
        elif is_character_line:
            current_line += line.strip() + ' '

def process_directory(directory_path, character_name):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ignore directories
            extract_character_lines(file_path, character_name)


process_directory("./sample_data/tng", 'DATA')

print (dialogues[0])

import openai
# Access the API key from the environment variable
from google.colab import userdata

api_key = userdata.get('OPENAI_API_KEY')

openai.api_key = api_key

with open("./sample_data/data_lines.txt", "w+") as f:
    for line in dialogues:
        f.write(line + "\n")


#Source: sample code from langchain docs
from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

loader = CustomDocumentLoader("./sample_data/data_lines.txt")

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

retriever=index.vectorstore.as_retriever(search_kwargs={'k': 10})

question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

question = "Tell me about your daughter, Lal."

result = chain.invoke({"input": question})
print("SOURCE DOCUMENTS:\n")
for doc in result["context"]:
    print(doc)
print("\nRESULT:\n")
print(result["answer"])


#Below is to validate using RAGAs
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

results.to_pandas()
