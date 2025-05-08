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


process_directory("./sample_data/tng", 'DATA')
print(dialogues[0])

from docarray import BaseDoc
from docarray.typing import NdArray

embedding_dimensions = 128


class DialogLine(BaseDoc):
    text: str = ''
    embedding: NdArray[embedding_dimensions]


from google.colab import userdata
from openai import OpenAI

client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

embedding_model = "text-embedding-3-small"

response = client.embeddings.create(
    input=dialogues[1],
    dimensions=embedding_dimensions,
    model=embedding_model
)

print(response.data[0].embedding)
print(len(response.data[0].embedding))

# Generate embeddings for everything Data ever said, 128 lines at a time.
embeddings = []

for i in range(0, len(dialogues), 128):
    dialog_slice = dialogues[i:i + 128]
    slice_embeddings = client.embeddings.create(
        input=dialog_slice,
        dimensions=embedding_dimensions,
        model=embedding_model
    )

embeddings.extend(slice_embeddings.data)
print(len(embeddings))

from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB

# Specify your workspace path
db = InMemoryExactNNVectorDB[DialogLine](workspace='./sample_data/workspace')

# Index our list of documents
doc_list = [DialogLine(text=dialogues[i], embedding=embeddings[i].embedding) for i in range(len(embeddings))]
db.index(inputs=DocList[DialogLine](doc_list))

# Perform a search query
queryText = 'Lal, my daughter'
response = client.embeddings.create(
    input=queryText,
    dimensions=embedding_dimensions,
    model=embedding_model
)
query = DialogLine(text=queryText, embedding=response.data[0].embedding)
results = db.search(inputs=DocList[DialogLine]([query]), limit=10)

for m in results[0].matches:
    print(m)

import openai


def generate_response(question):
    # Search for similar dialogues in the vector DB
    response = client.embeddings.create(
        input=question,
        dimensions=embedding_dimensions,
        model=embedding_model
    )

    query = DialogLine(text=queryText, embedding=response.data[0].embedding)
    results = db.search(inputs=DocList[DialogLine]([query]), limit=10)

    # Extract relevant context from search results
    context = "\n"
    for result in results[0].matches:
        context += "\"" + result.text + "\"\n"


#    context = '/n'.join([result.text for result in results[0].matches])


    prompt = f"Lt. Commander Data is asked: '{question}'. How might Data respond? Take into account Data's previous responses similar to this topic, listed here: {context}"
    print("PROMPT with RAG:\n")
    print(prompt)

    print("\nRESPONSE:\n")
    # Use OpenAI API to generate a response based on the context
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are Lt. Cmdr. Data from Star Trek: The Next Generation."},
            {"role": "user", "content": prompt}
        ]
    )

    return (completion.choices[0].message.content)

print(generate_response("Tell me about your daughter, Lal."))
