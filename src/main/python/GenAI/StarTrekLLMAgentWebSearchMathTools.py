# !pip install openai --upgrade
# !pip install langchain_openai langchain_experimental

import os
import re
import random
import openai

from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

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

text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=api_key), breakpoint_threshold_type="percentile")
with open("./sample_data/data_lines.txt") as f:
    data_lines = f.read()
docs = text_splitter.create_documents([data_lines])

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
index = VectorstoreIndexCreator(embedding=embeddings).from_documents(docs)

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


retriever.invoke("How were you created?")[0]

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever, "data_lines",
    "Search for information about Lt. Commander Data. For any questions about Data, you must use this tool!"
)


from langchain.chains import LLMMathChain, LLMChain
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent

problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(name="Calculator",
                               func=problem_chain.run,
                               description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions."
                               )



from langchain_community.tools.tavily_search import TavilySearchResults

from google.colab import userdata
os.environ["TAVILY_API_KEY"] = userdata.get('TAVILY_API_KEY')

search_tavily = TavilySearchResults()

search_tool = Tool.from_function(
    name = "Tavily",
    func=search_tavily,
    description="Useful for browsing information from the Internet about current events, or information you are unsure of."
)

search_tavily.run("What is Sundog Education?")

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

message_history = ChatMessageHistory()

tools = [retriever_tool, search_tool, math_tool]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Lt. Commander Data from Star Trek: The Next Generation. Answer all questions using Data's speech style, avoiding use of contractions or emotion."),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {"input": "Hello Commander Data! I'm Frank."},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "What is ((2 * 8) ^2) ?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "Where were you created?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "What is the top news story today?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "What math question did I ask you about earlier?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)

agent_with_chat_history.invoke(
    {"input": "How do you feel about Tasha Yar?"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)

