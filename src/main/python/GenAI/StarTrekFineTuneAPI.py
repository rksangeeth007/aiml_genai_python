import os
from openai import OpenAI

client = OpenAI()
#Using the files DATA_eval.jsonl, DATA_lines.jsonl & DATA_train.jsonl created as part of OpenAIPreValidationNewAPI.py script
client.files.create(
    file=open("./DATA_train.jsonl", "rb"),
    purpose='fine-tune'
)

client.files.create(
    file=open("./DATA_eval.jsonl", "rb"),
    purpose='fine-tune'
)

client.files.retrieve("file-UqPVnkk9z8Q74BEUqPlnhjHL")

client.fine_tuning.jobs.create(training_file="file-9lI2ovFA1UJskgOPpxDTwEhG", validation_file="file-UqPVnkk9z8Q74BEUqPlnhjHL", model="gpt-3.5-turbo")

client.fine_tuning.jobs.retrieve("ftjob-mQlhbPB5vsog1SeDLNx2xAMj")

client.fine_tuning.jobs.list_events(id="ftjob-mQlhbPB5vsog1SeDLNx2xAMj", limit=10)

completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "Data is an android in the TV series Star Trek: The Next Generation."},
    {"role": "user", "content": "PICARD: Mr. Data, scan for lifeforms."}
]
)

print(completion.choices[0].message)

completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-0613:sundog-software-llc::7qiBf2gI",
    messages=[
        {"role": "system", "content": "Data is an android in the TV series Star Trek: The Next Generation."},
        {"role": "user", "content": "PICARD: Mr. Data, scan for lifeforms."}
    ]
)

print(completion.choices[0].message)

