from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')

generator("This movie seemed really long.", max_length=300, num_return_sequences=5)

generator("Star Trek" , max_length=100, num_return_sequences=5)


# GPT2-Large model (812M parameters)
generator = pipeline('text-generation', model='gpt2-large')
generator("I read a good novel.", max_length=30, num_return_sequences=5)
