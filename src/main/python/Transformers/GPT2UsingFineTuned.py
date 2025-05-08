from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel

dir = "/tmp/test-clm"
generator = pipeline('text-generation', model=GPT2LMHeadModel.from_pretrained(dir), tokenizer=GPT2Tokenizer.from_pretrained(dir))

generator("I read a good novel.", max_length=30, num_return_sequences=5)

generator("This movie seemed really long.", max_length=300, num_return_sequences=5)

generator("Star Trek" , max_length=100, num_return_sequences=5)
