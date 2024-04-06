from transformers import pipeline

#transformers < 4.7.0
#ner = pipeline("ner", grouped_entities=True)

ner = pipeline("ner", aggregation_strategy='simple')

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."

output = ner(sequence)

print(output)