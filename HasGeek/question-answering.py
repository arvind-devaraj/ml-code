from transformers import pipeline

# Replace this with your own checkpoint
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
			Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a business magnate and investor. He is the founder, CEO and chief engineer of SpaceX; 
            angel investor, CEO and product architect of Tesla, Inc.; owner and CEO of Twitter, Inc.; founder of The Boring Company; co-founder of Neuralink 
            and OpenAI; and president of the philanthropic Musk Foundation. With an estimated net worth of around $175 billion as of February 3, 2023, primarily 
            from his ownership stakes in Tesla and SpaceX,[4][5] Musk is the second-wealthiest person in the world, according to both the Bloomberg Billionaires 
            Index and Forbes's real-time billionaires list.[6][7]
            Musk was born in Pretoria, South Africa, and briefly attended at the University of Pretoria before moving to Canada at age 18, acquiring citizenship 
            through his Canadian-born mother. Two years later, he matriculated at Queen's University and transferred to the University of Pennsylvania, 
            where he received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. After two days, he 
            dropped out and with his brother Kimbal, co-founded the online city guide software company Zip2. In 1999, Zip2 was acquired by Compaq for $307 million 
            and Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal, which eBay acquired for $1.5 billion in 2002.
            With $175.8 million, Musk founded SpaceX in 2002, a spaceflight services company. In 2004, he was an early investor in the electric vehicle manufacturer 
            Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, he helped create 
            SolarCity, a solar energy company that was later acquired by Tesla and became Tesla Energy. In 2015, he co-founded OpenAI, a nonprofit artificial 
            intelligence research company. The following year, he co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and The Boring 
            Company, a tunnel construction company. Musk has also proposed a hyperloop high-speed vactrain transportation system. In 2022, his acquisition of Twitter 
            for $44 billion was completed."""


question = "When was SpaceX founded?"
result=question_answerer(question=question, context=context)
print(result['answer'])

