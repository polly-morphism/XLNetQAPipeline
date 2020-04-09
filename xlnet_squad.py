from pipeline.pipelines import pipeline
from transformers import XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer

config_class, model_class, tokenizer_class = (
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
)

model_name_or_path = "ahotrod/xlnet_large_squad2_512"
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

pipeline = pipeline(
    task="question-answering",
    model="ahotrod/xlnet_large_squad2_512",
    tokenizer="ahotrod/xlnet_large_squad2_512",
)

q = "Where is Kiev?"
c = [
    "Kiev is the capital and most populous city of Ukraine. Kiev is in Ukraine. It is in north-central Ukraine along the Dnieper River. Its population in July 2015 was 2,887,974 though higher estimated numbers have been cited in the press), making Kiev the sixth-most populous city in Europe. Kiev is an important industrial, scientific, educational and cultural center of Eastern Europe. It is home to many high-tech industries, higher education institutions, and historical landmarks. The city has an extensive system of public transport and infrastructure, including the Kiev Metro.",
    "Kiev, Ukraine, is a city on the brink of something magical. Plagued by its recent history and even its current events in the eastern part of Ukraine, Kiev is going through its adolescent years and the results are nothing short of inspiring. Whether you are into religious history, hip neighborhoods, or brutal architecture, Kiev will surprise you and leave you pining to book a return trip to witness the Ukrainian capital’s newest developments.",
]
output = pipeline(question=q, context=c)
print(output)
