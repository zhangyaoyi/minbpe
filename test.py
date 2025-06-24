from minbpe.regex import RegexTokenizer


tokenizer = RegexTokenizer()

tokenizer.load("models/regex.model")

print(tokenizer.encode("Hello world!"))


print(tokenizer.decode([1581]))

