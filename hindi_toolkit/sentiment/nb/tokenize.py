import string

stopword_list = []
punct_list = []

def initialize():
	global stopword_list
	global punct_list	

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def tokenize(line):
	initialize()

	#ignore ID
	text = line.translate(None, string.punctuation).split()
	tokens = []
	for ele in text:
		tokens.append(ele)

	#return list of tokens
	return tokens

def main():
	#initialize lists
	initialize()

	#sample
	line = "ID Hello, I am Nikhila!!!"
	tokens = tokenize(line)
	print(tokens)

if __name__ == '__main__':
	main()