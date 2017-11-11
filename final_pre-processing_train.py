import csv
import nltk
import string
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import *
import pickle

def get_tokens(text):
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed



stemmer = PorterStemmer()
vocabulary = {}
classes = {}

#pkl_file = open('../savedVocabulary.txt', 'rb')
#vocabulary = pickle.load(pkl_file)

#pkl_file = open('../savedClasses.txt', 'rb')
#classes = pickle.load(pkl_file)


vocabulary_count = 0
classes_count = 0
with open("../data/full_train.txt") as f:
    for line in f:
	parts = line.strip().split("\t")
	classes_found = parts[0].split(",")
	for y in classes_found:
		if y.strip() not in classes:
			classes[y.strip()] = classes_count
			classes_count += 1
	parts[1] = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',"",parts[1],flags=re.MULTILINE)
	tokens = get_tokens(parts[1])
	filtered = [w for w in tokens if not w in stopwords.words('english')]
	stemmed = stem_tokens(filtered, stemmer)
	stemmed = [str(x) for x in stemmed]
	for x in stemmed:
		if x not in vocabulary:
			vocabulary[x] = vocabulary_count
			vocabulary_count += 1

print("Vocabulary Count is " + str(vocabulary_count))
print("Class Count is " + str(classes_count))

with open("../savedVocabulary.txt", "wb") as myFile:
    pickle.dump(vocabulary, myFile)

with open("../savedClasses.txt", "wb") as myFile:
    pickle.dump(classes, myFile)

ptrain = open("../data/full_processed_test.txt","w")

with open("../data/full_test.txt") as f:
    for line in f:
	parts = line.strip().split("\t")
	parts[1] = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',"",parts[1],flags=re.MULTILINE)
	tokens = get_tokens(parts[1])
	filtered = [w for w in tokens if not w in stopwords.words('english')]
	stemmed = stem_tokens(filtered, stemmer)
	stemmed = [str(x) for x in stemmed]
	classes_found = parts[0].split(",")
	newline = ""
	for y in classes_found:
		if y.strip() in classes:
			newline += str(classes[y.strip()]) + ","
		else:
			newline += str(-1) + ","
			print("50th Class is "+ y.strip())
	newline = newline.rstrip(",")
	newline += "\t"   
	for x in stemmed:
		if x in vocabulary:
			newline += str(vocabulary[x]) + ","
		else:
			newline += str(-1) + ","
	newline = newline.rstrip(",")
	newline = newline + "\n"
	ptrain.write(newline)
ptrain.close()
