import nltk
import random


def load_training():
    file = open('/data/kaggle/movie_reviews/data/train.tsv')
    sentences = []
    file.next() # Ignore header
    for l in file:
        phraseid, sentenceid, phrase, sentiment = l.strip().split('\t')
        sentences.append((phrase.split(' '), sentiment))

    return sentences


def get_words_in_comments(comments):
    all_words = []
    for words, sent in comments:
        all_words.extend(words)
    return all_words


def get_word_features(words):
    w = nltk.FreqDist(words)
    word_features = w.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    features['comment len'] = len(document)
    #for word in document:
    #    features[word] = True
    return features


def load_submission_data():
    testfile = open('/data/kaggle/movie_reviews/data/test.tsv')
    test_lines = []

    # Ignore Heading
    testfile.next()

    for l in testfile:
        phraseid, sentenceid, phrase = l.strip().split('\t')
        test_lines.append((phraseid, phrase))

    return test_lines


def write_submission(array):
    filename = '/data/kaggle/movie_reviews/data/submission.csv'
    print "Writing output to filename: " + filename

    outputfile = open(filename, 'w')
    outputfile.write("PhraseId,Sentiment\n")

    for el in array:
        outputfile.write(str(el[0]) + "," + str(el[1]) + "\n")

    outputfile.close()



sentences = load_training()
words = get_words_in_comments(sentences)
word_features = get_word_features(words)[:20000]

print len(words)


print sentences[:10]


training_size = 100000
training_index = random.sample(range(len(sentences)), training_size)
test_index = []
for i in range(len(sentences)):
    if i not in training_index:
        test_index.append(i)

training_sentences = [sentences[i] for i in training_index]
test_sentences = [sentences[i] for i in test_index]

print "---" 
print training_sentences[:10]



training_set = nltk.classify.apply_features(extract_features, training_sentences)
classifier = nltk.NaiveBayesClassifier.train(training_set)

print classifier.show_most_informative_features(32)


# Check test_sentences
predicted_and_real = []
for w, sentiment in test_sentences[:1000]:
    predicted_and_real.append((sentiment, classifier.classify(extract_features(w))))


prediction_matches = 0
total = 0

for w in predicted_and_real:
    if w[0] == w[1]:
        prediction_matches += 1
    total += 1

print "Prediction matches = " + str(prediction_matches) + " / Total = " + str(total)


test = load_submission_data()


dict_training = {}
for words, sent in sentences:
    dict_training[frozenset(words)] = sent


i = 0
founds = 0
test_predicted = []
for phraseid, phrase in test:
    lkp = dict_training.get(frozenset(phrase.split()))
    if lkp:
        test_predicted.append((phraseid, sent))
        founds += 1
    else:
        test_predicted.append((phraseid, classifier.classify(extract_features(phrase.split(' ')))))
    i += 1
    if i % 100 == 0:
        print str(founds) + '/' + str(i)



write_submission(test_predicted)
