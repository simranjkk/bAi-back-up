from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import aiml
import random
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from transformers import GPT2Tokenizer, GPT2LMHeadModel
ps = PorterStemmer()

# initialized GPT2 code
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
S= random.choice(['What I understood from ', ' I am trying to understand ', 'I could get your point about '])
k = aiml.Kernel()
k.learn("Learning.aiml")
k.respond("LEARN AIML")
#initalised sentiment analyzer
analyser = SentimentIntensityAnalyzer()
ps = PorterStemmer()

while True:
    us = input("User > ")
    # subtracting punctuations
    us = re.sub('[' + string.punctuation + ']', '', us)
    words = word_tokenize(us)
    s = []

    for w in words:
        # filtering synonyms words
        if w == 'workplace':
            w = 'work'
            print(words)
            s.append(ps.stem(w))
        elif w.upper() in ('ACCEPT', 'VALUE', 'VALUED', 'LOVEs','LIKEs', 'LIKED', 'LOVED' , 'likes', 'like', 'loved'):
            w = 'ACCEPT'
            s.append(ps.stem(w))
        elif w in ('REPEAT', 'REPETITIVE'):
            w = 'REPEAT'
            s.append(ps.stem(w))
        elif w in ('mocked', 'MOCKED', 'MOCK', 'MOCKING','mock', 'mocking'):
            w = 'TEASED'
            s.append(ps.stem(w))
        elif w in ('COMMUNITY', 'community'):
            w = 'COMMUNITY'
            s.append(w)
        else:
            s.append(ps.stem(w))
    filter_sentence = (" ").join(s)
    #print(filter_sentence)

    reply = k.respond(filter_sentence)
    #print(filter_sentence)
    #Analyzing sentiments using polarity score
    sentiment_statement = analyser.polarity_scores(us)
    sentiment_results = (sentiment_statement['compound'])
    sent_res_neg = (sentiment_statement['neg'])
    sent_res_pos = (sentiment_statement['neu'])
    sent_res_neu = (sentiment_statement['pos'])
    #print(sentiment_results)
    #print(sent_res_neg)
    #print(sent_res_pos)
    #print(sent_res_neu)

    pos = 0
    neg = 0
    neu = 0

    sntmnt = []
#Conditions for detecting emotions
    if sentiment_results > 0.2:
        pos += 1
        sntmnt.append('Positive')
    elif sentiment_results < -0.2:
        sntmnt.append('Negative')
        neg += 1
    else:
        sntmnt.append('Neutral')
        neu += 1
    print(sntmnt)
    if reply:
        print("beeAI > ", reply)
    else:
        res = re.sub('[' + string.punctuation + ']', '', us)
        #print(str(res),'')
        # add keywrod filter
        #print("bot > Why don't you try this article on Diversity and Inclusion: https://www.strasity.com/?gclid=Cj0KCQjw8p2MBhCiARIsADDUFVG-hEOccIiwlFqX7X3gg0npbNSbEvxzjRJWAqPAOk8ZC4uE-TwzSnkaAm-GEALw_wcB ", )
        text_tokens = word_tokenize(res)
        #condition to remove stop words
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

        #print(tokens_without_sw)
        filtered_sentence = (" ").join(tokens_without_sw)

        S = random.choice(['What I understood from ', ' I am trying to understand ', 'I could get your point about '])
        if(len(filtered_sentence)<1):
            #filtered the sentence to get gpt2 generated sentence
            inputs = tokenizer.encode(str(S + res), return_tensors="pt")
            outputs = model.generate(inputs, do_sample=True, max_length=80)
            t = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("beeAI > " + t)
        else:
            inputs = tokenizer.encode(str(S + filtered_sentence), return_tensors="pt")
            outputs = model.generate(inputs, do_sample=True, max_length=50)
            t = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("BeeAI > " + t)