from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import aiml
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


k = aiml.Kernel()
k.learn("Learning.aiml")
k.respond("LEARN AIML")
analyser = SentimentIntensityAnalyzer()

while True:
    us = input("User > ")
    reply = k.respond(us)

    print(us)
    sentiment_statement = analyser.polarity_scores(us)
    sentiment_results = (sentiment_statement['compound'])
    sent_res_neg = (sentiment_statement['neg'])
    sent_res_pos = (sentiment_statement['neu'])
    sent_res_neu = (sentiment_statement['pos'])
    print(sentiment_results)
    print(sent_res_neg)
    print(sent_res_pos)
    print(sent_res_neu)

    pos = 0
    neg = 0
    neu = 0

    sntmnt = []

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
        print("bot > ", reply)
    else:
        #print("bot > Why don't you try this article on Diversity and Inclusion: https://www.strasity.com/?gclid=Cj0KCQjw8p2MBhCiARIsADDUFVG-hEOccIiwlFqX7X3gg0npbNSbEvxzjRJWAqPAOk8ZC4uE-TwzSnkaAm-GEALw_wcB ", )
        inputs = tokenizer.encode(us, return_tensors="pt")
        outputs = model.generate(inputs, do_sample=True, max_length=80)
        t = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("bot > " + t)