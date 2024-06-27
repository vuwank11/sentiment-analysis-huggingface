from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)

# def preprocess(text:str):
#         new_text = []
#         for t in text.split(" "):
#             t = '@user' if t.startswith('@') and len(t) > 1 else t
#             t = 'http' if t.startswith('http') else t
#             new_text.append(t)
#         return " ".join(new_text)

class SentimentAnalyzer:
    def __init__(self, model: str= f"cardiffnlp/twitter-roberta-base-sentiment-latest") -> None:
        self.analyzer = AutoModelForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.config = AutoConfig.from_pretrained(model)

    def preprocess(self, text:str):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def analyze(self, input_text):
        preprocessed_text = self.preprocess(input_text)
        encoded_input = self.tokenizer(preprocessed_text, return_tensors='pt')
        output = self.analyzer(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        results = {}
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            print(type(l))
            s = float(scores[ranking[i]]) # The type np.float32 causes issue when returning the response.
            print(type(s))
            results[l] = s
            # print(f"{i+1}) {l} {np.round(float(s), 4)}")
        return results


if __name__== "__main__":
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    analyzer = SentimentAnalyzer(MODEL)
    text = "Covid cases are increasing fast!"
    
    results = analyzer.analyze(text)
    print(results)
   


# from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoConfig
# import numpy as np
# from scipy.special import softmax
# # Preprocess text (username and link placeholders)
# def preprocess(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)
# MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)
# # PT
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# #model.save_pretrained(MODEL)
# text = "Covid cases are increasing fast!"
# text = preprocess(text)
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print("Output", output)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)
# print("Scores",scores)
# # # TF
# # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# # model.save_pretrained(MODEL)
# # text = "Covid cases are increasing fast!"
# # encoded_input = tokenizer(text, return_tensors='tf')
# # output = model(encoded_input)
# # scores = output[0][0].numpy()
# # scores = softmax(scores)
# # Print labels and scores
# ranking = np.argsort(scores)
# print(ranking)
# ranking = ranking[::-1]
# print(ranking)
# print(scores.shape)
# for i in range(scores.shape[0]):
#     l = config.id2label[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")
