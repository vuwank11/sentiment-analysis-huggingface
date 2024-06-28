from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_analysis.sentiments import SentimentAnalyzer
import uvicorn

class TextRequest(BaseModel):
    text: str

app = FastAPI()

analyzer = SentimentAnalyzer(f"cardiffnlp/twitter-roberta-base-sentiment-latest")

@app.post("/sentiment")
def analyze_text(request: TextRequest):
    results = analyzer.analyze(request.text)
    print(results)
    return results

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline
# import uvicorn

# # You can check any other model in the Hugging Face Hub
# pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# # We define the app
# app = FastAPI()

# # We define that we expect our input to be a string
# class RequestModel(BaseModel):
#    input: str

# # Now we define that we accept post requests
# @app.post("/sentiment")
# def get_response(request: RequestModel):
#    prompt = request.input
#    response = pipe(prompt)
#    label = response[0]["label"]
#    score = response[0]["score"]
#    return f"The '{prompt}' input is {label} with a score of {score}"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


    


