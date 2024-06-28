# sentiment-analysis-huggingface

An end-to-end sentiment analysis project using huggingface, fast-api and docker.

Command to build docker image with docker file:
`docker build -t sentiment-app:1.1.0 .`

Command to run docker image:
`docker run --name sent_container -p 8000:8000 sentiment-app:1.0.0`
