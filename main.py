import os
from dotenv import load_dotenv
import openai


load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")
openai.Engine.list()  # check we have authenticated

MODEL = "text-embedding-ada-002"

# Create an embedding for a single document using the text-embedding-ada-002 model
res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], 
    engine=MODEL
)

embeds = [record["embedding"] for record in res["data"]]