import os
from dotenv import load_dotenv
import openai
import pinecone


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
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

# Create a Pinecone index
if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))
# Connect to the index
index = pinecone.Index('openai')