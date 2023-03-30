import os
from dotenv import load_dotenv
import openai
import pinecone
from datasets import load_dataset
from tqdm.auto import tqdm  # this is our progress bar

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

# Load 2000 rows from the TREC dataset
trec = load_dataset('trec', split='train[:2000]')

from tqdm.auto import tqdm  # this is our progress bar

batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))

query = input("Enter a query: ")

xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")