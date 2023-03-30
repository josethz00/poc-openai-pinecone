import os
from dotenv import load_dotenv
import openai


load_dotenv()

openai.api_key = os.getenv("OPEN_AI_API_KEY")
openai.Engine.list()  # check we have authenticated