from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()
embeddings_model = VertexAIEmbeddings(model_name="text-embedding-005")
def get_embeddings(texts: list[str]):
    return embeddings_model.embed_documents(texts)
