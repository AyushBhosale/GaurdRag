from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from rag.vector_db import vector_search
from rag.embedding import get_embeddings

load_dotenv()

# 1. Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# 2. Define Query and Retrieval Logic
query = "What was FinSolve Technologies' revenue growth in 2024, and which specific metric was identified as needing improvement for better cash flow stability?"

# 3. Define the RBAC-aware Prompt
system_instruction = (
    "You are a Secure AI Insights Agent. Use the provided context to answer the user's query. "
    "Adhere strictly to Role-Based Access Control (RBAC): only share information relevant to "
    "the user's department. If the context does not contain the answer, state that you do not have access."
    "\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("human", "{input}")
])

chain = prompt | llm


# --- Example Usage ---
def ask_secure_agent(query: str, department: str ):
    """
    Processes a user query by searching the vector DB and 
    generating an RBAC-aware response.
    """
    # 1. Convert query to embeddings
    # (Assuming get_embeddings returns a list, we take the first element)
    query_embedding = get_embeddings(query)[0]

    # 2. Search the Vector DB
    # vector_search returns a nested list [[hit1, hit2, ...]]
    search_results = vector_search(query_embedding, department)
    relevant_chunks = search_results[0] 

    # 3. Extract and join the 'content' field from results
    context_text = "\n\n".join([hit['entity']['content'] for hit in relevant_chunks])

    # 4. Invoke the LangChain chain
    response = chain.invoke({
        "context": context_text,
        "input": query
    })

    return response.content

# # --- Example Usage ---
# user_query = "What was the revenue growth in 2024?"
# answer = ask_secure_agent(user_query)
# print(answer)