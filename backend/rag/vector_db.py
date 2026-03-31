from pymilvus import DataType, MilvusClient
from rag.embedding import get_embeddings
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

client = MilvusClient(uri="http://localhost:19530")
vector_db_name = "finsolve_internal_docs"

schema = client.create_schema(auto_id=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768) 
schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="role_access", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=100)
schema.add_field(field_name="source_doc", datatype=DataType.VARCHAR, max_length=500)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector", 
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
index_params.add_index(
    field_name="role_access",
    index_type="INVERTED"
)

client.create_collection(
    collection_name=vector_db_name,
    schema=schema,
    index_params=index_params
)


# data ingestion part 
def process_and_save_file(file_path: str, client: MilvusClient, collection_name: str):
    path = Path(file_path)
    
    # 2. Setup Text Splitter
    # Standard chunk_size for 768-dim models is usually 500-1000 tokens
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    # 3. Read and Split Document
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = text_splitter.split_text(text)
    
    # 4. Generate Metadata
    # Extracts 'engineering' from 'data/engineering/doc.md'
    role_name = path.parent.name 
    source_name = path.name

    # 5. Generate Embeddings for all chunks
    # embeddings_model = get_embeddings_model()
    vectors = get_embeddings(chunks)

    # 6. Prepare data for Milvus
    data_to_insert = []
    for i, chunk in enumerate(chunks):
        data_to_insert.append({
            "vector": vectors[i],
            "content": chunk,
            "role_access": [role_name], # Array type requirement
            "source_doc": source_name
        })

    # 7. Insert into Collection
    if data_to_insert:
        res = client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        return res

def query_collection_source_doc(file_name: str):
    results = client.query(
        collection_name="finsolve_internal_docs",
        filter=f"source_doc == '{file_name}'",
        output_fields=["content", "role_access"]
    )
    return results

def query_collection_role_access(role_name: str):
    results = client.query(
        collection_name="finsolve_internal_docs",
        filter=f"'role_access'=='{role_name}'",
        output_fields=["*"]
    )
    return results

def vector_search(query_vector: list[float], role_name: str):
    results = client.search(
    collection_name="finsolve_internal_docs",
    data=[query_vector],         
    filter=f"ARRAY_CONTAINS(role_access, '{role_name}')", 
    anns_field="vector",    # The name of your vector column
    limit=5,
    search_params={"metric_type": "COSINE"},  # Direct application of Cosine Similarity
    output_fields=['content']
    )
    return results

# embeddings = get_embeddings('FinSolve Technologies’s stable net margin indicates sound control over both direct and indirect costs')
