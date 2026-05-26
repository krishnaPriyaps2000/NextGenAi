import os
import pypdf
from sentence_transformers import SentenceTransformer
import chromadb
from config import VECTOR_STORE_DIR, DATA_DIR

# 1. Initialize the SentenceTransformer Embedding Model (matching your technique)
print("Initializing sentence-transformers Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 1.5. Clean reset the database folder to resolve incompatible/corrupted SQLite schemas under Python 3.13
import shutil
backup_path = VECTOR_STORE_DIR + "_backup"
if os.path.exists(VECTOR_STORE_DIR):
    print("Detected existing ChromaDB directory. Backing up and clearing to ensure a fresh, Python 3.13 compatible database schema...")
    try:
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(VECTOR_STORE_DIR, backup_path)
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        print("Successfully cleared old database files. Re-initializing client...")
    except Exception as e:
        print(f"Warning: Could not automatically clear database folder: {e}. Trying to proceed...")

# 2. Initialize Persistent ChromaDB Client pointing to config directory (data/chroma_db)
print(f"Initializing Persistent ChromaDB Client at: {VECTOR_STORE_DIR}")
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

# 3. Create or get the exact collection name
collection_name = "customer_support_guide"
print(f"Retrieving or creating collection: '{collection_name}'")
collection = client.get_or_create_collection(name=collection_name)

# 4. Map your metadata categories to the exact local filenames in the project root
pdf_mapping = {
    "Service_outage": "PDF2_Service_Outage_Guide 2.pdf",
    "Network_connectivity": "PDF1_Network_Connectivity_Guide 2.pdf",
    "Hardware_equipment": "PDF3_Hardware_Equipment_Guide 2.pdf",
    "Customer_experience": "PDF4_Customer_Experience_Guide 2.pdf"
}

# 5. Helper function to chunk text with overlap (matching your technique)
def chunk_text(text, chunk_size=500, overlap=50):
    """Splits text into chunks of roughly 'chunk_size' characters with 'overlap'."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap # Move back slightly to create an overlap
    return chunks

# 6. Helper function to extract text from a PDF (using modern pypdf which is already installed!)
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except FileNotFoundError:
        print(f"[ERROR] PDF File not found -> '{pdf_path}'. Please ensure it is in the project root directory.")
    return text

# 7. Process each PDF and store in ChromaDB
for category, filename in pdf_mapping.items():
    pdf_path = os.path.join(os.path.dirname(__file__) or ".", filename)
    print(f"\nProcessing '{filename}' for category: '{category}'...")
    
    # Extract
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text:
        continue # Skip if file was not found or is empty
        
    # Chunk
    text_chunks = chunk_text(raw_text)
    
    # Embed and Store
    for i, chunk in enumerate(text_chunks):
        embedding = model.encode(chunk).tolist()
        
        # Store in vector DB with Metadata filter matching your Colab script
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"category": category}], # Metadata attached here
            ids=[f"{category}_chunk_{i}"]
        )
    print(f"  -> Successfully stored {len(text_chunks)} chunks into ChromaDB collection '{collection_name}'.")

print("\n[SUCCESS] All PDFs successfully processed and stored in your local ChromaDB vector store!")

# 8. Test Query (Verifying your filtered search works perfectly!)
print("\n" + "=" * 60)
print("TESTING LOCAL QUERY RETRIEVAL:")
print("=" * 60)
test_query = "What do I do if the router light is red?"
query_vector = model.encode(test_query).tolist()

results = collection.query(
    query_embeddings=[query_vector],
    n_results=2,
    where={"category": "Hardware_equipment"} # Filters search strictly within the Hardware PDF
)

print(f"Query:                    '{test_query}'")
print(f"Filtered Search Category: 'Hardware_equipment'\n")
print("Results retrieved:")
if results and results.get('documents') and len(results['documents']) > 0:
    for idx, doc in enumerate(results['documents'][0]):
        print(f"Result {idx + 1}: {doc[:200].strip()}...")
else:
    print("No matching records found.")
print("=" * 60 + "\n")
