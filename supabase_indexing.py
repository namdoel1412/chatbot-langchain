import os
import getpass
# from supabase import create_client, Client
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from transformers import GPT2TokenizerFast, GPTNeoXTokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter

#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
os.environ['SUPABASE_URL'] = "https://psjuizkurzqtarbaigor.supabase.co"
os.environ['SUPABASE_SERVICE_KEY'] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBzanVpemt1cnpxdGFyYmFpZ29yIiwicm9sZSI6ImFub24iLCJpYXQiOjE2ODU0MTc4NTMsImV4cCI6MjAwMDk5Mzg1M30.vLH5LcnVFb35vt1Tn_WVNlVyHr13xPZDTlXOMA7y-m8"

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
import textract
# doc = textract.process("/content/drive/MyDrive/Colab Notebooks/data/Cloud Computing for Dummies.pdf")
doc = textract.process("./docs/TentenDoc.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('data.txt', 'w', encoding="utf-8") as f:
    f.write(doc.decode('utf-8'))

# lines = []
# with open('data.txt', 'r', encoding="utf-8") as f:
#     lines = f.readlines()
# new_lines = []
# for line in lines:
#     if line.strip() != '':
#         new_lines.append(line)
# with open(f'data.txt', 'w', encoding="utf-8") as f:
#     f.writelines(new_lines)
#     f.close()
with open('data.txt', 'r', encoding="utf-8") as f:
    text = f.read()

# Step 3: Create function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 512,
    chunk_overlap = 32,
    separators=['\n\n', '\n', ' ', ''],
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

# Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
type(chunks[0])

# Quick data visualization to ensure chunking was successful

# Create a list of token counts
token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

"""# 2. Embed text and store embeddings"""

# Get embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = SupabaseVectorStore.from_documents(
    chunks, embeddings, client=supabase
)
query = "What did the president say about Ketanji Brown Jackson"
matched_docs = vector_store.similarity_search(query)
print(matched_docs[0].page_content)