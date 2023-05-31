import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from transformers import GPT2TokenizerFast, GPTNeoXTokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
# expired
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"

# active
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

# Set persist directory

loader = PyPDFLoader("./docs/TentenDoc.pdf")
pages = loader.load_and_split()
print(pages[0])

# SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
chunks = pages

# Advanced method - Split by chunk

# Step 1: Convert PDF to text
import textract
# doc = textract.process("/content/drive/MyDrive/Colab Notebooks/data/Cloud Computing for Dummies.pdf")
doc = textract.process("./docs/TentenDoc.pdf")

# Step 2: Save to .txt and reopen (helps prevent issues)
with open('data.txt', 'w', encoding="utf-8") as f:
    f.write(doc.decode('utf-8'))

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

# Create vector database
# db = FAISS.from_documents(chunks, embeddings)


# ========= Update ==========
persist_directory = 'db'

# Split documents and generate embeddings
# gmo_docs_split = text_splitter.split_documents(gmo_docs)

# Create Chroma instances and persist embeddings
gmoDB = Chroma.from_documents(chunks, embeddings, persist_directory=os.path.join(persist_directory, 'gmo'))
gmoDB.persist()
