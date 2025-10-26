!pip install langchain_community

!pip install langchain-experimental

!pip install pdfplumber

!pip install faiss-cpu

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from google.colab import files

# Upload the PDF file
uploaded = files.upload()

# The uploaded file will be saved in the current working directory
# You can check the filename using:
pdf_filename = list(uploaded.keys())[0]
print(f"Uploaded file: {pdf_filename}")

loader = PDFPlumberLoader(pdf_filename)
docs = loader.load()

# Check the number of pages
print("Number of pages in the PDF:",len(docs))

# Load the random page content
docs[1].page_content

text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

print("Number of chunks created: ", len(documents))

print(documents[0].page_content)

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

llm = ChatOpenAI(openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = "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", model = "anthropic/claude-haiku-4.5")

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store
vector = FAISS.from_documents(documents, embedder)
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n

from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

prompt = """
1. You are psychologist
2. Use the following pieces of context to answer the question at the end.
3. Answer only by using the context and articulate it better, use bullet point and emoji if required
4. Keep the answer crisp and limited to 3,4 sentences.

Context: {context}

Question: {question}

Answer to the question:"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
                  llm=llm,
                  prompt=QA_CHAIN_PROMPT,
                  callbacks=None,
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None,
              )

qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True,
              )

print(qa("Challenges of Digital Detox?")["result"])
