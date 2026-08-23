# RAG with LangChain + AWS Bedrock

A retrieval-augmented generation app built on AWS Bedrock, using LangChain to orchestrate document retrieval and generation over a PDF knowledge base (demonstrated with the "Attention Is All You Need" paper).

## How it works
- Chunks the source PDF and builds a FAISS vector index
- - Uses AWS Bedrock-hosted embedding and LLM models via LangChain integrations
  - - Retrieves relevant chunks and generates a grounded answer to user queries
   
    - ## Tech stack
    - Python, LangChain, AWS Bedrock, FAISS
   
    - ## Run locally
    - pip install -r requirements.txt
    - streamlit run app.py
   
    - Requires AWS credentials configured with Bedrock access.
    - 
