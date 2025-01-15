# Flowchart of app

```mermaid
flowchart TD
    A[Start] --> B[Load Environment Variables]
    B --> C[Parse Arguments: --link, --question, --model]
    C --> D[ask_question Function]
    D --> E[WebBaseLoader: Load Webpage Content]
    E --> F[RecursiveCharacterTextSplitter: Split Documents into Chunks]
    F --> G[HuggingFaceEmbeddings: Create Embeddings]
    G --> H[Chroma: Create Vector Store from Document Chunks]
    H --> I[PromptTemplate: Define Prompt for QA]
    I --> J[HuggingFaceEndpoint: Initialize LLM Model]
    J --> K[RetrievalQA: Create QA Chain]
    K --> L[Invoke QA Chain with Question]
    L --> M[Return Answer]
    M --> N[Print Answer]
    N --> O[End]
