# Interaction between Azure Production RAG and FrugalGPT

This document describes the interactions between the `azure-production-rag`, `FrugalGPT` and `PromptWIzard` projects, as well as the changes made to enable their integration. It aims to provide a clear and detailed overview.

---

## Context

- **Azure Production RAG**: This project implements a Retrieval-Augmented Generation (RAG) system in production on Azure. It integrates advanced language models and document retrieval pipelines.

- **FrugalGPT**: This project optimizes the cost of using language models by combining computational efficiency strategies and lightweight models. It relies on techniques such as prompt adaptation, LLM model approximation, and model cascades.
`
- **PromptWizard**: A system designed to enhance and optimize prompts based on user input context, ensuring more accurate and relevant responses. It plays a pre-processing role in the pipeline, improving prompt quality before inference.


The primary objective of this integration is to leverage FrugalGPT's cost optimizations and PromptWizard’s prompt refinement to enhance Azure Production RAG’s efficiency and response quality.


---

## Project Architecture

![Architecture](./image/archi.png)

---

## Changes Made

### In `azure-production-rag`

1. **Integration with FrugalGPT**:
   - Modified the `chatreadretrieveread.py` file (in the `approaches` folder of the backend) to call FrugalGPT. This file transmits the necessary information to generate a response and obtain a cost estimate for the query.
   - Created the `server_client.py` file, called by `chatreadretrieveread.py`, to handle communication with FrugalGPT via port 8000 on localhost. This file performs:
     - A call to the model cascade.
     - A cost comparison between different models.
     - An HTTP 200 response in case of successful operation.

### In `FrugalGPT`

1. **Support for RAG Pipelines**:
   - Added specific features for RAG pipelines, including:
     - Processing contextual queries to ensure response consistency.
     - Optimized management of retrieved documents for lightweight models.

2. **Integration API**:
   - Developed a REST API for seamless interaction with Azure Production RAG. This API includes:
     - Endpoints for cost-optimized inference.
     - Real-time cost metrics.
     - Robust error handling to ensure reliability.

3. **Computational Efficiency Strategies**:
   - Implemented fallback mechanisms: lightweight models are prioritized, and heavy models are only called when necessary.
   - Dynamic resource management to adjust models based on cost and latency constraints.
   - Introduced a caching solution in the `llmcache.py` file to reuse responses to similar questions, avoiding unnecessary API calls.
   - Used the `sentence-transformers` library with the `all-mpnet-base-v2` model available on Hugging Face: [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). This model can be downloaded locally or called directly.

### In `PromptWizard`

1. **Integration API**:
   - Built a REST API for generating optimized prompts before inference.
   - The refined prompts are sent to FrugalGPT, enhancing response accuracy and reducing the likelihood of invoking high-cost models unnecessarily.


---

## Interaction Points

1. **Inference Pipeline**:
   - Azure Production RAG delegates key inference operations to FrugalGPT via API.
   - PromptWizard acts as a pre-processing step, optimizing prompts before they enter the cascade pipeline.
   - FrugalGPT returns optimized results, which are integrated back into the RAG workflow.

2. **Cost Management**:
   - FrugalGPT provides detailed metrics, such as:
     - Cost per query.
     - Savings achieved compared to exclusive use of heavy models.
   - These metrics allow Azure Production RAG to dynamically adjust its execution strategies.

3. **Data Sharing**:
   - Both projects share data, including:
     - Retrieved documents to ensure response consistency.
     - Usage logs to refine models and optimization strategies.

---

## Developer Instructions

### Configuration in `azure-production-rag`

1. Configure your Azure environment with your subscription and required services.
2. Start the backend and frontend with the command:
   ```bash
   ./start.ps1
   ```
3. Ensure that the `server_client.py` file (in the `core` folder of the backend) contains the necessary parameters to establish the connection with FrugalGPT.

### Configuration in `FrugalGPT`

1. Download the required database with the command:
   ```bash
   wget -P db/ https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.sqlite
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file containing the necessary endpoints and API keys.

### Running `FrugalGPT`

1. Test the LLM models and configure the cascade strategy with:
   ```bash
   python approachfrugalgpt.py
   ```
2. Start the server to enable the REST API:
   ```bash
   uvicorn main:app --reload
   ```
3. You can run both commands sequentially with:
   ```bash
   ./run_scripts.ps1
   ```

### Running `PromptWizard`
1. Navigate to the project root directory.
2. Start the server with:
   ```bash
   uvicorn server:app --host 127.0.0.1 -- port 7000 --reload
   ```

### Running `FrugalGPT` and `azure-production-rag` Together

1. Launch the respective commands:
   - For FrugalGPT: `uvicorn main:app --reload`
   - For Azure Production RAG: `./start.ps1`
   - For PromptWizard: `uvicorn server:app --host 127.0.0.1 -- port 7000 --reload`
2. Once the projects are running, the interaction between the repositories allows answering questions posed via the Legal Copilot frontend interface.

---

## Conclusion

This integration brings together the robustness of Azure’s production RAG system, the cost-efficiency of FrugalGPT, and the input optimization of PromptWizard. Together, they deliver a highly scalable, responsive, and budget-conscious AI solution. For more details, refer to the `CONTRIBUTING.md` and `CHANGELOG.md` files in each repository, and don’t hesitate to submit issues or suggestions via GitHub.