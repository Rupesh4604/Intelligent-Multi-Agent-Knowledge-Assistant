# 1. Install all necessary libraries
# ------------------------------------
# !pip install langchain langchain-google-genai wikipedia pydantic
# !pip install langchain-community faiss-cpu
# !pip install flashrank

import os
import asyncio
import textwrap
from typing import Literal, List

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import wikipedia

# --- RAG Specific Imports ---
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank

# --- SETUP: Create Dummy Knowledge Base Files ---

# HR Policy Document
hr_policy_content = """
# Company Leave Policy

## Annual Leave
Full-time staff are entitled to 24 days of annual leave per year. This accrues at a rate of 2 days per month. Up to 5 unused days may be carried forward into the next year with manager approval. Requests for leave must be submitted through the HR portal at least two weeks in advance.

## Sick Leave
Sick leave is for personal illness or injury. It is determined by service duration. A doctor's note is required for absences longer than 3 consecutive days. After 1 year of service, employees get 4 weeks full pay.

## Compassionate Leave
Employees can take up to 5 days of paid compassionate leave per year for the loss of a close family member.

## Work From Home (WFH) Policy
Employees may work from home up to 2 days per week. The company provides a stipend for home office setup. All company IT and security policies must be adhered to while working remotely.
"""
with open("hr_policy.txt", "w") as f:
    f.write(hr_policy_content)

# Technical Knowledge Base Document
tech_docs_content = """
# Internal Technical Documentation

## Project Phoenix: Frontend
- Repository: git.corp.example.com/phoenix/frontend-app
- Language: TypeScript, Framework: React
- Description: This is the main customer-facing web application.

## Project Phoenix: Backend
- Repository: git.corp.example.com/phoenix/backend-services
- Language: Python, Framework: FastAPI
- Description: These are the microservices that power the Phoenix frontend.

## Deployment Procedures
- Deployments to production must go through the CI/CD pipeline in Jenkins.
- Request a production deployment by creating a JIRA ticket with the 'DEVOPS' component.
"""
with open("tech_docs.txt", "w") as f:
    f.write(tech_docs_content)
# ------------------------------------


# 2. Configure API Key
# --------------------
llm = None
try:
    from google.colab import userdata
    os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    llm.invoke("Test query")
    print("✅ Gemini API Key configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    exit()


# 3. Setup RAG Pipelines
# ----------------------
hr_retriever = None
tech_retriever = None

# Helper function to create a RAG retriever
def create_rag_retriever(file_path: str) -> FAISS.as_retriever:
    loader = TextLoader(file_path)
    docs = loader.load()
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_documents = parent_splitter.split_documents(docs)
    child_documents = []
    for parent_doc in parent_documents:
        child_docs = child_splitter.split_text(parent_doc.page_content)
        for child_doc_text in child_docs:
            child_documents.append(
                Document(page_content=child_doc_text, metadata={"parent_content": parent_doc.page_content})
            )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(child_documents, embeddings)
    # Retrieve more documents initially (k=5) to feed into the re-ranker
    return vectorstore.as_retriever(search_kwargs={"k": 5})

try:
    print("\n⚙️  Setting up HR RAG pipeline...")
    hr_retriever = create_rag_retriever("./hr_policy.txt")
    print("✅ HR RAG pipeline setup complete.")

    print("\n⚙️  Setting up Technical RAG pipeline...")
    tech_retriever = create_rag_retriever("./tech_docs.txt")
    print("✅ Technical RAG pipeline setup complete.")

except Exception as e:
    print(f"❌ Error setting up RAG pipelines: {e}")
    exit()


# 4. Instantiate Re-ranker and Query Expansion Chain
# --------------------------------------------------
print("\n✨ Initializing Re-ranker and Query Expansion chain...")
reranker = FlashrankRerank()
query_expansion_prompt = ChatPromptTemplate.from_template("Rewrite the user query into 3 alternative versions to improve vector search recall.\n\nOriginal Query:\n{query}\n\nExpanded Queries:")
query_expansion_chain = query_expansion_prompt | llm | StrOutputParser()
print("✅ Re-ranker and Query Expansion chain initialized.")


# 5. Define All Tools with Full RAG Capabilities
# ----------------------------------------------
@tool
def hr_rag_tool(query: str) -> str:
    """Searches the HR knowledge base for company policies. Use for HR-related questions."""
    print(f"\n[HR Tool] ➡️ Query: '{query}'")
    # 1. Pre-retrieval: Query Expansion
    expanded_queries_str = query_expansion_chain.invoke({"query": query})
    all_queries = [query] + expanded_queries_str.strip().split('\n')
    print(f"[HR Tool] 🔍 Expanded queries: {all_queries}")
    # 2. Retrieval
    all_retrieved_docs = []
    for q in all_queries:
        all_retrieved_docs.extend(hr_retriever.invoke(q))
    # 3. Post-retrieval: Re-ranking
    print(f"[HR Tool] ✨ Re-ranking {len(all_retrieved_docs)} documents...")
    reranked_docs = reranker.compress_documents(documents=all_retrieved_docs, query=query)
    unique_parent_contents = {doc.metadata['parent_content'] for doc in reranked_docs}
    return "Retrieved and re-ranked context:\n" + "\n\n".join(unique_parent_contents)

@tool
def tech_rag_tool(query: str) -> str:
    """Searches internal tech docs for company projects, repos, and standards."""
    print(f"\n[Tech Tool] ➡️ Query: '{query}'")
    # 1. Pre-retrieval: Query Expansion
    expanded_queries_str = query_expansion_chain.invoke({"query": query})
    all_queries = [query] + expanded_queries_str.strip().split('\n')
    print(f"[Tech Tool] 🔍 Expanded queries: {all_queries}")
    # 2. Retrieval
    all_retrieved_docs = []
    for q in all_queries:
        all_retrieved_docs.extend(tech_retriever.invoke(q))
    # 3. Post-retrieval: Re-ranking
    print(f"[Tech Tool] ✨ Re-ranking {len(all_retrieved_docs)} documents...")
    reranked_docs = reranker.compress_documents(documents=all_retrieved_docs, query=query)
    unique_parent_contents = {doc.metadata['parent_content'] for doc in reranked_docs}
    return f"Retrieved and re-ranked context:\n" + "\n\n".join(unique_parent_contents)

@tool
def wiki_tool(query: str) -> str:
    """Fetches a summary from Wikipedia for general knowledge technical questions."""
    print(f"\n[Wiki Tool] ➡️ Query: '{query}'...")
    try:
        return wikipedia.summary(query, auto_suggest=False, sentences=5)
    except Exception as e:
        return f"An error occurred fetching from Wikipedia: {e}"


# 6. Define All Agents
# --------------------
# Guardrail Agent
class GuardrailOutput(BaseModel):
    is_valid: bool; reasoning: str
guardrail_prompt = ChatPromptTemplate.from_messages([("system", "Is the user's query about HR or Technical topics? Respond only with the structured output."), ("human", "Query: {query}")])
guardrail_agent_runnable = guardrail_prompt | llm.with_structured_output(GuardrailOutput)

# HR Agent
hr_agent_prompt = ChatPromptTemplate.from_messages([("system", "You are an HR assistant. Use the `hr_rag_tool` to answer questions based ONLY on the retrieved context."), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
hr_tools = [hr_rag_tool]
hr_agent = create_tool_calling_agent(llm, hr_tools, hr_agent_prompt)
hr_agent_executor = AgentExecutor(agent=hr_agent, tools=hr_tools, verbose=False)

# Technical Agent
tech_agent_prompt = ChatPromptTemplate.from_messages([("system", "You are a technical assistant with two tools: `tech_rag_tool` for internal company tech info, and `wiki_tool` for public general tech knowledge. You must choose the appropriate tool."), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")])
tech_tools = [tech_rag_tool, wiki_tool]
tech_agent = create_tool_calling_agent(llm, tech_tools, tech_agent_prompt)
tech_agent_executor = AgentExecutor(agent=tech_agent, tools=tech_tools, verbose=False)

# Triage Agent
class TriageDecision(BaseModel):
    agent: Literal["HR", "Technical"]
triage_prompt = ChatPromptTemplate.from_messages([("system", "Is this query for 'HR' or 'Technical'? Respond only with the structured output."), ("human", "Query: {query}")])
triage_agent_runnable = triage_prompt | llm.with_structured_output(TriageDecision)


# 7. Orchestrate the System
# -------------------------
async def run_agent_system(query: str):
    print("="*80)
    print(f"👤 User Query: {query}")
    print("="*80)
    print("🛡️  Running Guardrail Check...")
    guardrail_result = await guardrail_agent_runnable.ainvoke({"query": query})
    if not guardrail_result.is_valid:
        print(f"❌ Guardrail Blocked Input. Reason: {guardrail_result.reasoning}\n")
        return
    print(f"✅ Guardrail Passed.")
    print("\n🚦 Running Triage Agent...")
    triage_decision = await triage_agent_runnable.ainvoke({"query": query})
    selected_agent = triage_decision.agent
    print(f"🎯 Specialist selected: {selected_agent}")
    print(f"\n▶️  Invoking {selected_agent} Agent...")
    if selected_agent == "HR":
        result = await hr_agent_executor.ainvoke({"input": query})
    else:
        result = await tech_agent_executor.ainvoke({"input": query})
    final_answer = result.get('output', 'Sorry, I could not process your request.')
    print("\n" + "-"*80)
    print("🤖 Final Answer:")
    print(textwrap.fill(final_answer, width=80))
    print("-"*80 + "\n")


# 8. Run Example Queries
# ----------------------
async def main():
    # Test 1: Internal tech question -> Triage to Tech -> Tech Agent uses tech_rag_tool
    await run_agent_system("Where can I find the backend repo for Project Phoenix?")
    
    # Test 2: General tech question -> Triage to Tech -> Tech Agent uses wiki_tool
    await run_agent_system("What is React?")
    
    # Test 3: HR question -> Triage to HR -> HR Agent uses hr_rag_tool
    await run_agent_system("Do I get paid for compassionate leave?")
    
    # Test 4: Out-of-scope question -> Blocked by Guardrail
    await run_agent_system("What's the weather like today?")


if __name__ == "__main__":
    if llm and hr_retriever and tech_retriever:
        asyncio.run(main())