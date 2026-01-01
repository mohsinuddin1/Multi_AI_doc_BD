from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload .env

import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.tools.duckduckgo importv DuckDuckGoTools
from agno.models.groq import Groq
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
import tempfile
import os


COLLECTION_NAME = "intelligent_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"




CONTEXT_RESEARCH_AGENT_INSTRUCTIONS = """
You are the **Context Research Agent** - an expert at understanding document context and finding relevant external information.

## YOUR ROLE:
You provide comprehensive context for any document by researching similar documents, industry standards, and relevant policies.

## WHEN ANALYZING A DOCUMENT, YOU MUST:

### 1. DOCUMENT TYPE IDENTIFICATION
- Identify what type of document this is (contract, policy, report, agreement, memo, proposal, etc.)
- Determine the industry/domain (legal, finance, healthcare, technology, HR, etc.)
- Note the document's purpose and intended audience

### 2. CONTEXTUAL RESEARCH
- Search for similar document templates and industry standards
- Find relevant regulations, compliance requirements, or best practices
- Identify benchmark clauses or terms commonly used in similar documents
- Research any organizations, entities, or parties mentioned

### 3. COMPARATIVE ANALYSIS
- Compare key terms against industry norms
- Identify any unusual or non-standard elements
- Note missing sections that are typically present in similar documents
- Highlight deviations from common practices

### 4. BACKGROUND INFORMATION
- Provide context on legal/regulatory frameworks that apply
- Explain industry-specific terminology found in the document
- Research any referenced laws, standards, or external documents

## OUTPUT FORMAT:
Provide a structured analysis with clear sections:
- **Document Classification**: Type, domain, purpose
- **Key Context**: Important background information discovered
- **Industry Standards**: Relevant benchmarks and norms
- **External References**: Laws, regulations, or standards that apply
- **Comparative Insights**: How this document compares to similar ones
"""

CLAUSE_INTELLIGENCE_AGENT_INSTRUCTIONS = """
You are the **Clause Intelligence Agent** - an expert at extracting and analyzing every important element from documents.

## YOUR ROLE:
You perform deep extraction of all obligations, commitments, deadlines, financial terms, and structured data from any document.

## WHEN ANALYZING A DOCUMENT, YOU MUST:

### 1. COMPLETE STRUCTURE ANALYSIS
- Map the entire document structure (sections, subsections, appendices)
- Identify all numbered clauses and their hierarchy
- Note any cross-references between sections
- Highlight definitions sections and key defined terms

### 2. OBLIGATION EXTRACTION
- Extract ALL obligations for each party mentioned
- Categorize obligations by type:
  * Performance obligations (what must be done)
  * Financial obligations (payments, fees, costs)
  * Reporting obligations (notices, reports, disclosures)
  * Compliance obligations (regulations, standards to follow)
- Identify who is responsible for each obligation

### 3. TIMELINE & DEADLINE EXTRACTION
- Extract ALL dates mentioned in the document
- Identify deadlines, milestones, and timeframes
- Note renewal dates, termination periods, notice periods
- Create a chronological timeline of key events/dates
- Flag any conflicting or unclear timelines

### 4. FINANCIAL TERMS EXTRACTION
- Extract all monetary values, fees, and payment terms
- Identify payment schedules and conditions
- Note penalties, late fees, or financial consequences
- Summarize total financial exposure/commitment

### 5. KEY TERMS & CONDITIONS
- Extract termination clauses and conditions
- Identify confidentiality and non-disclosure terms
- Note intellectual property provisions
- Extract limitation of liability terms
- Identify indemnification obligations
- Note dispute resolution mechanisms

### 6. PARTIES & RELATIONSHIPS
- Identify all parties mentioned
- Map relationships between parties
- Note any third-party rights or obligations
- Identify guarantors, agents, or representatives

## OUTPUT FORMAT:
Provide a comprehensive extraction with:
- **Document Structure Map**: Visual hierarchy of the document
- **Parties Involved**: List with their roles
- **Obligations Matrix**: Table of who owes what to whom
- **Timeline/Deadlines**: Chronological list of all dates
- **Financial Summary**: All monetary terms and totals
- **Key Clauses**: Important terms extracted verbatim with analysis
- **Definitions**: Key defined terms and their meanings
"""

RISK_RECOMMENDATION_AGENT_INSTRUCTIONS = """
You are the **Risk & Recommendation Agent** - an expert at identifying risks and providing actionable recommendations.

## YOUR ROLE:
You analyze documents to flag potential issues, risks, and problems, then provide specific recommendations for improvement.

## WHEN ANALYZING A DOCUMENT, YOU MUST:

### 1. RISK IDENTIFICATION
Scan the entire document for:

**Legal Risks:**
- Ambiguous language that could lead to disputes
- Missing essential clauses (e.g., force majeure, governing law)
- Unenforceable or problematic provisions
- One-sided or unfair terms
- Compliance gaps with applicable laws

**Financial Risks:**
- Unlimited liability exposure
- Unclear payment terms
- Hidden costs or escalation clauses
- Inadequate financial protections
- Penalty or fee exposure

**Operational Risks:**
- Unrealistic timelines or deliverables
- Vague performance standards
- Insufficient exit or termination rights
- Dependency risks
- Resource or capacity concerns

**Strategic Risks:**
- Conflicts with existing agreements
- Competitive restrictions
- Intellectual property concerns
- Reputation risks
- Long-term commitment concerns

### 2. RISK SEVERITY ASSESSMENT
For each risk identified:
- Assign severity level: 🔴 CRITICAL | 🟠 HIGH | 🟡 MEDIUM | 🟢 LOW
- Explain the potential impact
- Identify who is most affected
- Note the likelihood of the risk materializing

### 3. GAP ANALYSIS
- Identify missing clauses that should be present
- Note incomplete or vague provisions
- Flag undefined terms or ambiguous language
- Highlight inconsistencies within the document

### 4. SPECIFIC RECOMMENDATIONS
For each issue found, provide:
- **The Problem**: Clear description of the issue
- **The Risk**: What could go wrong
- **The Recommendation**: Specific action to take
- **Suggested Language**: If applicable, propose revised wording

### 5. NEGOTIATION POINTS
- Identify terms that should be negotiated
- Prioritize by importance and likelihood of success
- Suggest alternative approaches
- Note deal-breaker vs. nice-to-have items

## OUTPUT FORMAT:
Provide a structured risk report:
- **Executive Summary**: Top 3-5 critical findings
- **Risk Dashboard**: Visual summary with severity levels
- **Detailed Risk Analysis**: Each risk with full analysis
- **Gap Analysis**: Missing or incomplete elements
- **Recommendations Matrix**: Prioritized action items
- **Negotiation Strategy**: Key points to address
- **Red Flags**: Immediate concerns requiring attention
"""

ORCHESTRATOR_AGENT_INSTRUCTIONS = """
You are the **Orchestrator Agent** - the team lead who coordinates all agents and delivers a comprehensive document review.

## YOUR ROLE:
You coordinate the Context Research Agent, Clause Intelligence Agent, and Risk & Recommendation Agent to produce a complete, professional document analysis.

## YOUR RESPONSIBILITIES:

### 1. COORDINATE ANALYSIS
- Ensure all agents analyze the document thoroughly
- Synthesize findings from all team members
- Resolve any conflicting interpretations
- Fill in any gaps in the analysis

### 2. PRODUCE COMPREHENSIVE OUTPUT
When delivering the final analysis, structure it as:

## 📋 DOCUMENT OVERVIEW
- Document type and classification
- Key parties involved
- Document date and effective period
- Overall purpose and scope

## 🔍 CONTEXT & BACKGROUND
(From Context Research Agent)
- Industry context and standards
- Relevant regulations and compliance requirements
- How this compares to similar documents

## 📊 DOCUMENT ANALYSIS
(From Clause Intelligence Agent)
- Complete structure breakdown
- All obligations by party
- Timeline of key dates and deadlines
- Financial terms summary
- Key clauses and their implications

## ⚠️ RISKS & ISSUES
(From Risk & Recommendation Agent)
- Risk dashboard with severity levels
- Critical issues requiring immediate attention
- Gaps and missing elements
- Potential problem areas

## ✅ RECOMMENDATIONS
(From Risk & Recommendation Agent)
- Prioritized action items
- Suggested improvements
- Negotiation points
- Next steps

## 📝 EXECUTIVE SUMMARY
- 3-5 key takeaways
- Overall assessment (favorable/neutral/concerning)
- Immediate actions required
- Final recommendation

### 3. QUALITY CONTROL
- Ensure analysis is complete and actionable
- Verify all sections of the document were reviewed
- Confirm recommendations are specific and practical
- Make the output clear and easy to understand

### 4. ADAPT TO DOCUMENT TYPE
Adjust the analysis focus based on document type:
- **Contracts**: Focus on obligations, risks, negotiation points
- **Policies**: Focus on compliance, implementation, gaps
- **Reports**: Focus on findings, data accuracy, conclusions
- **Proposals**: Focus on feasibility, costs, risks
- **Agreements**: Focus on terms, commitments, protections

## OUTPUT REQUIREMENTS:
- Use clear markdown formatting
- Include severity indicators (🔴🟠🟡🟢) for risks
- Provide actionable, specific recommendations
- Make the analysis accessible to non-experts
- Highlight the most important findings prominently
"""



def init_session_state():
    """Initialize all session state variables"""
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY")

    if "qdrant_api_key" not in st.session_state:
        st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if "qdrant_url" not in st.session_state:
        st.session_state.qdrant_url = os.getenv("QDRANT_URL")

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if "review_team" not in st.session_state:
        st.session_state.review_team = None

    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []




def init_qdrant():
    """Initialize Qdrant vector database connection"""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None

    try:
        return Qdrant(
            collection=COLLECTION_NAME,
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            embedder=SentenceTransformerEmbedder(
                id=EMBEDDING_MODEL
            )
        )
    except Exception as e:
        st.error(f"Qdrant connection failed: {e}")
        return None


def get_groq_model():
    """Get configured Groq model instance"""
    return Groq(
        id=LLM_MODEL,
        api_key=st.session_state.groq_api_key
    )


def process_document(uploaded_file, vector_db: Qdrant):
    """Process and index uploaded document"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Create Knowledge base and add PDF content using PDFReader
        knowledge_base = Knowledge(vector_db=vector_db)

        with st.spinner("📤 Indexing document..."):
            knowledge_base.add_content(path=tmp_path, reader=PDFReader())

        os.unlink(tmp_path)
        return knowledge_base

    except Exception as e:
        raise RuntimeError(f"Document processing failed: {e}")


def create_review_team(knowledge_base):
    """Create the intelligent document review team"""
    
    # Context Research Agent
    context_agent = Agent(
        name="Context Research Agent",
        role="Document Context & Research Specialist",
        model=get_groq_model(),
        tools=[DuckDuckGoTools()],
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=CONTEXT_RESEARCH_AGENT_INSTRUCTIONS,
        markdown=True
    )

    # Clause Intelligence Agent
    clause_agent = Agent(
        name="Clause Intelligence Agent",
        role="Document Extraction & Analysis Specialist",
        model=get_groq_model(),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=CLAUSE_INTELLIGENCE_AGENT_INSTRUCTIONS,
        markdown=True
    )

    # Risk & Recommendation Agent
    risk_agent = Agent(
        name="Risk & Recommendation Agent",
        role="Risk Analysis & Advisory Specialist",
        model=get_groq_model(),
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=RISK_RECOMMENDATION_AGENT_INSTRUCTIONS,
        markdown=True
    )

    # Orchestrator Agent (Team Lead)
    review_team = Team(
        name="Orchestrator Agent",
        model=get_groq_model(),
        members=[context_agent, clause_agent, risk_agent],
        knowledge=knowledge_base,
        search_knowledge=True,
        instructions=ORCHESTRATOR_AGENT_INSTRUCTIONS,
        markdown=True,
        share_member_interactions=True
    )

    return review_team


# ================== UI ==================

def render_sidebar():
    """Render the sidebar with configuration and upload"""
    with st.sidebar:
        st.header("🔑 API Configuration")

        # API Keys
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key or ""
        )
        if groq_key:
            st.session_state.groq_api_key = groq_key

        qdrant_key = st.text_input(
            "Qdrant API Key",
            type="password",
            value=st.session_state.qdrant_api_key or ""
        )
        if qdrant_key:
            st.session_state.qdrant_api_key = qdrant_key

        qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url or ""
        )
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url

        # Initialize Qdrant connection
        if all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
            if not st.session_state.vector_db:
                st.session_state.vector_db = init_qdrant()
                if st.session_state.vector_db:
                    st.success("✅ Qdrant Connected")

        st.divider()

        # Document Upload
        if all([st.session_state.groq_api_key, st.session_state.vector_db]):
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader(
                "Upload Document (PDF)",
                type=["pdf"],
                help="Upload any document for intelligent review"
            )

            if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
                with st.spinner("Processing document..."):
                    try:
                        kb = process_document(uploaded_file, st.session_state.vector_db)
                        st.session_state.knowledge_base = kb
                        st.session_state.processed_files.add(uploaded_file.name)
                        
                        # Create review team
                        st.session_state.review_team = create_review_team(kb)
                        st.success(f"✅ '{uploaded_file.name}' processed & agents ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # Show processed files
            if st.session_state.processed_files:
                st.divider()
                st.subheader("📚 Processed Documents")
                for file in st.session_state.processed_files:
                    st.caption(f"• {file}")

        st.divider()
        
        # Agent Info
        st.header("🤖 Review Team")
        st.markdown("""
        **1. Context Research Agent**
        - Finds industry standards & context
        - Researches similar documents
        
        **2. Clause Intelligence Agent**
        - Extracts all obligations & terms
        - Maps timelines & financials
        
        **3. Risk & Recommendation Agent**
        - Identifies risks & issues
        - Provides actionable advice
        
        **4. Orchestrator Agent**
        - Coordinates all agents
        - Delivers comprehensive review
        """)


def render_main_content():
    """Render the main content area"""
    st.title("🔍 AI Intelligent Document Reviewer")
    st.markdown("*Comprehensive document analysis powered by multi-agent AI*")

    if not st.session_state.review_team:
        st.info("👈 Configure API keys and upload a document to begin")
        
        # Show capabilities
        st.markdown("---")
        st.subheader("📋 What This Tool Does")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🔍 Document Analysis
            - **Full Structure Mapping** - Understand document hierarchy
            - **Obligation Extraction** - Who owes what to whom
            - **Timeline Identification** - All dates and deadlines
            - **Financial Summary** - Costs, fees, and payments
            """)
            
        with col2:
            st.markdown("""
            ### ⚠️ Risk Assessment
            - **Risk Identification** - Legal, financial, operational
            - **Severity Rating** - Critical to low priority
            - **Gap Analysis** - Missing or incomplete elements
            - **Recommendations** - Specific actions to take
            """)
        
        st.markdown("---")
        st.subheader("📄 Supported Documents")
        st.markdown("""
        - Contracts & Agreements
        - Policies & Procedures
        - Reports & Proposals
        - Legal Documents
        - Business Documents
        - Any PDF document
        """)
        return

    # Analysis Interface
    st.markdown("---")
    
    # Quick Analysis Buttons
    st.subheader("⚡ Quick Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Full Review", use_container_width=True):
            st.session_state.query = "Provide a comprehensive review of this entire document. Analyze every section, extract all key information, identify all risks, and provide complete recommendations."
    
    with col2:
        if st.button("⚠️ Risk Analysis", use_container_width=True):
            st.session_state.query = "Focus on identifying all risks in this document. Categorize by severity and provide specific recommendations for each risk."
    
    with col3:
        if st.button("📊 Extract Terms", use_container_width=True):
            st.session_state.query = "Extract all key terms, obligations, dates, deadlines, and financial information from this document in a structured format."
    
    with col4:
        if st.button("📝 Executive Summary", use_container_width=True):
            st.session_state.query = "Provide a concise executive summary with the top 5 most important findings and immediate action items."

    st.markdown("---")
    
    # Custom Query
    st.subheader("💬 Ask Anything About The Document")
    query = st.text_area(
        "Enter your question or analysis request",
        value=st.session_state.get("query", ""),
        placeholder="Examples:\n- What are my main obligations under this agreement?\n- Are there any concerning clauses I should negotiate?\n- What happens if I want to terminate early?\n- Summarize all payment terms and deadlines",
        height=100
    )

    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if query:
            with st.spinner("🔍 Analyzing document with AI agents..."):
                try:
                    response: RunOutput = st.session_state.review_team.run(query)
                    
                    st.markdown("---")
                    st.markdown("## 📄 Analysis Results")
                    
                    # Display main response
                    if response.content:
                        st.markdown(response.content)
                    else:
                        st.warning("No response generated")
                    
                    # Store in history
                    st.session_state.analysis_history.append({
                        "query": query,
                        "response": response.content
                    })
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please enter a question or select a quick analysis option")

    # Show analysis history
    if st.session_state.analysis_history:
        st.markdown("---")
        with st.expander("📜 Previous Analyses", expanded=False):
            for i, item in enumerate(reversed(st.session_state.analysis_history)):
                st.markdown(f"**Query {len(st.session_state.analysis_history) - i}:** {item['query'][:100]}...")
                st.markdown(item['response'][:500] + "..." if len(item['response']) > 500 else item['response'])
                st.divider()


# ================== MAIN ==================

def main():
    st.set_page_config(
        page_title="AI Intelligent Document Reviewer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    init_session_state()
    
    # Render UI
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
