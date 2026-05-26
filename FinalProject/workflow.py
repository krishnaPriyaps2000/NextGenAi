"""
Stateful Graph Orchestrator for Telecom Incident Ticket Analyzer
Built on LangGraph with MemorySaver checkpointer.
"""

from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from config import (
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL
)
from agents import (
    TicketClassificationAgent,
    SeverityDetectionAgent,
    ResolutionSuggestionAgent,
    TavilySearchAgent,
    RestructuringAgent
)

# ============================================================
# 1. State Definition
# ============================================================

class AgentState(TypedDict):
    # Raw Inputs (mapped from parsed CSV)
    ticket_id: str
    ticket_created_at: str
    ticket_customer_id: str
    ticket_region: str
    ticket_subject: str
    ticket_description: str
    ticket_affected_service: str
    ticket_channel: str
    ticket_status: str
    
    # Classification State
    ticket_category: Optional[str]
    
    # Outage Log Data (optional telemetry)
    outage_data: Optional[Dict[str, Any]]
    
    # Severity State
    priority: Optional[str]        # P1, P2, P3, P4
    impact_score: Optional[float]  # 0.0 - 1.0
    severity_reasoning: Optional[str]
    
    # Resolution State
    resolution_summary: Optional[str]
    resolution_root_cause: Optional[str]
    resolution_steps: Optional[List[str]]
    confidence_score: Optional[float]
    
    # Tavily Web Search State
    tavily_query: Optional[str]
    tavily_results: Optional[str]
    
    # Human Approval State
    approved: Optional[bool]
    approval_risk: Optional[str]
    approval_concerns: Optional[List[str]]
    approval_modifications: Optional[List[str]]
    approval_notes: Optional[str]
    
    # Final Output
    final_resolution: Optional[List[str]]
    
    # Orchestration Tracking
    loop_count: int
    processing_log: List[str]


# ============================================================
# 2. ChromaDB RAG Helper
# ============================================================

def retrieve_customer_kb_context(query: str, category: str) -> str:
    """
    Performs similarity search in ChromaDB customer_support_guide collection
    using raw chromadb and sentence-transformers matching the exact ingestion technique.
    """
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        # 1. Initialize the exact same Embedding Model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Initialize the raw Persistent Client matching configuration path
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        
        # 3. Get the correct collection name
        collection = client.get_or_create_collection(name="customer_support_guide")
        
        # 4. Map the category correctly
        allowed_metadata = ["Network_connectivity", "Service_outage", "Hardware_equipment", "Customer_experience"]
        pdf_title_val = category if category in allowed_metadata else "Service_outage"
        
        # 5. Embed the query using the exact same sentence-transformers model
        query_vector = model.encode(query).tolist()
        
        # 6. Execute the raw query with metadata filter (k=1 for token reduction)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=3,
            where={"category": pdf_title_val}
        )
        
        # 7. Format and return the retrieved text
        if results and results.get("documents") and len(results["documents"]) > 0 and len(results["documents"][0]) > 0:
            doc = results["documents"][0][0]
            return f"[Source: customer_support_guide > Category: {pdf_title_val}]\n{doc}"
            
        return "No relevant troubleshooting context found in the knowledge base."
        
    except BaseException as e:
        return f"RAG context retrieval failed: {str(e)}"


# ============================================================
# 3. Graph Nodes
# ============================================================

def classify_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 1: Extract category from ticket description using Key 1."""
    logs = list(state.get("processing_log", []))
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Classifying ticket {state['ticket_id']}...")
    
    agent = TicketClassificationAgent(api_key=GEMINI_API_KEY_1)
    result = agent.classify(state["ticket_description"], config=config)
    
    logs.append(f"  Classified category: {result.category}")
    return {
        "ticket_category": result.category,
        "processing_log": logs
    }


def severity_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 2: Severity assessment using Key 1."""
    logs = list(state.get("processing_log", []))
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Assessing severity...")
    
    # 1. Pre-execute outage log check in Python (saves one LLM roundtrip and saves tokens)
    from agents import check_outage_logs
    outage_resp = check_outage_logs.invoke({
        "created_at": state["ticket_created_at"],
        "region": state["ticket_region"]
    })
    
    outage_data = None
    if "No active outages" not in outage_resp and "failed" not in outage_resp:
        try:
            import json
            outage_data = json.loads(outage_resp)
            logs.append(f"  Outage Telemetry Detected: {outage_data}")
        except Exception:
            pass
            
    # 2. Directly invoke severity agent with outage details (single structured LLM call)
    agent = SeverityDetectionAgent(api_key=GEMINI_API_KEY_1)
    result = agent.assess_severity(
        description=state["ticket_description"],
        created_at=state["ticket_created_at"],
        region=state["ticket_region"],
        outage_info=outage_resp,
        config=config
    )
    
    logs.append(f"  Priority: {result.priority} (Impact Score: {result.impact_score:.2f})")
    logs.append(f"  Reasoning: {result.reasoning}")
    
    return {
        "priority": result.priority,
        "impact_score": result.impact_score,
        "severity_reasoning": result.reasoning,
        "outage_data": outage_data,
        "processing_log": logs
    }


def resolution_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 3: Retrieve context and suggest resolution using Key 2."""
    logs = list(state.get("processing_log", []))
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching RAG context and generating suggestion...")
    
    category = state.get("ticket_category", "Network Outage")
    priority = state.get("priority", "P4")
    
    # Query ChromaDB with category metadata filter
    context = retrieve_customer_kb_context(
        query=state["ticket_description"],
        category=category
    )
    
    feedback = state.get("approval_modifications")
    
    agent = ResolutionSuggestionAgent(api_key=GEMINI_API_KEY_2)
    result = agent.suggest_resolution(
        description=state["ticket_description"],
        category=category,
        priority=priority,
        context=context,
        feedback=feedback,
        config=config
    )
    
    logs.append(f"  Resolution Summary: {result.summary}")
    logs.append(f"  Resolution Confidence: {result.confidence_score:.2f}")
    
    return {
        "resolution_summary": result.summary,
        "resolution_root_cause": result.root_cause_hypothesis,
        "resolution_steps": result.resolution_steps,
        "confidence_score": result.confidence_score,
        "final_resolution": result.resolution_steps, # default if no restructure
        "processing_log": logs
    }


def tavily_search_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 4: Search web via Tavily and re-evaluate suggestion & confidence using Key 1."""
    logs = list(state.get("processing_log", []))
    loop_cnt = state.get("loop_count", 0)
    
    if loop_cnt >= 2:
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loop protection triggered. Bypassing search.")
        return {
            "loop_count": loop_cnt + 1,
            "processing_log": logs
        }
        
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] [Loop {loop_cnt+1}] Low confidence detected. Querying Tavily...")
    
    agent = TavilySearchAgent(api_key=GEMINI_API_KEY_1)
    
    # Formulate optimized query using LLM
    query = agent.generate_search_query(
        description=state["ticket_description"],
        category=state.get("ticket_category", ""),
        affected_service=state.get("ticket_affected_service", ""),
        config=config
    )
    logs.append(f"  Optimized Tavily Search Query: '{query}'")
    
    result = agent.search_and_structure(
        query=query,
        description=state["ticket_description"],
        config=config
    )
    
    logs.append(f"  Restructured steps retrieved via Tavily.")
    logs.append(f"  Re-evaluated Confidence Score: {result.confidence_score:.2f}")
    
    return {
        "resolution_steps": result.resolution_steps,
        "final_resolution": result.resolution_steps,
        "confidence_score": result.confidence_score,
        "loop_count": loop_cnt + 1,
        "processing_log": logs
    }


def human_approval_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 5: Human Approval Agent (Human-in-the-Loop Intervention)."""
    logs = list(state.get("processing_log", []))
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Awaiting Human Review and Intervention...")
    
    # 1. Print the current state details to the terminal for review
    print("\n" + "=" * 60)
    print("HUMAN INTERVENTION REQUIRED - RESOLUTION PLAN REVIEW:")
    print("=" * 60)
    print(f"Ticket ID:         {state.get('ticket_id')}")
    print(f"Description:       {state.get('ticket_description')}")
    print(f"Incident Category: {state.get('ticket_category')}")
    print(f"Priority Level:    {state.get('priority')} (Impact Score: {state.get('impact_score', 0.0):.2f})")
    print("\nProposed Resolution Summary:")
    print(f"  {state.get('resolution_summary')}")
    print("\nHypothesized Root Cause:")
    print(f"  {state.get('resolution_root_cause')}")
    print("\nProposed Steps:")
    for i, s in enumerate(state.get('resolution_steps') or [], 1):
        print(f"  {i}. {s}")
    print("=" * 60)
    
    # 2. Collect input from standard input (interactive terminal)
    approved_str = input("\nDo you approve this resolution? (yes/no): ").strip().lower()
    approved = approved_str in ["y", "yes", "true"]
    
    modifications = []
    concerns = []
    notes = ""
    
    if approved:
        mod_input = input("Enter any modifications/notes to apply (press Enter to approve without changes):\n> ").strip()
        if mod_input:
            modifications.append(mod_input)
            notes = f"Approved with modifications: {mod_input}"
        else:
            notes = "Approved by human operator."
    else:
        rej_input = input("Enter concerns/rejection reasons:\n> ").strip()
        concerns.append(rej_input)
        modifications.append(rej_input)  # Pass back to resolution node
        notes = f"Rejected by human operator. Concerns: {rej_input}"
        
    logs.append(f"  Human Decision: {'APPROVED' if approved else 'REJECTED'}")
    if modifications:
        logs.append(f"  Human Feedback: {modifications}")
        
    return {
        "approved": approved,
        "approval_risk": "Low" if approved and not modifications else "Medium" if approved else "High",
        "approval_concerns": concerns,
        "approval_modifications": modifications,
        "approval_notes": notes,
        "processing_log": logs
    }


def restructure_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Node 6: Rewrite final resolution suggestion based on modifications using Key 2."""
    logs = list(state.get("processing_log", []))
    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Restructuring resolution based on human modifications...")
    
    agent = RestructuringAgent(api_key=GEMINI_API_KEY_2)
    final_steps = agent.restructure(
        original_steps=state.get("resolution_steps", []),
        modifications=state.get("approval_modifications", []),
        config=config
    )
    
    logs.append(f"  Restructuring complete. Final plan updated.")
    return {
        "final_resolution": final_steps,
        "processing_log": logs
    }


def confidence_router_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Pass-through node to represent the confidence router visually in the diagram."""
    return {}


def approval_router_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Pass-through node to represent the approval router visually in the diagram."""
    return {}


# ============================================================
# 4. Conditional Edges
# ============================================================

def route_based_on_confidence(state: AgentState) -> str:
    """Route after Resolution & Tavily Search node depending on confidence score."""
    conf = state.get("confidence_score", 0.0)
    loop_cnt = state.get("loop_count", 0)
    
    # If confidence is high OR we hit safety loop threshold, proceed
    if conf >= 0.65 or loop_cnt >= 2:
        priority = state.get("priority", "P4")
        if priority in ["P1", "P2"]:
            return "human_approval"
        else:
            return "end"
    else:
        return "tavily_search"


def route_after_approval(state: AgentState) -> str:
    """Route after human approval agent."""
    approved = state.get("approved", False)
    modifications = state.get("approval_modifications", [])
    
    if approved:
        if modifications:
            return "restructure"
        else:
            return "end"
    else:
        # Loop back to resolution suggestion with feed-back modifications
        return "resolution"


# ============================================================
# 5. Compiled LangGraph Orchestration
# ============================================================

def build_workflow() -> StateGraph:
    """Builds and compiles the multi-agent LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("severity", severity_node)
    workflow.add_node("resolution", resolution_node)
    workflow.add_node("tavily_search", tavily_search_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("restructure", restructure_node)
    
    # Explicit Router Nodes for visual representation in diagrams
    workflow.add_node("confidence_router", confidence_router_node)
    workflow.add_node("approval_router", approval_router_node)
    
    # Establish Entry
    workflow.set_entry_point("classify")
    
    # Normal Edges
    workflow.add_edge("classify", "severity")
    workflow.add_edge("severity", "resolution")
    workflow.add_edge("restructure", END)
    
    # Edges leading into explicit routers
    workflow.add_edge("resolution", "confidence_router")
    workflow.add_edge("human_approval", "approval_router")
    
    # Conditional Edges from Confidence Router
    workflow.add_conditional_edges(
        "confidence_router",
        route_based_on_confidence,
        {
            "tavily_search": "tavily_search",
            "human_approval": "human_approval",
            "end": END
        }
    )
    
    # Edge from Tavily Search back into Confidence Router (to recheck updated confidence)
    workflow.add_edge("tavily_search", "confidence_router")
    
    # Conditional Edges from Human Approval Router
    workflow.add_conditional_edges(
        "approval_router",
        route_after_approval,
        {
            "restructure": "restructure",
            "resolution": "resolution",
            "end": END
        }
    )
    
    return workflow
