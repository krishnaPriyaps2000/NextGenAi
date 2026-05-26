"""
Telecom Incident Ticket Analyzer - CLI Main Entry Point (LangGraph Version)
Ingests tickets in the new CSV format (ticket_id, created_at, customer_id, region, subject, description, affected_service, channel, status),
runs the stateful multi-agent LangGraph workflow, and outputs step-by-step diagnostic traces and final suggestions.
"""

# SSL fix - MUST be first
import ssl
import urllib3
import os
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["REQUESTS_CABUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import sys
# Resolve standard 'data' package conflict by prioritizing current directory in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
import io
from datetime import datetime
from config import SYNTHETIC_DIR
from workflow import build_workflow
from langgraph.checkpoint.memory import MemorySaver


def parse_csv_ticket_string(csv_line: str) -> dict:
    """
    Parses a single CSV formatted ticket string.
    Expected headers: ticket_id, created_at, customer_id, region, subject, description, affected_service, channel, status
    """
    f = io.StringIO(csv_line.strip())
    reader = csv.DictReader(f)
    try:
        row = next(reader)
        return row
    except Exception:
        # Fallback to hardcoded parsing if headers are missing
        f.seek(0)
        reader = csv.reader(f)
        row = next(reader)
        headers = ["ticket_id", "created_at", "customer_id", "region", "subject", "description", "affected_service", "channel", "status"]
        return dict(zip(headers, row))


def get_demo_csv_string() -> str:
    """Returns a realistic sample ticket in the new CSV format for testing by reading directly from the CSV file."""
    csv_path = os.path.join(SYNTHETIC_DIR, "telecom_tickets_india.csv")
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) >= 68:
            return lines[0].strip() + "\n" + lines[67].strip()
    except Exception:
        pass

    # Robust fallback if the file is not found or fails to read
    return (
        "ticket_id,created_at,customer_id,region,subject,description,affected_service,channel,status\n"
        'TKT0001,2024-04-01 14:15:00,CUST27464,Kochi,No signal since morning,"Customer is getting no signal since 7 AM. Phone shows \'No Service\'. Restarting phone did not help. Neighbours also facing the same issue.",Mobile Data,Walk-in,Pending Customer'
    )


def run_langgraph_workflow(csv_string: str):
    """Executes the stateful multi-agent LangGraph workflow on a single CSV ticket row."""
    # Parse CSV into a row dictionary
    ticket_row = parse_csv_ticket_string(csv_string)
    
    print("\n" + "=" * 60)
    print("INCOMING TICKET CSV PARSED (NEW SCHEMA):")
    print("=" * 60)
    print(f"Ticket ID:   {ticket_row.get('ticket_id')}")
    print(f"Customer ID: {ticket_row.get('customer_id')}")
    print(f"Region:      {ticket_row.get('region')}")
    print(f"Subject:     {ticket_row.get('subject')}")
    print(f"Created At:  {ticket_row.get('created_at')}")
    print(f"Description: {ticket_row.get('description')}")
    print(f"Service:     {ticket_row.get('affected_service')}")
    print("=" * 60)

    # Initialize Graph shared AgentState
    initial_state = {
        "ticket_id": ticket_row.get("ticket_id", "TKT-UNKNOWN"),
        "ticket_created_at": ticket_row.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "ticket_customer_id": ticket_row.get("customer_id", "CUST-UNKNOWN"),
        "ticket_region": ticket_row.get("region", "Central"),
        "ticket_subject": ticket_row.get("subject", "No Subject"),
        "ticket_description": ticket_row.get("description", ""),
        "ticket_affected_service": ticket_row.get("affected_service", "General"),
        "ticket_channel": ticket_row.get("channel", "Web"),
        "ticket_status": ticket_row.get("status", "Open"),
        
        # State variables filled in by nodes
        "ticket_category": None,
        "outage_data": None,
        "priority": None,
        "impact_score": None,
        "severity_reasoning": None,
        
        "resolution_summary": None,
        "resolution_root_cause": None,
        "resolution_steps": None,
        "confidence_score": 0.0,
        
        "tavily_query": None,
        "tavily_results": None,
        
        "approved": None,
        "approval_risk": None,
        "approval_concerns": None,
        "approval_modifications": None,
        "approval_notes": None,
        
        "final_resolution": None,
        
        "loop_count": 0,
        "processing_log": []
    }

    # Setup MemorySaver Checkpointer
    checkpointer = MemorySaver()
    workflow_graph = build_workflow().compile(checkpointer=checkpointer)
    
    # Configure run threading
    thread_id = ticket_row.get("ticket_id", "demo-thread")
    config = {"configurable": {"thread_id": thread_id}}
    
    print("\nExecuting LangGraph Multi-Agent Orchestrator...")
    
    # Check if LangSmith API key is configured
    langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")
    if langsmith_key:
         print(f"LangSmith logging active. View execution graph in your dashboard under project: '{os.getenv('LANGCHAIN_PROJECT', 'Telecom')}'")
         
    final_state = workflow_graph.invoke(initial_state, config=config)
    
    print("\n" + "=" * 60)
    print("AGENT TRACES & SYSTEM LOGS:")
    print("=" * 60)
    for log_entry in final_state.get("processing_log", []):
        print(log_entry)
        
    print("\n" + "=" * 60)
    print("FINAL RESOLUTION PLAN (OUTPUT):")
    print("=" * 60)
    print(f"Incident Category: {final_state.get('ticket_category')}")
    print(f"Priority Level:    {final_state.get('priority')} (Impact Score: {final_state.get('impact_score'):.2f})")
    print(f"Summary of Issue:  {final_state.get('resolution_summary')}")
    print(f"Hypothesized Cause: {final_state.get('resolution_root_cause')}")
    print(f"Approval Result:   {'APPROVED' if final_state.get('approved') else 'REJECTED'} (Risk: {final_state.get('approval_risk')})")
    if final_state.get("approval_notes"):
        print(f"Approval Notes:    {final_state.get('approval_notes')}")
    
    print("\nActionable Steps:")
    for i, step in enumerate(final_state.get("final_resolution", []), 1):
        print(f"  {i}. {step}")
    print("=" * 60 + "\n")


def main():
    """Main CLI entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "single":
            if len(sys.argv) > 2:
                # User passed a custom CSV row or CSV file path
                csv_arg = sys.argv[2]
                if os.path.exists(csv_arg):
                    with open(csv_arg, "r", encoding="utf-8") as f:
                        csv_data = f.read()
                else:
                    csv_data = csv_arg
            else:
                csv_data = get_demo_csv_string()
                
            run_langgraph_workflow(csv_data)
        elif mode == "ingest":
            from data.ingest_pdfs_by_heading import ingest_all_pdfs
            reset_flag = "--reset" in sys.argv
            ingest_all_pdfs(reset=reset_flag)
        else:
            print(f"\nUnknown mode: {mode}")
            print("Usage: python main.py [single [csv_data_or_file_path] | ingest [--reset]]")
    else:
        # Default: single ticket run
        csv_data = get_demo_csv_string()
        run_langgraph_workflow(csv_data)


if __name__ == "__main__":
    main()
