"""
Streamlit Web UI for Telecom Incident Ticket Analyzer
Provides an interactive interface for ticket analysis.
"""

import streamlit as st
import json
import os
import pandas as pd

from config import SYNTHETIC_DIR, PDF_DIR

st.set_page_config(
    page_title="Telecom Incident Analyzer",
    page_icon="📡",
    layout="wide"
)


def load_tickets():
    """Load synthetic tickets."""
    path = os.path.join(SYNTHETIC_DIR, "tickets.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def main():
    st.title("📡 AI-Based Telecom Incident Ticket Analyzer")
    st.markdown("**Multi-Agent System with RAG-Enhanced Resolution**")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    use_rag = st.sidebar.checkbox("Enable RAG Pipeline", value=True)
    auto_approve = st.sidebar.checkbox("Auto-approve Low Risk", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Workflow Pipeline")
    st.sidebar.markdown("""
    1. 🏷️ Ticket Classification
    2. 🚨 Severity Detection  
    3. 🔀 Router Decision
    4. 💡 Resolution Suggestion (RAG)
    5. ✅ Human Approval
    """)

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎫 Analyze Ticket", "📊 Batch Processing", "📈 Dashboard", "📚 Knowledge Base"
    ])

    with tab1:
        st.header("Analyze Single Ticket")

        col1, col2 = st.columns(2)

        with col1:
            ticket_desc = st.text_area(
                "Ticket Description",
                placeholder="Enter the ticket description here...\nExample: Complete network outage affecting North region. Approximately 5000 customers impacted.",
                height=150
            )
            region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
            equipment = st.selectbox("Equipment", [
                "Nokia BTS 4400", "Ericsson RBS 6000", "Huawei BBU5900",
                "Cisco ASR 9000", "Juniper MX960", "ZTE ZXSDR B8200"
            ])
            affected = st.number_input("Affected Customers (estimate)", 1, 100000, 100)

        with col2:
            st.markdown("### Or Load Sample Ticket")
            tickets = load_tickets()
            if tickets:
                sample_idx = st.selectbox(
                    "Select sample ticket",
                    range(min(20, len(tickets))),
                    format_func=lambda x: f"{tickets[x]['ticket_id']} - {tickets[x]['category']}"
                )
                if st.button("Load Sample"):
                    ticket_desc = tickets[sample_idx]["description"]
                    st.rerun()

        if st.button("🔍 Analyze Ticket", type="primary"):
            if not ticket_desc:
                st.warning("Please enter a ticket description.")
            else:
                ticket = {
                    "ticket_id": "TKT-MANUAL-001",
                    "description": ticket_desc,
                    "region": region,
                    "equipment": equipment,
                    "affected_customers": affected,
                    "created_at": "2024-01-15 10:30:00"
                }

                with st.spinner("Processing through agent pipeline..."):
                    try:
                        from workflow import TicketWorkflow
                        workflow = TicketWorkflow(use_rag=use_rag)
                        state = workflow.process_ticket(ticket)

                        # Display results
                        st.success("Analysis Complete!")

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Category", state.classification.category)
                            st.metric("Confidence", f"{state.classification.confidence:.0%}")
                        with col_b:
                            st.metric("Priority", state.severity.priority)
                            st.metric("Impact Score", f"{state.severity.impact_score:.2f}")
                        with col_c:
                            st.metric("Route", state.routed_to.upper())
                            st.metric("Reasoning", state.severity.reasoning[:30])

                        st.markdown("---")

                        st.subheader("💡 Resolution Plan")
                        st.write(f"**Summary:** {state.resolution.summary}")
                        st.write(f"**Root Cause:** {state.resolution.root_cause_hypothesis}")

                        st.markdown("**Resolution Steps:**")
                        for i, step in enumerate(state.resolution.resolution_steps, 1):
                            st.write(f"{i}. {step}")

                        st.write(f"**Estimated Time:** {state.resolution.estimated_resolution_time}")
                        st.write(f"**Resources Needed:** {', '.join(state.resolution.resources_needed)}")

                        st.markdown("---")

                        st.subheader("✅ Approval Status")
                        if state.approval.approved:
                            st.success(f"APPROVED - Risk: {state.approval.risk_level}")
                        else:
                            st.error(f"REJECTED - Risk: {state.approval.risk_level}")
                        st.write(state.approval.approval_notes)

                        # Processing log
                        with st.expander("📋 Processing Log"):
                            for entry in state.processing_log:
                                st.text(entry)

                    except Exception as e:
                        st.error(f"Error processing ticket: {str(e)}")

    with tab2:
        st.header("Batch Processing")
        tickets = load_tickets()
        if tickets:
            num_tickets = st.slider("Number of tickets to process", 1, 20, 5)
            if st.button("🚀 Process Batch"):
                with st.spinner(f"Processing {num_tickets} tickets..."):
                    try:
                        from workflow import TicketWorkflow
                        workflow = TicketWorkflow(use_rag=use_rag)
                        results = workflow.process_batch(tickets, max_tickets=num_tickets)

                        # Results table
                        rows = []
                        for r in results:
                            if not r.error:
                                rows.append({
                                    "Ticket ID": r.ticket["ticket_id"],
                                    "Category": r.classification.category,
                                    "Priority": r.severity.priority,
                                    "Impact": f"{r.severity.impact_score:.2f}",
                                    "Route": r.routed_to,
                                    "Approved": "✅" if r.approval.approved else "❌",
                                    "ETA": r.resolution.estimated_resolution_time,
                                })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    except Exception as e:
                        st.error(f"Batch processing error: {str(e)}")
        else:
            st.info("Run `python main.py setup` first to generate synthetic data.")

    with tab3:
        st.header("Ticket Analytics Dashboard")
        tickets = load_tickets()
        if tickets:
            df = pd.DataFrame(tickets)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Tickets by Category")
                cat_counts = df["category"].value_counts()
                st.bar_chart(cat_counts)

            with col2:
                st.subheader("Tickets by Severity")
                sev_counts = df["severity"].value_counts()
                st.bar_chart(sev_counts)

            st.subheader("Tickets by Region")
            region_counts = df["region"].value_counts()
            st.bar_chart(region_counts)

            st.subheader("Recent Tickets")
            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.info("No ticket data available. Run setup first.")

    with tab4:
        st.header("Knowledge Base (RAG)")
        st.markdown("Telecom troubleshooting documents loaded into vector store:")

        if os.path.exists(PDF_DIR):
            pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
            if pdfs:
                for pdf in pdfs:
                    st.write(f"📄 {pdf}")

                st.markdown("---")
                query = st.text_input("🔍 Query Knowledge Base",
                                      placeholder="How to troubleshoot fiber cuts?")
                if query:
                    with st.spinner("Searching..."):
                        try:
                            from rag_pipeline import query_knowledge_base
                            result = query_knowledge_base(query)
                            st.markdown("**Answer:**")
                            st.write(result["answer"])
                            st.markdown("**Sources:**")
                            for src in set(result["sources"]):
                                st.write(f"- {os.path.basename(src)}")
                        except Exception as e:
                            st.error(f"RAG query error: {str(e)}")
            else:
                st.info("No PDFs found. Run `python main.py setup` first.")


if __name__ == "__main__":
    main()
