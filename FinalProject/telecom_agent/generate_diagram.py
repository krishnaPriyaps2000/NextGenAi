import os
from workflow import build_workflow

# 1. Compile the graph
graph = build_workflow().compile()

# 2. Get the graph and generate Mermaid PNG bytes
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    
    # 3. Save the diagram to the workspace folder
    output_path = "workflow_diagram2.png"
    with open(output_path, "wb") as f:
        f.write(png_bytes)
        
    print(f"✅ LangGraph workflow diagram successfully generated and saved to: {os.path.abspath(output_path)}")
except Exception as e:
    print(f"❌ Failed to generate graph diagram: {e}")
