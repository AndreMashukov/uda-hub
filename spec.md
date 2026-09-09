Project Scenario
You’ve joined a fast-growing AI startup building the next frontier in customer support automation.

Your team is responsible for building UDA-Hub, a Universal Decision Agent designed to plug into existing customer support systems (Zendesk, Intercom, Freshdesk, internal CRMs) and intelligently resolve tickets. But this isn’t just another FAQ bot.

The goal? Build an agentic system that reads, reasons, routes, and resolves, acting as the operational brain behind support teams.

You’ll need to design an agent system that can:

Understand customer tickets across channels
Decide which agent or tool should handle each case
Retrieve or infer answers when possible
Escalate or summarize issues when necessary
Learn from interactions by updating long-term memory
Your agent should not only automate, it should decide how to automate!

Project Introduction
In this project, you will develop UDA-Hub, an intelligent, multi-agent decision suite capable of resolving customer support tickets across multiple platforms.

Key Capabilities:

Multi-Agent Architecture with LangGraph Design and orchestrate specialized agents (e.g., Supervisor, Classifier, Resolver, Escalation…).

Input Handling Accept incoming support tickets in natural language with metadata (e.g., platform, urgency, history).

Decision Routing and Resolution

Route tickets to the right agent based on classification
Retrieve relevant knowledge via RAG if needed
Resolve or escalate based on confidence and context
Memory Integration

Maintain state during steps of the execution
Short-term memory is used as context to keep conversation running during the same session
Store and recall long-term memory for preferences, as an example
Project Summary
Inputs:

Incoming support ticket (text + metadata)
Internal knowledge base (FAQ, previous tickets)
Optional internal tool (e.g., refund)
Memory store (for prior conversations and resolutions)
Deliverables:

A LangGraph-powered multi-agent system that:

Understands tickets
Routes to correct agent with tools
Resolves or escalates based on decision logic
Uses memory appropriately


Project Instructions
Your starter folder looks like the following structure:

starter/
├── agentic/
│   ├── agents/
│   ├── design/
│   ├── tools/
│   └── workflow.py
├── data/
│   ├── core/
│   ├── external/
│   └── models/
├── .env
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb
└── utils.py
Design
Start by designing the solution. Your implementation will follow it.
Place all the documentation and diagrams about the design of your agentic system inside agentic/design
Setup
Run notebook 01_external_db_setup.ipynb in order to have all the data related to the account Cultpass. It's the first customer that has purchased Uda-hub
Run notebook 02_core_db_setup.ipynb in order to have all the data related to Uda-hub application, including the files "received" from Cultpass like cultpass_articles.jsonl
You need to expand cultpass_articles form 4 to at least 14 articles. Make sure you have diverse topics for your agentic system.
Agentic Workflow
Develop your agents inside agentic/agents and your tools inside agentic/tools . This will help you with modularity.
Develop your workflow orchestration in the file workflow.py . There's already a sample for you, but donot use it, create the graph from scratch. Do not use the prebuilt workflow.
When developing tools that abstract the database both for retrieval or for actions, please mind the relative/absolute paths. I strongly recommend you to use something like MCP servers for the tools.
If you're using RAG for retrieval, make sure you have documented how it works.
For short-term memory (session), you can use thread_id. For long-term memory, you're free to use semantic search.
Run
There's a chat_interface() function inside utils.py. It's just a simple while True block. Starter code imports this inside the notebook. Feel free to improve it!
You're not forced to use 03_agentic_app.ipynb , you can develop inside a .py module, but please name it as 03_agentic_app.py and make it explicit how to run your project.
You must create test cases to pass the project!
Submission Instructions
As mentioned above, you're receiving the starter code, but please submit your project with all artifacts under solution/ . We'll not look into starter/ . Make sure you are copying and pasting the code from starter/ to solution/ if you're not modifying it.

If you have installed a package, share the name and version in the documentation. Ideally share your requirements.txt and Python version, If you're developing locally.

DON'Ts
import or reference a folder outside solution/ !
share your .env file
submit large .db files
submit only the notebooks without the other artifacts