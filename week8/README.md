# Week 8 – Single Agent Systems & Agent Pipelines

## Overview

This project was developed as part of **Week 8 of the Celebal Data Science Internship 2026**.

The assessment focuses on understanding **Single Agent Systems, Agent Workflows, Tools, Stateful Graphs, Conditional Routing, and Agent Evaluation** through a short quiz and a small Agentic AI pipeline.

The project implements a simple agent pipeline using **LangGraph, LangChain, and Groq**. The system analyzes a user's query, determines the appropriate action, executes the selected tool or LLM, and returns a final response.

---

## Objectives

- Understand the fundamentals of Agentic AI systems.
- Implement a stateful directed graph.
- Use nodes and edges to define an agent workflow.
- Implement conditional routing based on query type.
- Integrate tools into an agent pipeline.
- Handle errors and retry failed operations.
- Integrate an LLM for general-purpose queries.

---

## Agent Pipeline

```text
                    User Query
                        |
                        v
                 Query Router
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
     Calculator     Keyword        General
        Tool        Extraction       LLM
          |             |             |
          +-------------+-------------+
                        |
                        v
                 Final Response
```

---

## Features

- Stateful workflow using LangGraph
- Conditional query routing
- Calculator tool
- Keyword extraction tool
- General LLM responses
- Error handling
- Retry mechanism for failed calculations
- Interactive command-line interface

---

## How It Works

### 1. Query Router

The router analyzes the user's query and decides which capability should handle it.

| Query Type | Route |
|---|---|
| Calculation query | Calculator Tool |
| Keyword extraction query | Keyword Extraction Tool |
| Other questions | General LLM |

### 2. Calculator Tool

The calculator handles basic mathematical expressions.

**Example:**

```
calculate 25 * 4
```

**Output:**

```
The calculated result is: 100
```

The calculator also includes a retry mechanism to handle failed calculations.

### 3. Keyword Extraction Tool

The keyword tool extracts important words from a given query while removing common stop words.

**Example:**

```
extract keywords from machine learning artificial intelligence deep learning
```

### 4. General LLM

Queries that do not match the calculator or keyword routes are handled by a Large Language Model using Llama 3.1 through Groq.

**Example:**

```
What is artificial intelligence?
```

### 5. Final Response

The result produced by the selected tool or LLM is passed to the final response stage and displayed to the user.

---

## Technologies Used

- Python
- LangGraph
- LangChain
- Groq
- Llama 3.1
- Python-dotenv

---

## Project Structure

```
week8/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

The `.env` file is required for running the project locally but is not included in the repository for security reasons.

---

## Environment Setup

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Never upload the `.env` file or expose your API key publicly.

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

The application will start an interactive command-line interface.

---

## Example Queries

**Calculator**

```
calculate 25 * 4
```

**Keyword Extraction**

```
extract keywords from machine learning artificial intelligence deep learning
```

**General Question**

```
What is artificial intelligence?
```

**Error Handling Test**

```
calculate 25 / 0
```

The calculator attempts the operation again and returns an appropriate error message if it continues to fail.

---

## Key Concepts Demonstrated

This project demonstrates the following Agentic AI concepts:

- Stateful Directed Graphs
- Nodes and Edges
- Conditional Routing
- Tool Usage
- LLM Integration
- Sequential Tool Execution
- Error Handling
- Retry Mechanisms
- Agent Decision Making

---

## Learning Outcomes

Through this project, I gained practical understanding of how agent workflows can be designed using stateful graphs and how an agent can dynamically select different tools based on the user's query.

The project also provided hands-on experience with LangGraph, LangChain, tool integration, conditional routing, error handling, and LLM-based responses.

---

## Conclusion

This project demonstrates a simple but complete Agentic AI pipeline where a user query is analyzed, routed to the appropriate capability, processed using a tool or LLM, and returned as a final response.

It provides a practical foundation for understanding how more advanced agentic systems can be built using stateful workflows and tool-based decision making.