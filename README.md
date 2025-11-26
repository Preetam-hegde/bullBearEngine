# BullBear Engine

A Human-in-the-Loop financial analysis system using MCP (Model Context Protocol) and LangGraph.

## Architecture

- **Host**: Chainlit application with LangGraph agent (`host/`).
- **Server**: MCP Server providing financial tools (`server/`).

## Setup

1.  **Prerequisites**: Python 3.10+ installed.
2.  **Environment**:

    - Create a `.env` file in `host/` based on `host/.env.example`.
    - Add your `OPENAI_API_KEY`.

3.  **Installation**:

    ```bash
    # Install Server dependencies
    pip install -r server/requirements.txt

    # Install Host dependencies
    pip install -r host/requirements.txt
    ```

## Running the Application

1.  **Start the Host** (which internally connects to the Server tools for this local setup):

    ```bash
    chainlit run host/app.py
    ```

2.  **Usage**:
    - Open the browser at `http://localhost:8000`.
    - Ask the agent to analyze a stock, e.g., "Analyze the trend for AAPL".

# Project Structure Overview

## 📁 Server (`server/`)

### **Created Files**

#### **1. `tools/market_data.py`**

- Fetches market data using **yfinance**.
- Provides clean wrappers for price history, indicators, and intraday data.

#### **2. `tools/market_curve.py`**

- Analyzes trends and curves using **SciPy**.
- Performs smoothing, trend detection, and curve-fit–based insights.

#### **3. `tools/sr_zones.py`**

- Detects **Support / Resistance zones** using **scikit-learn** clustering.
- Generates SR bands for decision-making.

#### **4. `main.py`**

- Entry point for the MCP Server.
- Uses **fastmcp** to expose analysis tools as callable MCP functions.

---

## 📁 Host (`host/`)

### **Created Files**

#### **1. `app.py`**

- Chainlit UI application.
- Renders messages, charts, and agent responses interactively.

#### **2. `agent.py`**

- LangGraph agent orchestrating the full reasoning loop.
- Handles:
  - LLM reasoning
  - Tool planning
  - Validation & error recovery
  - Conversation state

#### **3. `mcp_interface.py`**

- Bridge layer to call the server’s MCP tools.
- Currently uses direct imports for simplicity.
- Architecture is flexible—can be swapped with a real MCP client later.

---

## ⚙️ Configuration

### **Created**

#### **`requirements.txt`**

- Separate lists for **host** and **server**.
- Includes:
  - FastMCP
  - LangGraph
  - Chainlit
  - yfinance
  - scipy
  - scikit-learn
  - python-dotenv
