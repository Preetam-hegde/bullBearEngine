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

### **Tools (`server/tools/`)**

- **`market_data.py`**: Fetches historical market data using **yfinance**.
- **`company_info.py`**: Retrieves company profile, sector, industry, and key financial metrics.
- **`technical_analysis.py`**: Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.) and generates AI-driven market analysis.
- **`prediction.py`**: Uses **Random Forest** (scikit-learn) to predict future price movements based on technical indicators and lag features.
- **`performance_analysis.py`**: Analyzes historical performance, calculating metrics like Sharpe Ratio, Volatility, and Max Drawdown.
- **`visualize.py`**: Generates interactive **Plotly** charts for market data and indicators.
- **`create_chart.py`**: Helper module for creating advanced candlestick and indicator charts.
- **`perfomance_metric.py`**: Helper module for calculating and visualizing performance metrics.

### **Main Entry Point**

- **`main.py`**: The MCP Server entry point using **fastmcp**. Exposes the tools as callable functions for the Host.

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
  - plotly
  - xgboost
  - lightgbm
  - scipy
  - scikit-learn
  - python-dotenv
