# Quantum-Enhanced Financial Risk & Scenario Analysis

This project was built for the **Amrita QuantumLeap Bootcamp 2026 Hackathon**.

It provides a single, unified dashboard using Streamlit to compare classical financial risk models with quantum algorithms. Specifically, it contrasts classical Monte Carlo simulations and Value at Risk (VaR) calculations against a Quantum Circuit Born Machine (QCBM) modeled with Qiskit.

## Features

1. **Frontend & Data Loading**:
   - Sidebar for User Input Ticker (e.g., AAPL, NVDA, TSLA, BTC-USD).
   - Downloads 2 years of daily returns using `yfinance`.
   - Extracts Mean ($\mu$) and Volatility ($\sigma$).

2. **Classical Analysis & Scenario Assembler**:
   - Performs a Monte Carlo Simulation to project a 4-year timeline of potential price paths.
   - Calculates 95% Value at Risk (VaR) using the historical and parametric methods.

3. **Quantum Engine (QCBM)**:
   - Builds a 4-qubit Quantum Circuit Born Machine (QCBM) using Qiskit.
   - Gates used: Hadamard (H) for superposition, RY gates to encode the Mean ($\mu$), RZ gates to encode Volatility ($\sigma$), and CNOT gates to create market entanglement.
   - Runs on `AerSimulator` for 1024 shots.
   - Uses the Born Rule to translate measurement probabilities into Scenario Probability Weightings bounded between $[\mu-3\sigma, \mu+3\sigma]$.

4. **Interactive Visualizations**:
   - Displays the actual quantum circuit diagram.
   - Side-by-side comparison histograms of Classical returns vs. Quantum samples with VaR lines.
   - A line chart showing the 4-year projected paths from the Classical Scenario Assembler.
   - Metric Cards displaying Volatility, Max Drawdown, Classical Hist VaR, and Quantum VaR.

## Educational Context

*   **Quantum Measurement provides true randomness**: Classical random number generators are pseudo-random, relying on deterministic algorithms. Quantum measurement, dictated by the Born Rule, provides true randomness derived from the inherent probabilistic nature of quantum mechanics.
*   **CNOT gates allow modeling complex correlations**: Classical models often assume normal distributions and linear correlations. By using CNOT gates, we create quantum entanglement between qubits. This allows our QCBM to model complex, non-linear market correlations and fat-tailed distributions that classical models might miss.

## Installation & Setup

Ensure you have Python installed. You can install the required dependencies using:

```bash
pip install streamlit yfinance pandas numpy matplotlib qiskit qiskit_aer scipy pylatexenc
```

## Running the Application

To start the Streamlit dashboard, run the following command in your terminal:

```bash
streamlit run app.py
```
