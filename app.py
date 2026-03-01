import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import logic modules
import finance_logic as fl
import quantum_logic as ql

# Configure Streamlit page
st.set_page_config(page_title="Quantum-Enhanced Financial Risk & Scenario Analysis", layout="wide")

st.title("Quantum-Enhanced Financial Risk & Scenario Analysis")
st.markdown("### Amrita QuantumLeap Bootcamp 2026 Hackathon")

# Sidebar for User Input
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Ticker Symbol (e.g., AAPL, NVDA, TSLA, BTC-USD)", value="AAPL")

    st.markdown("---")
    st.markdown("### Educational Context")
    st.info(
        "**Quantum Measurement & True Randomness:**\n\n"
        "Classical random number generators are pseudo-random, relying on deterministic algorithms. "
        "Quantum measurement, dictated by the Born Rule, provides **true randomness** derived from the inherent "
        "probabilistic nature of quantum mechanics.\n\n"
        "**CNOT Gates & Correlation:**\n\n"
        "Classical models often assume normal distributions and linear correlations. By using **CNOT gates**, "
        "we create **quantum entanglement** between qubits. This allows our Quantum Circuit Born Machine (QCBM) "
        "to model complex, non-linear market correlations and fat-tailed distributions that classical models might miss."
    )

if ticker:
    try:
        # 1. Frontend & Data Loading
        with st.spinner(f"Downloading 2 years of daily returns for {ticker}..."):
            returns, mu, sigma, last_price = fl.fetch_data(ticker, period="2y")

        st.success(f"Successfully loaded data for {ticker}. Last Price: ${last_price:.2f}")

        # 2. Classical Analysis & Scenario Assembler
        hist_var, param_var = fl.calculate_var(returns, mu, sigma)

        with st.spinner("Running Monte Carlo Simulation..."):
            price_paths = fl.monte_carlo_simulation(last_price, mu, sigma, days=1008, num_simulations=100)
            avg_max_drawdown = fl.calculate_max_drawdown(price_paths)

        # 3. Quantum Engine (QCBM)
        with st.spinner("Building and running Quantum Circuit..."):
            qc = ql.create_qcbm(mu, sigma)

            # Extract measurement probabilities (counts)
            counts = ql.run_circuit_and_get_samples(qc, shots=1024)

            # Map to scenario probability weightings
            quantum_sampled_returns = ql.map_bitstrings_to_returns(counts, mu, sigma, shots=1024)

            # Calculate Quantum VaR
            quantum_var = ql.calculate_quantum_var(quantum_sampled_returns)

        # 4. Interactive Visualizations

        # Metric Cards
        st.subheader("Key Risk Metrics")
        col1, col2, col3, col4 = st.columns(4)

        ann_volatility = sigma * np.sqrt(252)

        with col1:
            st.metric("Volatility (Annualized)", f"{ann_volatility:.2%}")
        with col2:
            st.metric("Avg Max Drawdown (4 Yr)", f"{avg_max_drawdown:.2%}")
        with col3:
            st.metric("Classical Hist VaR (95%)", f"{hist_var:.2%}")
        with col4:
            st.metric("Quantum VaR (95%)", f"{quantum_var:.2%}")

        st.markdown("---")

        # Circuit Diagram and 4-Year Scenario Plot
        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.subheader("Quantum Circuit Diagram (QCBM)")
            fig_circuit = qc.draw(output='mpl')
            st.pyplot(fig_circuit)

        with col_c2:
            st.subheader("4-Year Scenario Plot (Classical Monte Carlo)")
            fig_mc, ax_mc = plt.subplots(figsize=(8, 5))
            # Plot a subset to avoid clutter
            for i in range(min(50, price_paths.shape[0])):
                ax_mc.plot(price_paths[i, :], lw=1, alpha=0.5)
            ax_mc.set_title(f"Projected Price Paths for {ticker} (Next 4 Years)")
            ax_mc.set_xlabel("Trading Days")
            ax_mc.set_ylabel("Price")
            st.pyplot(fig_mc)

        st.markdown("---")

        # Comparison Histograms
        st.subheader("Comparison: Classical vs. Quantum Returns Distribution")

        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Classical Returns Histogram
        ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(hist_var, color='red', linestyle='dashed', linewidth=2, label=f'Hist VaR: {hist_var:.2%}')
        ax1.axvline(param_var, color='orange', linestyle='dashed', linewidth=2, label=f'Param VaR: {param_var:.2%}')
        ax1.set_title("Historical Classical Returns")
        ax1.set_xlabel("Daily Return")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # Quantum Returns Histogram
        ax2.hist(quantum_sampled_returns, bins=16, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(quantum_var, color='red', linestyle='dashed', linewidth=2, label=f'Quantum VaR: {quantum_var:.2%}')
        ax2.set_title("Quantum Sampled Returns (QCBM)")
        ax2.set_xlabel(f"Return State [$\mu-3\sigma, \mu+3\sigma$]")
        ax2.set_ylabel("Measurement Counts")
        ax2.legend()

        st.pyplot(fig_hist)

        st.markdown("---")

        # 5. Final Risk Assessment
        st.subheader("Final Quantum-Enhanced Risk Assessment")

        # Calculate a simple composite risk score (0-100)
        # Higher Volatility -> Higher Risk (Cap at 100% vol for scoring)
        vol_score = min((ann_volatility / 1.0) * 100, 100)

        # Higher Drawdown -> Higher Risk (Drawdown is negative, so take abs)
        dd_score = min((abs(avg_max_drawdown) / 0.8) * 100, 100)

        # Higher Quantum VaR (more negative) -> Higher Risk (Cap at -15% daily VaR for scoring)
        var_score = min((abs(quantum_var) / 0.15) * 100, 100)

        # Weighted Average Score
        final_score = (0.4 * vol_score) + (0.3 * dd_score) + (0.3 * var_score)
        final_score = min(max(final_score, 0), 100)

        col_risk1, col_risk2 = st.columns([1, 2])

        with col_risk1:
            st.metric(label="Overall Risk Score (0-100)", value=f"{final_score:.0f}")

        with col_risk2:
            if final_score < 35:
                st.success("### Status: Low Risk (Not Highly Risky)")
                st.write(f"Based on quantum and classical metrics, **{ticker}** exhibits relatively stable price action with limited downside projected. It is not currently considered a highly risky asset.")
            elif final_score < 65:
                st.warning("### Status: Moderate Risk")
                st.write(f"Based on quantum and classical metrics, **{ticker}** exhibits moderate volatility and drawdown potential. The stock carries standard market risk.")
            else:
                st.error("### Status: High Risk (Risky)")
                st.write(f"Based on quantum and classical metrics, **{ticker}** exhibits significant volatility, large drawdown potential, and a severe Quantum Value at Risk. This asset is considered highly risky.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer Educational Context
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
    "<i>Quantum Measurement provides true randomness via the Born Rule. "
    "CNOT gates allow us to model entanglements and correlations that classical normal distributions might miss.</i>"
    "</div>",
    unsafe_allow_html=True
)
