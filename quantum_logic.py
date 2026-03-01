import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def create_qcbm(mu, sigma):
    """Build a 4-qubit Quantum Circuit Born Machine (QCBM) using Qiskit."""
    qc = QuantumCircuit(4, 4)

    # Apply Hadamard (H) for superposition
    for i in range(4):
        qc.h(i)

    # Apply RY gates to encode the Mean (mu)
    for i in range(4):
        qc.ry(mu, i)

    # Apply RZ gates to encode Volatility (sigma)
    for i in range(4):
        qc.rz(sigma, i)

    # Apply CNOT gates to create market entanglement
    for i in range(3):
        qc.cx(i, i+1)
    qc.cx(3, 0)

    # Measure
    qc.measure(range(4), range(4))

    return qc

def run_circuit_and_get_samples(qc, shots=1024):
    """Run on AerSimulator for 1024 shots."""
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts

def map_bitstrings_to_returns(counts, mu, sigma, shots=1024):
    """Use the Born Rule to translate measurement probabilities into Scenario Probability Weightings."""
    # We have 16 bitstrings (from '0000' to '1111')
    # We map them linearly from [mu - 3*sigma, mu + 3*sigma]

    min_return = mu - 3 * sigma
    max_return = mu + 3 * sigma

    # 16 possible states
    num_states = 16
    return_values = np.linspace(min_return, max_return, num_states)

    # Create mapping from bitstring to index (0 to 15)
    sampled_returns = []

    for bitstring, count in counts.items():
        # Convert bitstring to integer index
        index = int(bitstring, 2)

        # Get corresponding return value
        ret_val = return_values[index]

        # Add to our sampled returns based on the count (probability weighting)
        sampled_returns.extend([ret_val] * count)

    return np.array(sampled_returns)

def calculate_quantum_var(sampled_returns, confidence_level=0.95):
    """Calculate 95% Value at Risk (VaR) using the quantum samples."""
    return np.percentile(sampled_returns, (1 - confidence_level) * 100)
