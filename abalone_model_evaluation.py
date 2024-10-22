import matplotlib.pyplot as plt
import numpy as np

# Data based on market projections for the quantum computing industry
years = np.array([2024, 2026, 2028, 2030, 2032, 2034, 2036])
quantum_market = np.array([1.65, 5.2, 10.8, 18.6, 26.5, 34.1, 39.7])  # Projected market growth in billions USD
quantum_finance_share = np.array([0, 0.5, 1.2, 2.5, 4.0, 5.5, 6.0])  # QuantumFinance's anticipated share in billions USD

# Create the plot
plt.figure(figsize=(10, 6))

# Plotting both lines
plt.plot(years, quantum_market, label='Quantum Computing Market', marker='o', linestyle='-', color='blue')
plt.plot(years, quantum_finance_share, label='QuantumFinance Market Share', marker='o', linestyle='--', color='green')

# Adding titles and labels
plt.title('Market Growth Projections for Quantum Computing and QuantumFinance', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Market Size (Billions of USD)', fontsize=12)

# Adding a grid, legend, and setting limits for better readability
plt.grid(True)
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()
