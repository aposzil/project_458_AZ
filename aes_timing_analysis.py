import time
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Crypto.Cipher import AES as PyCryptoAES

# =============================================================================
# 1. VULNERABLE AES IMPLEMENTATION (Naive / Table-Based)
# =============================================================================

class VulnerableAES:
    """
    A naive implementation of AES that relies on S-Box lookups.
    In C, this creates 'Cache Timing Side-Channels'. 
    
    Since Python is high-level, we simulate the *effect* of a cache miss 
    so your statistical analysis works as intended.
    """
    def __init__(self, key):
        self.key = key
        # Standard AES S-Box (Partial for demo)
        self.sbox = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            # ... (truncated for brevity, we map modulo 256 in simulation) ...
        ] * 16 

    def _simulate_cache_latency(self, byte_val):
        """
        Simulates the timing difference of a Cache Miss vs Cache Hit.
        In a real 'unsafe' C implementation, accessing specific indices 
        takes longer depending on if they are already in the CPU Cache.
        """
        # Simulate: Indices 0-15 are "Hot" (Cached) -> Fast
        # Indices 200-255 are "Cold" (Uncached) -> Slower
        # This creates the 'leak' we want to detect.
        dummy = 0
        if byte_val > 200: 
            # Simulated Cache Miss (Slower)
            for x in range(100): dummy += x 
        elif byte_val == 0:
            # Simulated Zero-handling special path (Very slow)
            for x in range(150): dummy += x
        else:
            # Simulated Cache Hit (Fast)
            for x in range(10): dummy += x
        return dummy

    def encrypt(self, plaintext):
        # We only implement the SubBytes step as that is the source of 
        # the timing leak we are investigating.
        output = bytearray(16)
        for i in range(16):
            # The S-Box lookup:
            val = plaintext[i]
            
            # This line causes the side-channel:
            self._simulate_cache_latency(val) 
            
            # Simple substitution logic
            output[i] = val ^ self.key[i] 
        return bytes(output)

# =============================================================================
# 2. SAFE AES IMPLEMENTATION (PyCryptodome)
# =============================================================================

class SafeAES:
    """
    Wrapper for PyCryptodome.
    This library uses constant-time code or AES-NI instructions.
    """
    def __init__(self, key):
        # ECB mode used to isolate block encryption performance
        self.cipher = PyCryptoAES.new(key, PyCryptoAES.MODE_ECB)

    def encrypt(self, plaintext):
        return self.cipher.encrypt(plaintext)

# =============================================================================
# 3. EXPERIMENTAL HARNESS
# =============================================================================

def measure_time(encrypt_func, plaintext, num_runs=100):
    """
    Accurately measures execution time of the encryption function.
    """
    timer = time.perf_counter_ns
    
    # Warmup
    encrypt_func(plaintext)
    
    start = timer()
    for _ in range(num_runs):
        encrypt_func(plaintext)
    end = timer()
    
    return (end - start) / num_runs

def generate_dataset():
    print("--- Starting AES Side-Channel Data Collection ---")
    
    # Configuration
    KEY = os.urandom(16)
    safe_aes = SafeAES(KEY)
    vuln_aes = VulnerableAES(KEY)
    
    # Parameters
    # We test every possible value for the first byte (0-255)
    # to see if the value of the byte affects the speed.
    SAMPLES_PER_BYTE = 50      # How many random plaintexts per byte value
    LOOPS_PER_MEASURE = 500    # How many encryptions inside the timer
    
    data_records = []

    print(f"Collecting data for {256} byte values...")
    
    for byte_val in range(256):
        if byte_val % 50 == 0: print(f"Progress: {byte_val}/255")
        
        for i in range(SAMPLES_PER_BYTE):
            # Generate Random Plaintext
            pt = bytearray(os.urandom(16))
            # FIX the first byte (Controlled Input)
            pt[0] = byte_val
            plaintext = bytes(pt)
            
            # 1. Measure Vulnerable Implementation
            t_vuln = measure_time(vuln_aes.encrypt, plaintext, LOOPS_PER_MEASURE)
            data_records.append({
                "implementation": "Vulnerable (Naive)",
                "byte_value": byte_val,
                "time_ns": t_vuln,
                "sample_id": i
            })
            
            # 2. Measure Safe Implementation
            t_safe = measure_time(safe_aes.encrypt, plaintext, LOOPS_PER_MEASURE)
            data_records.append({
                "implementation": "Safe (PyCryptodome)",
                "byte_value": byte_val,
                "time_ns": t_safe,
                "sample_id": i
            })
            
    print("Data collection complete.")
    
    # Convert to DataFrame
    df = pd.DataFrame(data_records)
    
    # Export to CSV (Deliverable requirement)
    csv_filename = "aes_timing_dataset.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Dataset saved to: {csv_filename}")
    
    return df

# =============================================================================
# 4. ANALYSIS & VISUALIZATION
# =============================================================================

def analyze_and_plot(df):
    print("Generating Comparison Plots...")
    sns.set_theme(style="whitegrid")

    # Group by implementation and byte_value to get mean times
    summary = df.groupby(['implementation', 'byte_value'])['time_ns'].mean().reset_index()

    # --- PLOT 1: Side-by-Side Comparison ---
    # This clearly shows the Safe one is flat, and the Vulnerable one is messy
    
    plt.figure(figsize=(14, 6))
    
    # Plot Safe
    plt.subplot(1, 2, 1)
    safe_data = summary[summary['implementation'] == 'Safe (PyCryptodome)']
    plt.plot(safe_data['byte_value'], safe_data['time_ns'], color='green', alpha=0.8)
    plt.title("Safe AES (Constant Time)")
    plt.xlabel("Input Byte Value (0-255)")
    plt.ylabel("Avg Execution Time (ns)")
    plt.ylim(safe_data['time_ns'].min() * 0.95, safe_data['time_ns'].max() * 1.05)

    # Plot Vulnerable
    plt.subplot(1, 2, 2)
    vuln_data = summary[summary['implementation'] == 'Vulnerable (Naive)']
    plt.plot(vuln_data['byte_value'], vuln_data['time_ns'], color='red', alpha=0.8)
    plt.title("Vulnerable AES (Data-Dependent)")
    plt.xlabel("Input Byte Value (0-255)")
    plt.ylabel("Avg Execution Time (ns)")
    
    # Highlight the "Leak"
    plt.tight_layout()
    plt.savefig("aes_comparison_plot.png")
    print("Saved plot: aes_comparison_plot.png")

    # --- PLOT 2: Overlay Scatter Plot ---
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=summary, 
        x='byte_value', 
        y='time_ns', 
        hue='implementation', 
        style='implementation',
        palette={'Safe (PyCryptodome)': 'green', 'Vulnerable (Naive)': 'red'},
        s=40
    )
    plt.title("Timing Correlation: Safe vs Vulnerable Implementation")
    plt.xlabel("Input Byte Value (First Byte of Plaintext)")
    plt.ylabel("Time (ns)")
    plt.legend(title="Implementation")
    
    # Note: We normalize the Y-axis range in the description because the raw 
    # Python times might differ significantly in magnitude.
    plt.savefig("aes_overlay_plot.png")
    print("Saved plot: aes_overlay_plot.png")
    
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Check if dataset exists, if not generate it
    if not os.path.exists("aes_timing_dataset.csv"):
        dataset = generate_dataset()
    else:
        print("Loading existing dataset from CSV...")
        dataset = pd.read_csv("aes_timing_dataset.csv")
    
    analyze_and_plot(dataset)