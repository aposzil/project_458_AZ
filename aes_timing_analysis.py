import time
import os
import random
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. THE MOCK AES IMPLEMENTATION
# ==========================================

def mock_aes_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Simulates an AES encryption.
    
    SIMULATED LEAKAGE: 
    To demonstrate that the harness works, this function simulates a 
    timing leak. If the first byte of the plaintext is '0x00' or '0xFF', 
    it performs extra mathematical operations, taking slightly longer.
    In a real scenario, you would replace this function call with your C wrapper.
    """
    # Verify input size (AES-128 standard block is 16 bytes)
    if len(plaintext) != 16:
        raise ValueError("Plaintext must be 16 bytes")

    # --- SIMULATE CRYPTO WORKLOAD (Busy wait) ---
    # We use a loop of math operations instead of time.sleep() because
    # time.sleep() has poor precision (OS scheduling noise).
    
    work_loops = 5000 
    
    # === ARTIFICIAL LEAK INJECTION ===
    # If the first byte is 0 (0x00), add a delay (e.g., representing a cache miss)
    if plaintext[0] == 0x00:
        work_loops += 1000  # Significant simulated delay
    # If the first byte is 255 (0xFF), add a smaller delay
    elif plaintext[0] == 0xFF:
        work_loops += 500
        
    # Perform the "Encryption"
    acc = 0
    for i in range(work_loops):
        acc += (i * i) % 12345
        
    return acc.to_bytes(16, 'big') # Return dummy ciphertext

# ==========================================
# 2. MEASUREMENT HARNESS
# ==========================================

def measure_execution_time(func, plaintext, key, num_runs=100):
    """
    Executes the encryption function multiple times and measures the total duration.
    
    Args:
        func: The encryption function to test.
        plaintext: 16-byte input.
        key: 16-byte key.
        num_runs: Number of encryptions to loop per measurement (to average out OS noise).
    
    Returns:
        Average execution time per encryption in nanoseconds.
    """
    # Choose the most precise timer available
    timer = time.perf_counter_ns
    
    # Warmup (optional, helps load code into CPU cache)
    func(plaintext, key)

    start_time = timer()
    for _ in range(num_runs):
        func(plaintext, key)
    end_time = timer()

    total_time_ns = end_time - start_time
    return total_time_ns / num_runs

# ==========================================
# 3. EXPERIMENT DRIVER (Input Generation)
# ==========================================

def run_experiment_suite():
    print("Starting Experimental Phase...")
    
    # Configuration
    KEY = os.urandom(16)
    SAMPLES_PER_BYTE = 50  # How many measurements per input class
    LOOPS_PER_SAMPLE = 100  # Encryptions inside the timer loop
    
    results = []

    print(f"Collecting data: Testing all 256 possible values for the first byte.")
    print(f"Total measurements: {256 * SAMPLES_PER_BYTE}")

    # We iterate through all possible values (0-255) for the first byte (Pt[0])
    # This corresponds to "Fixed first byte, others random" in the prompt.
    for byte_val in range(256):
        if byte_val % 50 == 0:
            print(f"Progress: Testing byte value {byte_val}/255...")
            
        for _ in range(SAMPLES_PER_BYTE):
            # 1. Generate Input
            # Create a random 16-byte block
            random_pt = bytearray(os.urandom(16))
            # Fix the first byte to the current test value
            random_pt[0] = byte_val
            plaintext = bytes(random_pt)

            # 2. Measure Time
            avg_time_ns = measure_execution_time(
                mock_aes_encrypt, 
                plaintext, 
                KEY, 
                num_runs=LOOPS_PER_SAMPLE
            )

            # 3. Store Data
            results.append({
                "pt_byte_0": byte_val,
                "time_ns": avg_time_ns,
                "type": "Random (Fixed Byte 0)"
            })
            
    # Add a Baseline check (All Zeros)
    print("Running Baseline check (All Zeros)...")
    zeros = bytes([0] * 16)
    for _ in range(SAMPLES_PER_BYTE):
        t = measure_execution_time(mock_aes_encrypt, zeros, KEY, num_runs=LOOPS_PER_SAMPLE)
        results.append({"pt_byte_0": 0, "time_ns": t, "type": "Structured (All Zeros)"})

    print("Data collection complete.")
    return pd.DataFrame(results)

# ==========================================
# 4. & 5. STATISTICAL ANALYSIS & VISUALIZATION
# ==========================================

def analyze_and_plot(df):
    print("\nRunning Statistical Analysis...")
    
    # -- 1. Basic Stats --
    # Calculate Mean and Variance for every byte value (0-255)
    # We exclude the specific "Structured" type for the main distribution plot
    main_data = df[df["type"] == "Random (Fixed Byte 0)"]
    
    stats_df = main_data.groupby("pt_byte_0")["time_ns"].agg(['mean', 'std', 'var'])
    print("\nTop 5 slowest input bytes (Simulated Leak Candidates):")
    print(stats_df.sort_values(by="mean", ascending=False).head(5))

    # -- 2. Visualization --
    sns.set_theme(style="whitegrid")
    
    # FIGURE A: Scatter/Line Plot of Execution Time vs First Byte Value
    # This is the classic side-channel view. If flat, no leak. If spikes, leak.
    plt.figure(figsize=(12, 6))
    
    # We plot the mean time for each byte value with an error bar (standard deviation)
    plt.errorbar(
        stats_df.index, 
        stats_df['mean'], 
        yerr=stats_df['std'], 
        fmt='o', 
        markersize=3, 
        ecolor='red', 
        capsize=2, 
        alpha=0.7, 
        label='Mean Execution Time'
    )
    
    plt.title("AES Execution Time vs. Value of 1st Plaintext Byte")
    plt.xlabel("Value of Plaintext Byte[0] (0-255)")
    plt.ylabel("Execution Time (ns)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("aes_timing_scatter.png")
    print("Saved plot: aes_timing_scatter.png")

    # FIGURE B: Histogram / Density
    # Shows the distribution of times. Bimodal distribution suggests leakage.
    plt.figure(figsize=(10, 6))
    sns.histplot(main_data['time_ns'], kde=True, bins=50)
    plt.title("Distribution of Execution Times")
    plt.xlabel("Time (ns)")
    plt.ylabel("Frequency")
    plt.savefig("aes_timing_histogram.png")
    print("Saved plot: aes_timing_histogram.png")
    
    # FIGURE C: Boxplot to highlight outliers
    # We define "Leaky" vs "Normal" groups based on our specific knowledge of the mock
    # In a real analysis, you would group by 'High Hamming Weight' vs 'Low', etc.
    plt.figure(figsize=(12, 6))
    # Let's filter just a few bytes to make the boxplot readable
    subset_bytes = [0, 1, 100, 200, 255] 
    subset_df = main_data[main_data['pt_byte_0'].isin(subset_bytes)]
    
    sns.boxplot(x='pt_byte_0', y='time_ns', data=subset_df)
    plt.title("Timing Variation for Specific Input Bytes")
    plt.savefig("aes_timing_boxplot.png")
    print("Saved plot: aes_timing_boxplot.png")
    
    plt.show()

if __name__ == "__main__":
    # Run the full pipeline
    dataset = run_experiment_suite()
    analyze_and_plot(dataset)