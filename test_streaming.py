"""Quick test: verify streaming USBVPN parsing works on a large file."""
import gc
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.clean_pipeline.usbvpn_parser import iter_usbvpn_flows
from src.clean_pipeline.feature_extractor import extract_flow_features
import numpy as np

# Pick a large file to test
large_file = Path("data/raw/usbvpn/vpn/L2TP/streaming.json")  # ~200 MB
if not large_file.exists():
    print(f"File not found: {large_file}")
    sys.exit(1)

file_mb = large_file.stat().st_size / (1024 * 1024)
print(f"Testing streaming on {large_file.name} ({file_mb:.1f} MB)...")

t0 = time.time()
count = 0
for flow in iter_usbvpn_flows(large_file, min_packets=3):
    # Extract features immediately (like the pipeline does)
    feat = extract_flow_features(
        np.asarray(flow["timestamps"], dtype=np.float64),
        np.asarray(flow["sizes"], dtype=np.float64),
        np.asarray(flow["directions"], dtype=np.int32),
        max_packets=300,
    )
    count += 1
    if count % 1000 == 0:
        gc.collect()
        elapsed = time.time() - t0
        print(f"  {count} flows processed in {elapsed:.1f}s...")

    # Stop after 5000 to keep test quick
    if count >= 5000:
        print(f"  (stopped at {count} for speed test)")
        break

elapsed = time.time() - t0
print(f"\nDone: {count} flows from {file_mb:.1f} MB file in {elapsed:.1f}s")
print(f"Rate: {count/elapsed:.0f} flows/sec")
print("Memory OK — no crash!")

