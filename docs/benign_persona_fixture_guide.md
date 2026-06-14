# Benign Persona Fixture Guide

This guide explains how to add and structure benign persona fixtures for evaluating false-positive rates in the clean VPN firewall pipeline.

## Folder Structure

Place all benign persona data under:

```
data/benign_personas/
  https_browsing/
  video_call/
  gaming_udp/
  rdp/
  cloud_backup/
  enterprise_proxy/
  streaming/
```

Each persona folder may contain:
- Raw PCAP or JSON flow files (if available)
- Pre-extracted clean feature CSV or Parquet files
- `metadata.json` describing the persona

## metadata.json Format

Example:
```json
{
  "persona_name": "Normal HTTPS Browsing",
  "source": "Synthetic diagnostic data or real capture",
  "collection_date": "2026-05-12",
  "expected_label": "benign",
  "notes": "No VPN, typical web browsing",
  "synthetic": true
}
```

## Adding a New Benign Scenario
1. Create a new folder under `data/benign_personas/`.
2. Add your data files (PCAP, flow JSON, or feature CSV/Parquet).
3. Add a `metadata.json` file as shown above.
4. Ensure feature files match the model's input schema (see `feature_columns.json`).

## Why Each Scenario is Risky
- **Video call**: Encrypted, bidirectional, stable timing—resembles VPN.
- **Gaming UDP**: Bursty, low-latency UDP—may look like tunneling.
- **RDP**: Encrypted remote tunnel-like interaction.
- **Cloud backup**: Long, encrypted upload sessions.
- **Enterprise proxy**: Aggregates traffic, can resemble tunneling.
- **Streaming**: Sustained encrypted flow, high throughput.

## Accepted File Formats
- PCAP (raw packet capture)
- JSON (flow records)
- CSV/Parquet (clean features)

## Required: Feature Schema Match
- Feature files must match the model's input columns exactly (see `feature_columns.json`).
- No missing or extra columns.
- No NaN or Inf values.

## Contact
For questions, contact the pipeline maintainer.
