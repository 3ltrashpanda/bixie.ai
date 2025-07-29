# 🛡️ Bixie – Autonomous Vulnerability Detection Framework

**Bixie** is an AI-powered cybersecurity research agent designed to automatically discover vulnerabilities in binary executables, source code, and cloud configurations. It blends machine learning, static/dynamic analysis, and cloud misconfiguration detection into an extensible agent-based architecture.

Inspired by modern AI penetration testing frameworks,  Bixie is built for red teams, threat researchers, and security engineers who want scalable automation with explainable results.

---

## ⚙️ Features

- 🔍 Binary analysis using SAFE/VulBERT embeddings, Ghidra, and BinDiff-style clustering  
- 🧠 Code vulnerability detection via semantic and AST-based analysis  
- ☁️ Cloud config scanning (Terraform, AWS, Azure) for misconfigs and privilege paths  
- 🪝 Extensible modules for CI/CD integration, alerting, and custom benchmarks  
- 🧪 Real-world CVE benchmarking suite included  

---

## 🚀 Getting Started

### 1. Clone the Project

```bash
git clone https://github.com/your-org/bixie.git
cd bixie

2. Install Dependencies

pip install -r requirements.txt

Optional: use conda or virtualenv.
3. Build Docker Images

docker-compose build

This builds containers for:

    Binary analysis (with Ghidra or Binwalk)

    Source scanning (Bandit, Semgrep, custom NLP)

    Cloud scanning (Checkov, custom Terraform logic)

4. Run the Agent

You can run a full scan locally or as a container:

python -m bixie.agent_core --target ./datasets/code_vulns/

Or use:

./scripts/run_agent.sh

5. Run Benchmarks

Use the benchmark suite to test Bixie against known vulnerable datasets:

cd ../bixie_benchmark/
./scripts/run_benchmark.sh
python scripts/eval_metrics.py

## Benchmarking Multiple ELF Files

Place your `.o` or `.elf` files anywhere under the `data/` directory (including subdirectories).  
Run the benchmark script:

```bash
python benchmark.py
```

The script will automatically discover and benchmark all ELF files under `data/`.

🧱 Repo Structure

bixie/
├── agent_core.py            # Main agent logic (dispatch, state, execution)
├── config.py                # Runtime and integration settings
├── tasks/
│   ├── binary_scanner.py
│   ├── code_scanner.py
│   ├── cloud_scanner.py
│   └── result_parser.py
├── models/
│   ├── safe_embedder.py
│   ├── vulbert_infer.py
│   └── utils.py
├── integrations/
│   ├── github.py
│   ├── cloud_apis.py
│   ├── siem.py
│   └── messaging.py
├── docker/                  # Dockerfiles for agent components
├── scripts/                 # CLI helpers
├── tests/                   # Unit tests
└── notebooks/               # Experiments, clustering, visualizations

🧠 How to Extend Bixie
➕ Add New Scan Module

    Create a new scanner under bixie/tasks/your_scanner.py

    Implement run() interface that returns findings in standard format

    Register it inside agent_core.py

🔌 Add Cloud Provider Support

Extend cloud_apis.py and map to your provider’s API (e.g., GCP, Azure).
🔎 Add New ML Model

    Drop model in models/ (ONNX, PyTorch, or Hugging Face format)

    Create wrapper script to load and use embeddings

    Optionally integrate into the clustering pipeline

📣 Send Results to External Tools

Use integrations/siem.py or messaging.py to push to:

    Sentinel

    Splunk

    Slack/MS Teams

    Webhooks

💡 Future Work & Ideas
Area	Idea
ML Expansion	Integrate GNN-based models like VulBERTa, or fine-tune SAFE on your binary corpus
Fuzzing	Add AFL++ or libFuzzer hooks to automatically fuzz and crash test identified inputs
Exploitation	Autonomous payload generation and exploit validation (XSS, RCE, IDOR)
Voice Interface	Voice control for operator prompts via Whisper + TTS
SIEM Loop	Auto-ingest and enrich live alert data to prioritize agent scans (threat intel loop)
K8s Defender	Build a mode for Kubernetes misconfiguration and in-cluster privilege path analysis
HackerOne AI	Align findings to bug bounty platforms with templated report output
