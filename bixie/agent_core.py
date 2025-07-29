# bixie/agent_core.py

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

from bixie.tasks.binary_scanner import run_binary_scan
from bixie.models.model_inference import run_model_inference
from bixie.vector_store.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bixie.agent_core")

def analyze_targets(target_paths: List[Path], output_dir: Path = None, save_vectors: bool = False, chroma_store: ChromaStore = None) -> Dict[str, List[Dict]]:
    """
    Main agent core logic to orchestrate scanning and model inference.

    Args:
        target_paths (List[Path]): List of file or directory paths to analyze.
        output_dir (Path): Optional path to directory to store vectorized output.
        save_vectors (bool): If True, save embedded vectors during inference.
        chroma_store (ChromaStore): Optional Chroma vector store to persist embeddings.

    Returns:
        Dict[str, List[Dict]]: Results including vulnerabilities and ML inferences.
    """
    results = {
        "vulnerabilities": [],
        "model_inferences": []
    }

    logger.info(f"Analyzing {len(target_paths)} targets...")

    # Run static analysis (e.g., Ghidra)
    vuln_findings = run_binary_scan(target_paths)
    results["vulnerabilities"].extend(vuln_findings)
    logger.info(f"Static analysis complete: {len(vuln_findings)} findings")

    # Run ML inference over binaries (e.g., embeddings/CVE match)
    model_outputs = run_model_inference(target_paths, output_dir=output_dir, save_vectors=save_vectors, chroma_store=chroma_store)
    results["model_inferences"].extend(model_outputs)
    logger.info(f"Model inference complete: {len(model_outputs)} items")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bixie Agent: Analyze binaries using Ghidra and ML models.")
    parser.add_argument("targets", nargs="+", help="Files or directories to analyze")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output results to JSON file")
    parser.add_argument("--vector-cache", type=str, default=None, help="Directory to store cached embeddings")
    

    args = parser.parse_args()
    paths = [Path(p).resolve() for p in args.targets]
    output_dir = Path("./vector-store").resolve()

    chroma_store = ChromaStore() 

    logger.info("Starting Bixie agent core...")
    results = analyze_targets(paths, output_dir=output_dir, save_vectors=args.save_vectors, chroma_store=chroma_store)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

