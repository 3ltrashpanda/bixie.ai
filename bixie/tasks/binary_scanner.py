import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Dict
import magic

GHIDRA_PATH = "/usr/share/ghidra/Ghidra/Features/PyGhidra/ghidra_scripts"  
GHIDRA_SCRIPT_NAME = "bixie_ghidra_script.py"        

magic_instance = magic.Magic(mime=True)

EXECUTABLE_MIME_TYPES = {
    "application/x-executable",             # ELF
    "application/x-pie-executable",
    "application/x-sharedlib",
    "application/x-dosexec",                # PE/EXE/DLL (Windows)
    "application/x-mach-binary",            # Mach-O
}

DOCUMENT_MIME_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}


def detect_file_type(file_path: Path) -> str:
    try:
        return magic_instance.from_file(str(file_path))
    except Exception:
        return "unknown"


def is_supported_file(file_path: Path) -> bool:
    mime_type = detect_file_type(file_path)
    return mime_type in EXECUTABLE_MIME_TYPES.union(DOCUMENT_MIME_TYPES)


def collect_targets_from_args(args: List[str]) -> List[Path]:
    """
    Recursively scan CLI args (files or directories) and return all supported binaries.
    """
    targets = []
    for arg in args:
        path = Path(arg)
        if not path.exists():
            print(f"[WARN] Path not found: {arg}")
            continue
        if path.is_file():
            if is_supported_file(path):
                targets.append(path)
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and is_supported_file(f):
                    targets.append(f)
    return targets


def run_ghidra_analysis(binary_path: Path) -> List[Dict]:
    """
    Run Ghidra headless analysis on a single binary file and parse results.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        project_dir = Path(tempdir) / "bixie_ghidra_project"
        project_dir.mkdir(parents=True, exist_ok=True)
        output_file = project_dir / "ghidra_output.json"

        cmd = [
            GHIDRA_PATH,
            str(project_dir),
            "bixie_project",
            "-import", str(binary_path),
            "-postScript", GHIDRA_SCRIPT_NAME, str(output_file),
            "-deleteProject",
            "-overwrite"
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Ghidra failed on {binary_path}:\n{proc.stderr}")

        if not output_file.exists():
            raise FileNotFoundError(f"Ghidra output file missing: {output_file}")

        with open(output_file, "r") as f:
            raw_findings = json.load(f)

        normalized = []
        for item in raw_findings:
            normalized.append({
                "vuln": item.get("issue", "Unknown vulnerability"),
                "desc": f"Function: {item.get('function')} at {item.get('address')}",
                "location": str(binary_path)
            })

        return normalized


def run_binary_scan(targets: List[Path]) -> List[Dict]:
    """
    Run binary scans on a list of binary file paths.
    """
    all_results = []
    for binary_path in targets:
        try:
            findings = run_ghidra_analysis(binary_path)
            all_results.extend(findings)
        except Exception as e:
            print(f"[ERROR] Failed to analyze {binary_path}: {e}")
    return all_results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python binary_scanner.py <file_or_directory> [...]")
        sys.exit(1)

    input_targets = collect_targets_from_args(sys.argv[1:])
    if not input_targets:
        print("[INFO] No supported files found.")
        sys.exit(0)

    results = run_binary_scan(input_targets)
    print(json.dumps(results, indent=2))

