import os

def find_elf_files(root_dir):
    """Recursively find all ELF files in root_dir and subdirectories."""
    elf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.o') or filename.endswith('.elf'):
                elf_files.append(os.path.join(dirpath, filename))
    return elf_files

def benchmark_file(filepath):
    """Run benchmark on a single ELF file. Returns result dict."""
    # ...existing code for benchmarking a single file...
    # Replace the hardcoded file path with 'filepath'
    # Return a dict or result object
    pass  # Replace with actual benchmarking logic

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    elf_files = find_elf_files(data_dir)
    if not elf_files:
        print("No ELF files found in data directory.")
        return

    all_results = []
    for elf_file in elf_files:
        print(f"Benchmarking: {elf_file}")
        result = benchmark_file(elf_file)
        all_results.append((elf_file, result))

    print("\nBenchmark Summary:")
    for elf_file, result in all_results:
        print(f"{elf_file}: {result}")

if __name__ == "__main__":
    main()