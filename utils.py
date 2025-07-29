import os

def find_elf_files(root_dir):
    """Recursively find all ELF files in root_dir and subdirectories."""
    elf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.o') or filename.endswith('.elf'):
                elf_files.append(os.path.join(dirpath, filename))
    return elf_files