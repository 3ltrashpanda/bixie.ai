#@pandarat 
#@category Analysis

import json
import os

results = []

# Find all functions calling 'strcpy' as a simple unsafe pattern example
for func in currentProgram.getFunctionManager().getFunctions(True):
    instructions = currentProgram.getListing().getInstructions(func.getBody(), True)
    for inst in instructions:
        if inst.toString().startswith("CALL") and "strcpy" in inst.toString():
            results.append({
                "function": func.getName(),
                "address": str(inst.getAddress()),
                "issue": "Potential unsafe strcpy call"
            })

# Get output file path from script args
output_path = getScriptArgs()[0]

with open(output_path, "w") as f:
    f.write(json.dumps(results))

