import os
import ast
import networkx as nx
import matplotlib.pyplot as plt
from radon.complexity import cc_visit

# List of Python files to analyze
FILES = [
    "rule_based.py",
    "ml.py",
    "dl.py",
    "dashboard.py"
]

def analyze_cyclomatic_complexity(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        code = file.read()
    results = cc_visit(code)
    complexity_data = []
    for item in results:
        if hasattr(item, 'name'):
            complexity_data.append((item.name, item.lineno, item.complexity))
    return complexity_data

def build_cfg_ast(file_path):
    """Build a simple control-flow-like graph from function definitions."""
    with open(file_path, "r", encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=file_path)

    G = nx.DiGraph()
    prev_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            G.add_node(node.name)
            if prev_func:
                G.add_edge(prev_func, node.name)
            prev_func = node.name
    return G

def plot_cfg(graph, title):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightgreen', node_size=2200,
            font_size=10, font_weight='bold', edge_color='black')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Process each file
for path in FILES:
    print(f"\n🔍 Analyzing {path}...")

    # Cyclomatic Complexity
    cc_data = analyze_cyclomatic_complexity(path)
    print("📈 Cyclomatic Complexity:")
    for name, line, complexity in cc_data:
        print(f"  {name} (line {line}): Complexity {complexity}")

    # Plot CFG
    cfg = build_cfg_ast(path)
    if cfg.number_of_nodes() > 0:
        print("🧩 Displaying Control Flow Graph (function-level)...")
        plot_cfg(cfg, f"CFG: {path}")
    else:
        print("⚠️ No functions found for CFG.")
