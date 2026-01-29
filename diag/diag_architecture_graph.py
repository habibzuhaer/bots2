# diag/diag_architecture_graph.py
"""
ARCHITECTURE GRAPH DIAGNOSTIC

Цель:
- найти циклические импорты
- проверить направление зависимостей
- зафиксировать архитектурные слои
"""

import ast
import os
from collections import defaultdict

PROJECT_ROOT = "."


ALLOWED_DEPENDENCIES = {
    "cme": [],
    "context": ["cme"],
    "filters": ["context", "cme"],
    "engine": ["context", "filters", "cme"],
    "diag": ["engine", "context", "filters", "cme"],
}


def module_layer(path):
    parts = path.split(os.sep)
    return parts[0] if parts else None


def parse_imports(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def run():
    graph = defaultdict(set)

    for root, _, files in os.walk(PROJECT_ROOT):
        for file in files:
            if not file.endswith(".py"):
                continue

            path = os.path.join(root, file)
            layer = module_layer(root.replace(PROJECT_ROOT + os.sep, ""))

            if not layer or layer.startswith("."):
                continue

            imports = parse_imports(path)
            for imp in imports:
                graph[layer].add(imp)

    print("\nARCHITECTURE DEPENDENCY CHECK\n")

    for layer, deps in graph.items():
        allowed = set(ALLOWED_DEPENDENCIES.get(layer, []))
        illegal = deps - allowed - {layer}

        if illegal:
            raise RuntimeError(
                f"ILLEGAL DEPENDENCY: {layer} → {illegal}"
            )

        print(f"OK   | {layer} → {sorted(deps)}")

    print("\nARCHITECTURE GRAPH OK")


if __name__ == "__main__":
    run()
