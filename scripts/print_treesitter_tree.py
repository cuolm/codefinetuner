#!/usr/bin/env python3
import argparse
import tree_sitter as ts
from tree_sitter_language_pack import get_language, get_parser

def print_tree(node: ts.Node, source_bytes: bytes, indent: int = 0):
    prefix = "  " * indent
    snippet = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    snippet = snippet.split("\n")[0][:60].replace("\t", " ")
    node_col = f"{prefix}{node.type}"
    print(f"{node_col:<50}  {repr(snippet)}")
    for child in node.children:
        print_tree(child, source_bytes, indent + 1)

def main():
    parser = argparse.ArgumentParser(description="Print tree-sitter AST of a code file.")
    parser.add_argument("-f", "--file", required=True, help="Path to the source file")
    parser.add_argument("-l", "--language", required=True, help="Language (e.g. python, javascript, rust)")
    args = parser.parse_args()

    with open(args.file, "rb") as fh:
        source = fh.read()

    ts_parser = get_parser(args.language)
    tree: ts.Tree = ts_parser.parse(source)
    print_tree(tree.root_node, source)

if __name__ == "__main__":
    main()