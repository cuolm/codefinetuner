import os
import tree_sitter as ts
from tree_sitter_language_pack import get_parser
from pathlib import Path


def extract_code_blocks(node: ts.Node, source_code_utf8: bytes) -> list:
    logical_blocks = []
    block_types = {'function_definition'}

    if node.type in block_types:
        snippet = source_code_utf8[node.start_byte:node.end_byte]
        logical_blocks.append((snippet, node))

    for child in node.children:
        logical_blocks.extend(extract_code_blocks(child, source_code_utf8))
    return logical_blocks


def _extract_subblock_ranges(node: ts.Node, tree_sitter_subblock_types: set[str], base_offset: int) -> list:
    """
    Recursively depth-first search (DFS) the Abstract syntax tree (AST) and collect all subblock indices.
    Returns subblock ranges relative to the containing code block.
    E.g., a subblock range of (2, 30) means the subblock starts at byte position 2
    and ends at byte position 30, counting from the beginning of the specific code block
    (not the file).
    """
    subblock_ranges = []

    if node.type in tree_sitter_subblock_types:
        relative_start_byte = node.start_byte - base_offset
        relative_end_byte = node.end_byte - base_offset
        subblock_ranges.append((relative_start_byte, relative_end_byte))
    for child in node.children:
        child_subblock_ranges = _extract_subblock_ranges(child, tree_sitter_subblock_types, base_offset) 
        subblock_ranges.extend(child_subblock_ranges)
        
    return subblock_ranges


def _generate_fim_examples_from_code_block(
        code_utf8: bytes, 
        subblock_ranges: list, 
        bytes_per_token_ratio: float,
        min_middle_tokens_length: int, 
        max_middle_tokens_length: int
    ) -> list:
    fim_prefix_token_utf8 = "<|fim_prefix|>".encode('utf8')
    fim_suffix_token_utf8 = "<|fim_suffix|>".encode('utf8')
    fim_middle_token_utf8 = "<|fim_middle|>".encode('utf8')
    eos_token_utf8 = "<|endoftext|>".encode('utf8')
    
    fim_examples = [] 

    for r in subblock_ranges:
        middle_start_byte = r[0]
        middle_end_byte = r[1]
        middle_bytes_length = middle_end_byte - middle_start_byte
        middle_tokens_length = middle_bytes_length / bytes_per_token_ratio
        # allow only examples within a certain range
        if (middle_tokens_length < min_middle_tokens_length) or (middle_tokens_length > max_middle_tokens_length):
            continue

        prefix = code_utf8[:middle_start_byte]
        middle = code_utf8[middle_start_byte:middle_end_byte]
        suffix = code_utf8[middle_end_byte:]

        fim_example = (
            fim_prefix_token_utf8 + prefix +
            fim_suffix_token_utf8 + suffix +
            fim_middle_token_utf8 + middle +
            eos_token_utf8
        )

        fim_examples.append(fim_example)

    return fim_examples


def print_code_blocks_and_fim_examples(
        data_path: Path, 
        python_parser: ts.Parser, 
        extensions: tuple, 
        bytes_per_token_ratio: int, 
        bytes_per_code_block: int, 
        tree_sitter_subblock_types: set[str]
    ) -> None:         
    for root, _, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(extensions):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'rb') as f:
                        source_code_utf8 = f.read()
                        source_code_utf8.decode('utf-8')  # validate UTF-8, if not exception is raised, file skipped
                except UnicodeDecodeError:
                    print(f"Skipping file '{filename}': Not a valid UTF-8 file.")
                    continue
                
                tree = python_parser.parse(source_code_utf8)
                root_node = tree.root_node

                code_blocks = extract_code_blocks(root_node, source_code_utf8)

                for code_utf8, node in code_blocks:
                    code_utf8 = code_utf8[:bytes_per_code_block]  # trunctate code block if it is larger than bytes_per_code_block
                    base_offset = node.start_byte
                    subblock_ranges = _extract_subblock_ranges(node, tree_sitter_subblock_types, base_offset)
                    subblock_ranges = sorted(subblock_ranges, key=lambda x: x[1])

                    # discard subblocks that have a larger end index than bytes_per_code_block
                    i = 0
                    while (i < len(subblock_ranges)) and (subblock_ranges[i][1] <= bytes_per_code_block):
                        i += 1
                    subblock_ranges = subblock_ranges[:i]

                    print("=== Code Block Info ===")
                    print(f"Code:\n{code_utf8}")
                    print(f"Length: {len(code_utf8)}")
                    print(f"Subblock Indices: {subblock_ranges}\n")

                    print("=== Subblocks ===")
                    for idxs in subblock_ranges:
                        subblock_start, subblock_end = idxs
                        print(f"{code_utf8[subblock_start:subblock_end]}\n")

                    fim_examples = _generate_fim_examples_from_code_block(code_utf8, subblock_ranges, bytes_per_token_ratio, 10, 200)


                    print("=== FIM Examples ===")
                    for example in fim_examples:
                        print(f"{example}\n")


def main():
    project_root_path = Path(__file__).resolve().parent.parent
    data_path = project_root_path / 'data'
    extensions = (".c", ".h")
    tree_sitter_parser= get_parser("c")
    bytes_per_token_ratio = 3  # assume token to byte ration of 3
    max_bytes_per_func = bytes_per_token_ratio*50 

    logical_subblock_types = {
        "compound_statement",      # block of code enclosed in braces `{ ... }` representing a function body or other scoped block
        "parameter_list",          # the list of parameters in a function declaration or definition (e.g., `(int a, int b)`)
        "declaration",             # variable or other declarations (e.g., `int x;`)
        "expression_statement",    # statements that are expressions terminated by a semicolon (e.g., `x = y + 1;`)
        "if_statement",            # if-else conditional statement with optional else branch
        "while_statement",         # while loop statement
        "for_statement",           # for loop statement, including header and body
        "switch_statement",        # switch statement controlling multiple cases
        "case_statement",          # individual case or default in a switch block
        "return_statement"         # return statement returning an expression or void from a function
    }

    print_code_blocks_and_fim_examples(
        data_path, 
        tree_sitter_parser, 
        extensions, 
        bytes_per_token_ratio, 
        max_bytes_per_func, 
        logical_subblock_types
    )


if __name__ == "__main__":
    main()
