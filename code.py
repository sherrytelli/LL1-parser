import sys
from collections import defaultdict
from tabulate import tabulate

# Represents a node in the parse tree
class Node:
    """A node for the parse tree."""
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def __repr__(self):
        return f"Node({self.value})"

class LL1Parser:
    """
    Implements a full LL(1) parser, including grammar transformation,
    set computation, table generation, and string parsing.
    """

    def __init__(self, grammar_file):
        print(f"--- Initializing Parser with '{grammar_file}' ---")
        self.grammar = defaultdict(list)
        self.original_grammar = defaultdict(list)
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = None
        self.nullable = set()
        self.first = defaultdict(set)
        self.follow = defaultdict(set)
        self.table = {}
        self.is_ll1 = True
        self.conflicts = []

        try:
            self._read_grammar(grammar_file)
            self._print_grammar(self.original_grammar, "1. Original Grammar")
            
            # 1. Eliminate Left Recursion
            self._eliminate_left_recursion()
            self._print_grammar(self.grammar, "2. Grammar after Eliminating Left Recursion")

            # 2. Apply Left Factoring
            self._left_factor()
            self._print_grammar(self.grammar, "3. Grammar after Left Factoring")

            # 3. Compute Sets
            print("\n--- 4. Computing NULLABLE, FIRST, and FOLLOW sets ---")
            self._compute_nullable()
            self._compute_first()
            self._compute_follow()
            self._print_sets()

            # 4. Construct Parsing Table
            print("\n--- 5. Constructing Predictive Parsing Table ---")
            self._build_parsing_table()
            self._print_table()

            # 5. Check for Conflicts
            print("\n--- 6. Checking for LL(1) Conflicts ---")
            if not self.is_ll1:
                print("Grammar is NOT LL(1). Conflicts detected:")
                for conflict in self.conflicts:
                    print(f"  - {conflict}")
            else:
                print("Grammar is LL(1). No conflicts found.")

        except Exception as e:
            print(f"An error occurred during initialization: {e}", file=sys.stderr)
            self.is_ll1 = False # Halt parsing

    def _read_grammar(self, file_path):
        """Reads a grammar from a file. Format: A -> B C | e"""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                head, body = line.split('->')
                head = head.strip()
                
                if self.start_symbol is None:
                    self.start_symbol = head
                
                self.non_terminals.add(head)
                
                productions = [p.strip() for p in body.split('|')]
                for prod_str in productions:
                    symbols = prod_str.split()
                    if symbols == ['e']: # 'e' represents epsilon
                        self.grammar[head].append(['e'])
                        self.original_grammar[head].append(['e'])
                    else:
                        self.grammar[head].append(symbols)
                        self.original_grammar[head].append(symbols)
        
        self._derive_terminals()

    def _derive_terminals(self):
        """Finds all terminals based on what's not a non-terminal."""
        self.terminals.clear()
        for prods in self.grammar.values():
            for prod in prods:
                for symbol in prod:
                    if symbol not in self.non_terminals and symbol != 'e':
                        self.terminals.add(symbol)

    def _eliminate_left_recursion(self):
        """Eliminates immediate left recursion from the grammar."""
        new_grammar = defaultdict(list)
        nt_list = list(self.grammar.keys()) # Fixed order
        
        for i in range(len(nt_list)):
            nt = nt_list[i]
            prods = self.grammar[nt]
            
            # Step 1: Substitution (for indirect LR)
            substituted_prods = []
            for prod in prods:
                if prod[0] in nt_list[:i]: # If starts with a processed NT
                    nt_j = prod[0]
                    suffix = prod[1:]
                    for j_prod in new_grammar[nt_j]: # Use the *new* rules for j
                        substituted_prods.append(j_prod + suffix)
                else:
                    substituted_prods.append(prod)
            prods = substituted_prods

            # Step 2: Eliminate immediate LR
            alphas = [] # Recursive productions (A -> A alpha)
            betas = []  # Non-recursive productions (A -> beta)
            
            for prod in prods:
                if prod[0] == nt:
                    alphas.append(prod[1:])
                else:
                    betas.append(prod)

            if alphas: # If there is left recursion
                new_nt = nt + "'"
                while new_nt in self.grammar or new_nt in new_grammar:
                    new_nt += "'"
                
                self.non_terminals.add(new_nt)
                
                # A -> beta A'
                for beta in betas:
                    if beta == ['e']: # Handle A -> e
                         new_grammar[nt].append([new_nt])
                    else:
                        new_grammar[nt].append(beta + [new_nt])
                
                # A' -> alpha A' | e
                for alpha in alphas:
                    new_grammar[new_nt].append(alpha + [new_nt])
                new_grammar[new_nt].append(['e'])
            else:
                # No LR, just copy the (potentially substituted) rules
                new_grammar[nt].extend(prods)
                
        self.grammar = new_grammar
        self._derive_terminals() # Update terminals

    def _find_lcp(self, p1, p2):
        """Finds the longest common prefix between two productions."""
        lcp = []
        for i in range(min(len(p1), len(p2))):
            if p1[i] == p2[i]:
                lcp.append(p1[i])
            else:
                break
        return lcp

    def _left_factor(self):
        """Performs left factoring on the grammar until no changes occur."""
        changed = True
        while changed:
            changed = False
            new_grammar = defaultdict(list)
            nt_to_add = {} # To avoid modifying dict while iterating
            
            for nt, prods in self.grammar.items():
                # Group productions by their LCP
                sorted_prods = sorted(prods)
                i = 0
                while i < len(sorted_prods):
                    current_prod = sorted_prods[i]
                    # Find all other prods with the same LCP
                    group = [current_prod]
                    lcp = current_prod # LCP is at least the prod itself
                    
                    j = i + 1
                    while j < len(sorted_prods):
                        current_lcp = self._find_lcp(current_prod, sorted_prods[j])
                        if current_lcp: # If there's a common prefix
                            lcp = current_lcp # LCP can only get shorter
                            group = [p for p in sorted_prods[i:] if self._find_lcp(p, current_prod) == lcp]
                            j = i + len(group)
                            break
                        else:
                            j += 1 # No LCP, move to next
                    
                    if len(group) > 1:
                        changed = True
                        new_nt = nt + "'"
                        while new_nt in self.grammar or new_nt in nt_to_add:
                            new_nt += "'"
                        
                        # Add A -> lcp A'
                        new_grammar[nt].append(lcp + [new_nt])
                        
                        # Add rules for A'
                        for p in group:
                            suffix = p[len(lcp):]
                            nt_to_add.setdefault(new_nt, []).append(suffix if suffix else ['e'])
                        
                        i += len(group) # Skip over the processed group
                    else:
                        # No common prefix, just add the rule
                        new_grammar[nt].append(current_prod)
                        i += 1
            
            # Add new non-terminals and their rules to the main grammar
            if changed:
                new_grammar.update(nt_to_add)
                self.grammar = new_grammar
                self.non_terminals.update(nt_to_add.keys())
                self._derive_terminals()

    def _compute_nullable(self):
        """Computes the set of nullable non-terminals."""
        self.nullable = set()
        changed = True
        while changed:
            changed = False
            for nt, prods in self.grammar.items():
                if nt in self.nullable:
                    continue
                for prod in prods:
                    if prod == ['e']:
                        self.nullable.add(nt)
                        changed = True
                        break
                    
                    all_symbols_nullable = True
                    for symbol in prod:
                        if symbol not in self.nullable:
                            all_symbols_nullable = False
                            break
                    
                    if all_symbols_nullable:
                        self.nullable.add(nt)
                        changed = True
                        break

    def _compute_first(self):
        """Computes the FIRST sets for all symbols."""
        # Initialize FIRST sets
        for t in self.terminals:
            self.first[t] = {t}
        for nt in self.non_terminals:
            self.first[nt] = set()
        self.first['e'] = {'e'}
        
        changed = True
        while changed:
            changed = False
            for nt, prods in self.grammar.items():
                for prod in prods:
                    
                    # --- BUG FIX START ---
                    # Handle A -> e production directly
                    if prod == ['e']:
                        if 'e' not in self.first[nt]:
                            self.first[nt].add('e')
                            changed = True
                        continue # Move to the next production
                    # --- BUG FIX END ---
                    
                    # prod = [Y1, Y2, ..., Yk]
                    rhs_first = set()
                    all_nullable = True
                    
                    for symbol in prod:
                        f_symbol = self.first[symbol]
                        rhs_first.update(f_symbol - {'e'})
                        
                        if symbol not in self.nullable:
                            all_nullable = False
                            break
                    
                    if all_nullable:
                        rhs_first.add('e')
                        
                    if not rhs_first.issubset(self.first[nt]):
                        self.first[nt].update(rhs_first)
                        changed = True

    def _compute_first_of_sequence(self, sequence):
        """Helper to compute FIRST set for a sequence of symbols."""
        result = set()
        all_nullable = True
        
        for symbol in sequence:
            f_symbol = self.first[symbol]
            result.update(f_symbol - {'e'})
            
            # --- BUG FIX START ---
            # The check must correctly identify non-nullable symbols.
            # 1. 'e' is always nullable.
            # 2. Terminals (other than 'e') are never nullable.
            # 3. Non-terminals are nullable only if they are in the nullable set.
            is_symbol_nullable = False
            if symbol == 'e':
                is_symbol_nullable = True
            elif symbol in self.non_terminals and symbol in self.nullable:
                is_symbol_nullable = True
            
            if not is_symbol_nullable:
                all_nullable = False
                break
            # --- BUG FIX END ---
                
        if all_nullable:
            result.add('e')
            
        return result

    def _compute_follow(self):
        """Computes the FOLLOW sets for all non-terminals."""
        for nt in self.non_terminals:
            self.follow[nt] = set()
            
        self.follow[self.start_symbol].add('$')
        
        changed = True
        while changed:
            changed = False
            for nt, prods in self.grammar.items(): # A -> alpha
                for prod in prods:
                    # For each A -> B C D
                    for i in range(len(prod)):
                        symbol = prod[i] # e.g., B
                        if symbol in self.non_terminals:
                            remaining_sequence = prod[i+1:] # e.g., [C, D]
                            
                            # Rule 2: A -> ... B beta
                            if remaining_sequence:
                                first_of_rest = self._compute_first_of_sequence(remaining_sequence)
                                
                                # Add FIRST(beta) - {e} to FOLLOW(B)
                                added = first_of_rest - {'e'}
                                if not added.issubset(self.follow[symbol]):
                                    self.follow[symbol].update(added)
                                    changed = True
                                    
                                # Rule 3: If beta is nullable, add FOLLOW(A) to FOLLOW(B)
                                if 'e' in first_of_rest:
                                    if not self.follow[nt].issubset(self.follow[symbol]):
                                        self.follow[symbol].update(self.follow[nt])
                                        changed = True
                            
                            # Rule 3: A -> ... B (B is last symbol)
                            else:
                                if not self.follow[nt].issubset(self.follow[symbol]):
                                    self.follow[symbol].update(self.follow[nt])
                                    changed = True

    def _build_parsing_table(self):
        """Builds the LL(1) predictive parsing table."""
        all_terminals = self.terminals.union({'$'})
        self.table = {nt: {t: None for t in all_terminals} for nt in self.non_terminals}
        
        for nt, prods in self.grammar.items(): # A -> ...
            for prod in prods: # A -> alpha
                first_alpha = self._compute_first_of_sequence(prod)
                
                # Rule 1: For each 'a' in FIRST(alpha), add A -> alpha to M[A, a]
                for terminal in (first_alpha - {'e'}):
                    if self.table[nt][terminal] is not None:
                        self.is_ll1 = False
                        self.conflicts.append(
                            f"FIRST/FIRST conflict at M[{nt}, {terminal}]: "
                            f"Existing='{nt} -> {' '.join(self.table[nt][terminal])}', "
                            f"New='{nt} -> {' '.join(prod)}'"
                        )
                    else:
                        self.table[nt][terminal] = prod
                        
                # Rule 2: If 'e' in FIRST(alpha), for each 'b' in FOLLOW(A), add A -> alpha to M[A, b]
                if 'e' in first_alpha:
                    for terminal in self.follow[nt]:
                        if self.table[nt][terminal] is not None:
                            self.is_ll1 = False
                            self.conflicts.append(
                                f"FIRST/FOLLOW conflict at M[{nt}, {terminal}]: "
                                f"Existing='{nt} -> {' '.join(self.table[nt][terminal])}', "
                                f"New='{nt} -> {' '.join(prod)}' (from FOLLOW set)"
                            )
                        else:
                            self.table[nt][terminal] = prod

    def parse(self, input_string):
        """Parses an input string using the generated table."""
        if not self.is_ll1:
            print("\nCannot parse: Grammar is not LL(1).")
            return

        print(f"\n--- 7. Parsing Input String: '{input_string}' ---")
        
        tokens = input_string.strip().split() + ['$']
        stack = ['$', self.start_symbol]
        
        # For building the parse tree
        root = Node(self.start_symbol)
        node_stack = [root] # Stack of nodes to be expanded
        
        trace = []
        step = 1
        input_ptr = 0
        
        while stack:
            stack_str = " ".join(reversed(stack))
            input_str = " ".join(tokens[input_ptr:])
            
            stack_top = stack[-1]
            current_token = tokens[input_ptr]
            
            action = ""
            
            if stack_top == current_token == '$':
                action = "Accept"
                trace.append([step, stack_str, input_str, action])
                self._print_trace(trace)
                print("\nString Accepted.")
                print("\n--- 8. Generating Parse Tree ---")
                self._print_tree_recursive(root)
                return True
                
            elif stack_top in self.terminals:
                if stack_top == current_token:
                    action = f"Match {current_token}"
                    stack.pop()
                    input_ptr += 1
                    node_stack.pop() # This terminal node is matched
                else:
                    action = f"Error: Mismatch (Stack: {stack_top}, Input: {current_token})"
                    trace.append([step, stack_str, input_str, action])
                    self._print_trace(trace)
                    print(f"\nString Rejected: Mismatch.")
                    return False
                    
            elif stack_top in self.non_terminals:
                production = self.table[stack_top].get(current_token)
                
                if production is None:
                    action = f"Error: No rule in M[{stack_top}, {current_token}]"
                    trace.append([step, stack_str, input_str, action])
                    self._print_trace(trace)
                    print(f"\nString Rejected: No parsing table entry.")
                    return False
                else:
                    action = f"Apply {stack_top} -> {' '.join(production) or 'e'}"
                    stack.pop()
                    
                    parent_node = node_stack.pop()
                    
                    if production == ['e']:
                        parent_node.children = [Node('e')]
                    else:
                        # Add children to tree and stack
                        child_nodes = [Node(symbol) for symbol in production]
                        parent_node.children = child_nodes
                        
                        # Push children to stacks in reverse order
                        for symbol in reversed(production):
                            stack.append(symbol)
                        
                        # Only push expandable/matchable nodes to the node_stack
                        for child in reversed(child_nodes):
                             node_stack.append(child)

            else: # Should not happen
                action = f"Error: Unknown symbol on stack {stack_top}"
                trace.append([step, stack_str, input_str, action])
                self._print_trace(trace)
                print(f"\nString Rejected: Internal Error.")
                return False

            trace.append([step, stack_str, input_str, action])
            step += 1
            
        # Should be caught by the '$' == '$' check
        self._print_trace(trace)
        print("\nString Rejected: Stack emptied unexpectedly.")
        return False

    # --- Printing Helper Methods ---
    
    def _print_grammar(self, grammar, title):
        """Prints the grammar in a readable format."""
        print(f"\n--- {title} ---")
        for nt, prods in grammar.items():
            prod_str = " | ".join([" ".join(p) for p in prods])
            print(f"{nt: <10} -> {prod_str}")

    def _print_sets(self):
        """Prints NULLABLE, FIRST, and FOLLOW sets."""
        print(f"NULLABLE: {self.nullable or '{}'}\n")
        
        print("FIRST Sets:")
        for nt in self.non_terminals:
            print(f"  FIRST({nt:<5}) = {self.first[nt]}")
            
        print("\nFOLLOW Sets:")
        for nt in self.non_terminals:
            print(f"  FOLLOW({nt:<5}) = {self.follow[nt]}")

    def _print_table(self):
        """Prints the parsing table using tabulate."""
        terminals_list = sorted(list(self.terminals)) + ['$']
        headers = ["Non-Terminal"] + terminals_list
        table_data = []

        for nt in sorted(list(self.non_terminals)):
            row = [nt]
            for t in terminals_list:
                prod = self.table[nt].get(t)
                if prod is None:
                    row.append(" ") # Error/empty entry
                elif prod == ['e']:
                    row.append(f"{nt} -> e")
                else:
                    row.append(f"{nt} -> {' '.join(prod)}")
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def _print_trace(self, trace):
        """Prints the parsing trace table."""
        headers = ["Step", "Stack", "Input String", "Action"]
        print(tabulate(trace, headers=headers, tablefmt="grid"))

    def _print_tree_recursive(self, node, prefix="", is_last=True):
        """Recursively prints the parse tree with connecting lines."""
        print(prefix + ("└── " if is_last else "├── ") + node.value)
        child_prefix = prefix + ("    " if is_last else "│   ")
        
        for i, child in enumerate(node.children):
            self._print_tree_recursive(child, child_prefix, i == len(node.children) - 1)


# --- Main execution ---
def main():
    if len(sys.argv) != 2:
        print("No grammer file provided.")
        print("\nExample usage: python code.py path/to/grammer/file")
        
        example_grammer = "\nnexample of grammer in a file: \n\n"
        example_grammer += "S -> A B\n"
        example_grammer += "A -> a | e\n"
        example_grammer += "B -> b\n\n"
        example_grammer += "rules: 1. you must use '->' to separate the non-terminal on the left from its productions on the right.\n"
        example_grammer += "       2. You must put a space ' ' between every symbol in a production.\n"
        example_grammer += "       3. You must use a single lowercase 'e' to represent epsilon(ϵ).\n"
        example_grammer += "       4. You must use the '|' symbol to separate multiple productions for the same non-terminal.\n"
        example_grammer += "       5. The non-terminal on the first valid grammar line in the file is automatically set as the start symbol.\n"
        example_grammer += "       6. You can add comments to your file by starting a line with the '#' symbol. The parser will ignore these lines."
        
        print(example_grammer)
        return
    
    # 1. Initialize the parser
    # This will automatically run all steps 1-6    
    parser = LL1Parser(sys.argv[1])

    # 2. Parse an input string
    if parser.is_ll1:
        try:
            input_str = input("\nEnter an input string to parse: ")
            parser.parse(input_str)

        except EOFError:
            print("\nNo input provided. Exiting.")
            
    else:
        print("\nCannot proceed with parsing as the grammar is not LL(1).")

if __name__ == "__main__":
    main()