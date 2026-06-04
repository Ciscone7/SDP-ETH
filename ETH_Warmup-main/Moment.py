from itertools import product
import re
from collections import defaultdict
import numpy as np
from scipy.constants import alpha
from scipy.sparse import kron, identity, csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
import time
from qutip import *
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.sparse import csc_matrix
import csv
import sys

id2 = np.array([[1, 0], [0, 1]], dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)


class Monomial: 

    op: str
    site: list[tuple[int, int, int]]
    phase: complex

    def __init__(self, op: str, site: tuple[int, int, int], phase: complex = 1.0):
        self.op = op
        if isinstance(site, tuple):
            self.site = [site]
        else:
            self.site = site
            
        self.phase = phase

    def __repr__(self):
        return f"({self.op}, {self.site}, {self.phase})"

    def __eq__(self, other):
        return self.op == other.op and self.site == other.site

    def __hash__(self):
        return hash((self.op, tuple(self.site)))

    def __lt__(self, other):
        # Sorting rules (normal form): identity first, then length, then op string, then sites
        def get_priority(m):
            if m.op == 'I':
                return (0, "", ())
            return (len(m.op), m.op, tuple(m.site))
        
        return get_priority(self) < get_priority(other)

    def dagger(self):
        return Monomial(self.op, self.site, self.phase.conjugate())

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Forces Python to save to disk immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def spin_multiplication(a: Monomial, b: Monomial) -> Monomial:
    # Pauli Algebra Rules
    single_rules = {
        ('I', 'I'): ('I', 1), ('I', 'X'): ('X', 1), ('I', 'Y'): ('Y', 1), ('I', 'Z'): ('Z', 1),
        ('X', 'I'): ('X', 1), ('X', 'X'): ('I', 1), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
        ('Y', 'I'): ('Y', 1), ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1), ('Y', 'Z'): ('X', 1j),
        ('Z', 'I'): ('Z', 1), ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1),
    }

    stream = []
    

    if a.op == 'I': 
        pass 
    elif len(a.op) != len(a.site):
         for char, s in zip(a.op, a.site):
             stream.append((s, char))
    else:
        for char, s in zip(a.op, a.site):
            stream.append((s, char))
            
    if b.op == 'I':
        pass
    else:
        for char, s in zip(b.op, b.site):
            stream.append((s, char))


    current_phase = a.phase * b.phase
    site_map = {} 

    for s, op_char in stream:
        if s in site_map:
            prev_op = site_map[s]
            new_op, rule_phase = single_rules[(prev_op, op_char)]
            
            site_map[s] = new_op
            current_phase *= rule_phase
        else:
            site_map[s] = op_char


    final_pairs = []
    for s, op in site_map.items():
        if op != 'I':
            final_pairs.append((s, op))
    
    final_pairs.sort(key=lambda x: x[0])

    if not final_pairs:

        ref_site = a.site[0] 
        return Monomial("I", [ref_site], current_phase)
    
    new_sites = [p[0] for p in final_pairs]
    new_op_str = "".join([p[1] for p in final_pairs])
    
    return Monomial(new_op_str, new_sites, current_phase)



def shift_monomial(m: Monomial, d: int, N: int) -> Monomial:
    #Shifts a monomial by distance d on a periodic lattice of size N

    if m.op == 'I':
        return Monomial('I', (1, 0, 0), m.phase)
        
    new_pairs = []
    for op_char, s in zip(m.op, m.site):
        new_site_idx = ((s[0] - 1 + d) % N) + 1
        new_site = (new_site_idx, s[1], s[2])
        new_pairs.append((new_site, op_char))
        
    new_pairs.sort(key=lambda x: x[0][0])
    
    new_sites = [p[0] for p in new_pairs]
    new_op_str = "".join([p[1] for p in new_pairs])
    
    return Monomial(new_op_str, new_sites, m.phase)


def get_canonical_ti(m: Monomial, N: int) -> Monomial:
    if m.op == 'I' or not m.site:
        return Monomial('I', (1, 0, 0), m.phase)

    shifts_to_test = [((1 - s[0]) % N) for s in m.site]
    candidates = [shift_monomial(m, d, N) for d in shifts_to_test]


    def canonical_key(cand):

        span = cand.site[-1][0] - cand.site[0][0] if cand.site else 0
        return (span, cand.op, tuple(cand.site))

    return min(candidates, key=canonical_key)


def get_circulant_eigenvector(N, k):
    j = np.arange(N)
    eigenvector = np.exp(-2j * np.pi * k * j / N)

    return eigenvector #/ np.sqrt(N)

def get_circulant_eigenvalue(c, omega):
    eigenvalue_terms = []
    
    for monomial, scalar in zip(c, omega):
        new_phase = monomial.phase * scalar
        new_monomial = Monomial(monomial.op, monomial.site, new_phase)
        eigenvalue_terms.append(new_monomial)
        
    return eigenvalue_terms

def get_circulant_vectors(symbolic_matrix, N):
    size = len(symbolic_matrix)
    C = []
    
    for r in range(1, size, N):
        for c in range(1, size, N):
            aux = [symbolic_matrix[r + k][c] for k in range(N)]
            C.append(aux)
            
    return C

def print_circulant_vectors(vectors: list[list[Monomial]]):

    def format_monomial(m: Monomial) -> str:
        if m is None:
            return "0"
            
        phase_str = ""
        if m.phase == 1.0:
            phase_str = ""
        elif m.phase == -1.0:
            phase_str = "-"
        elif m.phase == 1j:
            phase_str = "i"
        elif m.phase == -1j:
            phase_str = "-i"
        else:
            phase_str = f"({m.phase:g})"

        if m.op == 'I':
            op_str = f"I{m.site[0][0]}" if m.site else "I"
        else:
            op_str = "".join([f"{char}{s[0]}" for char, s in zip(m.op, m.site)])

        return f"{phase_str}{op_str}"

    for idx, vec in enumerate(vectors):
        formatted_vec = [format_monomial(m) for m in vec]
        print(f"Block Column {idx}: [ " + "  ".join(formatted_vec) + " ]")


def NPA(n: int, N: int, input_list: list[Monomial] = None, use_full_basis: bool = False):

    if n <= 0:
        return []

    alphabet = []

    if use_full_basis:
        
        for i in range(1, N + 1):
            site = (i, 0, 0)
            alphabet.append(Monomial("I", site, 1.0))
            alphabet.append(Monomial("X", site, 1.0))
            alphabet.append(Monomial("Y", site, 1.0))
            alphabet.append(Monomial("Z", site, 1.0))
    else:
        if input_list is None:
            raise ValueError("input_list cannot be None when use_full_basis is False")
        alphabet = input_list

    
    combos = product(alphabet, repeat=n)
    
   
    seen_keys = set()
    result_monomials = []

    for combo_tuple in combos:
        current_product = combo_tuple[0]
        for i in range(1, len(combo_tuple)):
            current_product = spin_multiplication(current_product, combo_tuple[i])
            
        if current_product.op == 'I':
            clean_monomial = Monomial("I", (1,0,0), 1.0) 
        else:
            clean_monomial = Monomial(current_product.op, current_product.site, 1.0)

        if clean_monomial not in seen_keys:
            seen_keys.add(clean_monomial)
            result_monomials.append(clean_monomial)

    return sorted(result_monomials)



def sort_basis(monomial_list: list[Monomial], N: int) -> list[Monomial]:
    has_identity = False
    unique_roots = set()

    for m in monomial_list:
        if m.op == 'I':
            has_identity = True
        else:
            canon_m = get_canonical_ti(m, N)
            clean_root = Monomial(canon_m.op, canon_m.site, 1.0)
            unique_roots.add(clean_root)

    sorted_roots = sorted(list(unique_roots))

    final_basis = []
    if has_identity:
        final_basis.append(Monomial('I', (1, 0, 0), 1.0))

    for root in sorted_roots:
        for d in range(N):
            shifted_m = shift_monomial(root, d, N)
            final_basis.append(shifted_m)

    return final_basis

def generate_neighbour_monomials(initial_list, m = 1, l = 1, pbc=False):
    ops_by_site = defaultdict(list)
    max_site = 0
    
    for mon in initial_list:
        if mon.op != 'I' and len(mon.site) > 0:
            site_idx = mon.site[0][0]
            ops_by_site[site_idx].append(mon)
            if site_idx > max_site:
                max_site = site_idx

    N = max_site
    result_monomials = []
    seen_keys = set()

    for i in range(1, N + 1):
        seq = []
        valid = True
        
        for j in range(m + 1):
            site_idx = i + j * l
            if site_idx > N:
                if pbc:
                    site_idx = ((site_idx - 1) % N) + 1
                else:
                    valid = False
                    break
            seq.append(site_idx)

        if not valid:
            continue

        site_ops = [ops_by_site[s] for s in seq]

        for combo in product(*site_ops):
            current_product = combo[0]
            for k in range(1, len(combo)):
                current_product = spin_multiplication(current_product, combo[k])

            clean_monomial = Monomial(current_product.op, current_product.site, 1.0)

            if clean_monomial not in seen_keys:
                seen_keys.add(clean_monomial)
                result_monomials.append(clean_monomial)

    return sorted(result_monomials)

def print_symbolic_matrix(matrix: list[list[Monomial]]):
    def format_monomial(m: Monomial) -> str:
        if m is None:
            return "0"
            
        phase_str = ""
        if m.phase == 1.0:
            phase_str = ""
        elif m.phase == -1.0:
            phase_str = "-"
        elif m.phase == 1j:
            phase_str = "i"
        elif m.phase == -1j:
            phase_str = "-i"
        else:
            phase_str = f"({m.phase:g})"

        if m.op == 'I':
            op_str = f"I{m.site[0][0]}" if m.site else "I"
        else:
            op_str = "".join([f"{char}{s[0]}" for char, s in zip(m.op, m.site)])

        return f"{phase_str}{op_str}"

    str_matrix = []
    for row in matrix:
        str_matrix.append([format_monomial(m) for m in row])
        
    if not str_matrix:
        return

    num_cols = len(str_matrix[0])
    col_widths = [0] * num_cols
    
    for row in str_matrix:
        for j, s in enumerate(row):
            if len(s) > col_widths[j]:
                col_widths[j] = len(s)
                
    for row in str_matrix:
        formatted_row = "  ".join([f"{s:<{col_widths[j]}}" for j, s in enumerate(row)])
        print(formatted_row)


def parse_sdp_logs(filepath):
    exp_values = []
    lower_bounds = []
    upper_bounds = []
    bound_width = []

    try:
        with open(filepath, 'r', encoding='utf-16') as file:
            lines = file.readlines()
    except UnicodeError:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

    for line in lines:
        clean_line = line.strip()

        if clean_line.startswith("Min bound moment matrix:"):
            value_str = clean_line.split("Min bound moment matrix:")[1].strip()
            lower_bounds.append(float(value_str))

        elif clean_line.startswith("Max bound moment matrix:"):
            value_str = clean_line.split("Max bound moment matrix:")[1].strip()
            upper_bounds.append(float(value_str))

        elif clean_line.startswith("True value:"):
            value_str = clean_line.split("True value:")[1].strip()
            exp_values.append(float(value_str))

        elif clean_line.startswith("Moment matrix bound width:"):
            value_str = clean_line.split("Moment matrix bound width:")[1].strip()
            bound_width.append(float(value_str))

    return exp_values, lower_bounds, upper_bounds, bound_width

def parse_bounds_logs(filepath):
    lower_bounds = []
    upper_bounds = []

    try:
        with open(filepath, 'r', encoding='utf-16') as file:
            lines = file.readlines()
    except UnicodeError:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()

    for line in lines:
        clean_line = line.strip()

        if clean_line.startswith("GS lower bound:"):
            value_str = clean_line.split("GS lower bound:")[1].strip()
            lower_bounds.append(float(value_str))

        elif clean_line.startswith("GS upper bound:"):
            value_str = clean_line.split("GS upper bound:")[1].strip()
            upper_bounds.append(float(value_str))

    return lower_bounds, upper_bounds

def export_symbolic_matrix_to_csv(matrix: list[list[Monomial]], filename: str = "symbolic_matrix.csv"):
    
    def format_monomial(m: Monomial) -> str:
        if m is None:
            return "0"
            
        phase_str = ""
        if m.phase == 1.0:
            phase_str = ""
        elif m.phase == -1.0:
            phase_str = "-"
        elif m.phase == 1j:
            phase_str = "i"
        elif m.phase == -1j:
            phase_str = "-i"
        else:
            phase_str = f"({m.phase:g})"

        if m.op == 'I':
            op_str = f"I{m.site[0][0]}" if m.site else "I"
        else:
            op_str = "".join([f"{char}{s[0]}" for char, s in zip(m.op, m.site)])

        final_str = f"{phase_str}{op_str}"
        
        if final_str.startswith('-'):
            return f"'{final_str}"
            
        return final_str

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix:
            formatted_row = [format_monomial(m) for m in row]
            writer.writerow(formatted_row)
            
    print(f"Matrix successfully exported to {filename}")

def is_block_circulant(matrix: list[list[Monomial]], row_i: int, row_f: int, col_i: int, col_f: int) -> bool:
    # Checks if a sub-block of the symbolic matrix is strictly right-circulant(the column vector shifts downward).
    # row_f and col_f are EXCLUSIVE
    
    nrows = row_f - row_i
    ncols = col_f - col_i
    
    
    if nrows != ncols:
        print(f"Fail: Block is not square ({nrows}x{ncols}).")
        return False
        
    N_block = nrows
    if N_block <= 1:
        return True 
        
    
    for r in range(N_block):
        for c in range(N_block):
            
            
            ref_c = (c - r) % N_block
            
            m_current = matrix[row_i + r][col_i + c]
            m_expected = matrix[row_i + 0][col_i + ref_c]
            
            
            op_match = (m_current.op == m_expected.op)
            site_match = (m_current.site == m_expected.site)
            phase_match = (m_current.phase == m_expected.phase)
            
            if not (op_match and site_match and phase_match):
                print(f"Fail: Broken circulant structure at matrix index ({row_i + r}, {col_i + c})")
                print(f"Expected: ({m_expected.op}, {m_expected.site}, {m_expected.phase})")
                print(f"Found:    ({m_current.op}, {m_current.site}, {m_current.phase})")
                return False
                
    return True

def create_moment_matrix(basis_list: list[Monomial]):
    
    size = len(basis_list)
    
    symbolic_matrix = [[None for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        symbolic_matrix[0][i] = basis_list[i]
        symbolic_matrix[i][0] = basis_list[i]

    for i in range(1, size):
        for j in range(1, size):
            symbolic_matrix[i][j] = spin_multiplication(symbolic_matrix[i][0].dagger(), symbolic_matrix[0][j])

    groups = defaultdict(list)

    for i in range(size):
        for j in range(size):
            monomial = symbolic_matrix[i][j]
            groups[monomial].append((i, j, monomial.phase))

    N = size
    M_expr = 0
    var_dict = {}

    for key, locations in groups.items():
        
        if key.op == 'I':
            variable = 1.0
        else:
            
            if key in var_dict:
                variable = var_dict[key]
            else:
                
                variable = cp.Variable(name=f"<{key.op}{key.site}>")
                var_dict[key] = variable

        
        rows = [loc[0] for loc in locations]
        cols = [loc[1] for loc in locations]
        data = [loc[2] for loc in locations] 

        
        C_op = coo_matrix((data, (rows, cols)), shape=(N, N))
        
        
        M_expr += variable * C_op

    return M_expr, var_dict, symbolic_matrix

def create_block_diagonal_moment_matrix(basis_list: list[Monomial], lattice_size: int):
    size = len(basis_list)
    
    symbolic_matrix = [[None for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        symbolic_matrix[0][i] = basis_list[i]
        symbolic_matrix[i][0] = basis_list[i]

    for i in range(1, size):
        for j in range(1, size):
            raw_product = spin_multiplication(symbolic_matrix[i][0].dagger(), symbolic_matrix[0][j])
            symbolic_matrix[i][j] = get_canonical_ti(raw_product, lattice_size)

    C = get_circulant_vectors(symbolic_matrix, lattice_size)
    
    N_B = (size - 1) // lattice_size
    N_dim = N_B + 1
    
    unique_keys = set()
    for r in range(size):
        for c in range(size):
            m = symbolic_matrix[r][c]
            canonical_m = get_canonical_ti(m, lattice_size)
            if canonical_m.op != 'I':
                unique_keys.add(Monomial(canonical_m.op, canonical_m.site, 1.0))
                
    non_id_keys = list(unique_keys)
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}
    
    var_dict = {}
    expr_list = []
    for key in non_id_keys:
        var = cp.Variable(name=f"<{key.op}{key.site}>")
        var_dict[key] = var
        expr_list.append(var)
        
    X = cp.hstack(expr_list) if expr_list else None
    
    M_expr_list = []
    
    for k in range(lattice_size):
        omega = get_circulant_eigenvector(lattice_size, k)
        
        A_rows, A_cols, A_data = [], [], []
        b = np.zeros(N_dim * N_dim, dtype=complex)
        
        for r in range(N_dim):
            for c in range(N_dim):
                flat_idx = c * N_dim + r
                terms = []
                
                if r == 0 and c == 0:
                    terms = [Monomial('I', (1, 0, 0), 1.0)]
                elif r == 0 and c > 0:
                    if k == 0:
                        term = symbolic_matrix[0][1 + (c - 1) * lattice_size]
                        terms = [Monomial(term.op, term.site, term.phase )] #* np.sqrt(lattice_size)
                elif c == 0 and r > 0:
                    if k == 0:
                        term = symbolic_matrix[1 + (r - 1) * lattice_size][0]
                        terms = [Monomial(term.op, term.site, term.phase )] #* np.sqrt(lattice_size)
                else:
                    block_idx = (r - 1) * N_B + (c - 1)
                    c_vector = C[block_idx]
                    terms = get_circulant_eigenvalue(c_vector, omega)
                    
                for term in terms:
                    canonical_term = get_canonical_ti(term, lattice_size)
                    if canonical_term.op == 'I':
                        b[flat_idx] += canonical_term.phase
                    else:
                        clean_key = Monomial(canonical_term.op, canonical_term.site, 1.0)
                        if clean_key in key_to_col:
                            A_rows.append(flat_idx)
                            A_cols.append(key_to_col[clean_key])
                            A_data.append(canonical_term.phase)
                            
        if X is not None:
            A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N_dim * N_dim, len(non_id_keys)))
            flat_M = A @ X + b
        else:
            flat_M = b
            
        Mk_expr = cp.reshape(flat_M, (N_dim, N_dim), order='F')
        M_expr_list.append(Mk_expr)
        
    return M_expr_list, var_dict, symbolic_matrix


def create_block_diagonal_localizing_matrix(basis_list: list[Monomial], symbolic_H, var_dict, N: int):
    size = len(basis_list)
    N_B = (size - 1) // N
    N_dim = N_B + 1


    symbolic_matrix_H = [[[] for _ in range(size)] for _ in range(size)]
    groups = defaultdict(list)

    for i in range(size):
        Li_dag = basis_list[i].dagger()
        for j in range(size):
            Lj = basis_list[j]
            for coeff, h_term in symbolic_H:

                product = spin_multiplication(Li_dag, spin_multiplication(Lj, h_term))
                canonical_product = get_canonical_ti(product, N)

                total_phase = canonical_product.phase * coeff
                clean_canonical = Monomial(canonical_product.op, canonical_product.site, 1.0)

                symbolic_matrix_H[i][j].append((total_phase, clean_canonical))
                groups[clean_canonical].append((i, j, total_phase))

    non_id_keys = [k for k in groups.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    expr_list = []
    for key in non_id_keys:
        if key in var_dict:
            var = var_dict[key]
        else:
            var = cp.Variable(name=f"<{key.op}{key.site}>")
            var_dict[key] = var
        expr_list.append(var)


    A_rows_full, A_cols_full, A_data_full = [], [], []
    b_full = np.zeros(size * size, dtype=complex)

    for key, locations in groups.items():
        if key.op == 'I':
            for r, c, phase in locations:
                flat_idx = c * size + r
                b_full[flat_idx] += phase
        else:
            col_idx = key_to_col[key]
            for r, c, phase in locations:
                flat_idx = c * size + r
                A_rows_full.append(flat_idx)
                A_cols_full.append(col_idx)
                A_data_full.append(phase)

    if len(expr_list) > 0:
        X = cp.hstack(expr_list)
        A_full = csc_matrix((A_data_full, (A_rows_full, A_cols_full)), shape=(size * size, len(non_id_keys)))
        flat_M_full = A_full @ X + b_full
    else:
        flat_M_full = b_full


    M_H_expr_list = []

    for k in range(N):
        omega = get_circulant_eigenvector(N, k)

        A_rows_k, A_cols_k, A_data_k = [], [], []
        b_k = np.zeros(N_dim * N_dim, dtype=complex)

        for r in range(N_dim):
            for c in range(N_dim):
                flat_idx = c * N_dim + r
                block_element_terms = defaultdict(complex)

                if r == 0 and c == 0:
                    for phase, clean_mon in symbolic_matrix_H[0][0]:
                        block_element_terms[clean_mon] += phase
                elif r == 0 and c > 0:
                    if k == 0:
                        for phase, clean_mon in symbolic_matrix_H[0][1 + (c - 1) * N]:
                            block_element_terms[clean_mon] += phase
                elif c == 0 and r > 0:
                    if k == 0:
                        for phase, clean_mon in symbolic_matrix_H[1 + (r - 1) * N][0]:
                            block_element_terms[clean_mon] += phase
                else:
                    # FIX: Correctly iterate down the row index to extract the column vector
                    col_idx = 1 + (c - 1) * N
                    for d in range(N):
                        row_idx = 1 + (r - 1) * N + d
                        phase_omega = omega[d]
                        for phase, clean_mon in symbolic_matrix_H[row_idx][col_idx]:
                            block_element_terms[clean_mon] += phase * phase_omega

                for clean_mon, total_phase in block_element_terms.items():
                    if abs(total_phase) > 1e-12:
                        if clean_mon.op == 'I':
                            b_k[flat_idx] += total_phase
                        else:
                            if clean_mon in key_to_col:
                                A_rows_k.append(flat_idx)
                                A_cols_k.append(key_to_col[clean_mon])
                                A_data_k.append(total_phase)

        if len(expr_list) > 0:
            A_k = csc_matrix((A_data_k, (A_rows_k, A_cols_k)), shape=(N_dim * N_dim, len(non_id_keys)))
            flat_M_k = A_k @ X + b_k
        else:
            flat_M_k = b_k

        Mk_expr = cp.reshape(flat_M_k, (N_dim, N_dim), order='F')
        M_H_expr_list.append(Mk_expr)

    return M_H_expr_list, var_dict, None, flat_M_full

def kron_n(ops):
    
    result = ops[0]
    for op in ops[1:]:
        result = kron(result, op, format='csr')
    return result



def get_symbolic_hamiltonian(N, J, g, h):
    terms = []

    for i in range(1, N):  
        m = Monomial("ZZ", [(i, 0, 0), (i+1, 0, 0)], 1.0)
        terms.append((-J, m)) 
    
    m = Monomial("ZZ", [(1, 0, 0), (N, 0, 0)], 1.0)
    terms.append((-J, m))

    for i in range(1, N + 1):
        m = Monomial("X", [(i, 0, 0)], 1.0)
        terms.append((-g, m))

    for i in range(1, N + 1):
        m = Monomial("Z", [(i, 0, 0)], 1.0)
        terms.append((-h, m))
        
    return terms



def compute_commutator(operators, hamiltonian):
    is_single = False
    if not isinstance(operators, list):
        operators = [operators]
        is_single = True

    all_results = []

    for operator in operators:
        terms_map = {}

        for h_coeff, h_term in hamiltonian:
            prod_AH = spin_multiplication(operator, h_term)
            key_AH = (prod_AH.op, tuple(prod_AH.site))
            val_AH = h_coeff * prod_AH.phase
            terms_map[key_AH] = terms_map.get(key_AH, 0j) + val_AH

            prod_HA = spin_multiplication(h_term, operator)
            key_HA = (prod_HA.op, tuple(prod_HA.site))
            val_HA = h_coeff * prod_HA.phase
            terms_map[key_HA] = terms_map.get(key_HA, 0j) - val_HA

        single_op_result = []
        for (op, site_tuple), total_coeff in terms_map.items():
            if abs(total_coeff) > 1e-9:
                m = Monomial(op, list(site_tuple), total_coeff)
                single_op_result.append(m)
        
        all_results.append(single_op_result)

    if is_single:
        return all_results[0]
        
    return all_results

def create_localizing_matrix(basis_list: list[Monomial], symbolic_H, var_dict):

    size = len(basis_list)

    groups = defaultdict(list)

    for i in range(size):
        Li_dag = basis_list[i].dagger()

        for j in range(size):
            Lj = basis_list[j]

            for coeff, h_term in symbolic_H:

                product = spin_multiplication(
                    Li_dag,
                    spin_multiplication(Lj, h_term)
                )

                product.phase *= coeff

                groups[product].append((i, j, product.phase))


    N = size
    M_expr = 0

    for key, locations in groups.items():

        # if key.op == 'I':
        #     variable = 1.0
        # else:
        if key in var_dict:
            variable = var_dict[key]
        else:
            variable = cp.Variable(name=f"<{key.op}{key.site}>")
            var_dict[key] = variable

        rows = [loc[0] for loc in locations]
        cols = [loc[1] for loc in locations]
        data = [loc[2] for loc in locations]

        C_op = coo_matrix((data, (rows, cols)), shape=(N, N))

        M_expr += variable * C_op

    return M_expr, var_dict, None


def create_objective_function(hamiltonian, var_dict):
    objective_expr = 0
    
    for coeff, term in hamiltonian:
        key = Monomial(term.op, term.site, 1.0)
        
        full_coeff = coeff * term.phase
        
        if key.op == 'I':
            objective_expr += full_coeff
        elif key in var_dict:
            
            objective_expr += full_coeff * var_dict[key]
        else:
            new_v = cp.Variable(name=f"H_term_{key.op}")
            var_dict[key] = new_v
            objective_expr += full_coeff * new_v
            
    return objective_expr

def create_objective_function_TI(hamiltonian, var_dict, N):
    objective_expr = 0
    
    for coeff, term in hamiltonian:
        canonical_m = get_canonical_ti(term, N)
        clean_key = Monomial(canonical_m.op, canonical_m.site, 1.0)
        
        # Combine the original coefficient with the phase from the canonical mapping
        full_coeff = coeff * canonical_m.phase
        
        if clean_key.op == 'I':
            objective_expr += full_coeff
        elif clean_key in var_dict:
            objective_expr += full_coeff * var_dict[clean_key]
        else:
            # new_v = cp.Variable(name=f"Obj_{clean_key.op}{clean_key.site}")
            # var_dict[clean_key] = new_v
            # objective_expr += full_coeff * new_v
            raise ValueError(
                f"Cannot evaluate objective: Operator {clean_key} is not contained within the Moment Matrix.")
            
    return objective_expr

def create_commutator_constraint(operators, hamiltonian, var_dict):

    is_single = False
    if not isinstance(operators, list):
        operators = [operators]
        is_single = True

    expressions = []
    commutators_list = compute_commutator(operators, hamiltonian)

    for comm_terms in commutators_list:
        expr = 0
        
        for term in comm_terms:
            
            key = Monomial(term.op, term.site, 1.0)
            coeff = term.phase 

            if key.op == 'I':
                expr += coeff
            elif key in var_dict:
                expr += coeff * var_dict[key]
            else:
                new_v = cp.Variable(name=f"Comm_{key.op}{key.site}")
                var_dict[key] = new_v
                expr += coeff * new_v
        
        expressions.append(expr)

    if is_single:
        return expressions[0]
        
    return expressions

def create_commutator_constraint_TI(operators, hamiltonian, var_dict, N):
    is_single = False
    if not isinstance(operators, list):
        operators = [operators]
        is_single = True

    expressions = []
    commutators_list = compute_commutator(operators, hamiltonian)

    for comm_terms in commutators_list:
        expr = 0
        ti_grouped_terms = defaultdict(complex)

        for term in comm_terms:
            canonical_m = get_canonical_ti(term, N)
            clean_key = Monomial(canonical_m.op, canonical_m.site, 1.0)
            ti_grouped_terms[clean_key] += canonical_m.phase

        is_valid_constraint = True

        for clean_key, total_coeff in ti_grouped_terms.items():
            if abs(total_coeff) < 1e-9:
                continue

            if clean_key.op == 'I':
                expr += total_coeff
            elif clean_key in var_dict:
                expr += total_coeff * var_dict[clean_key]
            else:
                is_valid_constraint = False
                break

        if is_valid_constraint:
            expressions.append(expr)

    if is_single:
        return expressions[0] if expressions else 0

    return expressions

def random_pm1_hermitian(site):

    v = np.random.randn(3)
    v /= np.linalg.norm(v)
    sym_op = []

    her_op = v[0]*sx + v[1]*sy + v[2]*sz
    sym_op.append((v[0], Monomial("X", site, 1.0))) 
    sym_op.append((v[1], Monomial("Y", site, 1.0))) 
    sym_op.append((v[2], Monomial("Z", site, 1.0))) 
    return her_op, sym_op

def build_hamiltonian(L, J1, J2, h, g, J_xy, delta, gamma):
    H = csr_matrix((2**L, 2**L), dtype=complex)

    # Build -J1 * sum sigma^z_i sigma^z_{i+1}
    for i in range(L-1):
        ops = [id2] * L
        ops[i] = sz
        ops[(i + 1)] = sz
        H -= J1 * kron_n(ops)

    #PBC
    if L > 1:
        ops = [id2] * L
        ops[0] = sz
        ops[L - 1] = sz
        H -= J1 * kron_n(ops)

    # Build -J2 * sum sigma^z_i sigma^z_{i+2}
    for i in range(L-2):
        ops = [id2] * L
        ops[i] = sz
        ops[(i + 2)] = sz
        H -= J2 * kron_n(ops)

    # PBC
    if L > 2:
        ops = [id2] * L
        ops[L - 2] = sz
        ops[0] = sz
        H -= J2 * kron_n(ops)

        ops = [id2] * L
        ops[L - 1] = sz
        ops[1] = sz
        H -= J2 * kron_n(ops)

    # Build -h * sum sigma^x_i
    for i in range(L):
        ops = [id2] * L
        ops[i] = sx
        H -= h * kron_n(ops)


    # Build -g * sum sigma^z_i 
    for i in range(L):
        ops = [id2] * L
        ops[i] = sz
        H -= g * kron_n(ops)

    for i in range(L-1):
        ops = [id2] * L
        ops[i] = sx
        ops[(i + 1)] = sx
        H -= J_xy*(1-(-1)**i*delta)*(1+gamma)/2 * kron_n(ops)

    # PBC
    if L > 1:
        i = L - 1
        ops = [id2] * L
        ops[L - 1] = sx
        ops[0] = sx
        H -= J_xy*(1-(-1)**i*delta)*(1+gamma)/2 * kron_n(ops)

    for i in range(L-1):
        ops = [id2] * L
        ops[i] = sy
        ops[(i + 1)] = sy
        H -= J_xy*(1-(-1)**i*delta)*(1-gamma)/2 * kron_n(ops)

    # PBC
    if L > 1:
        i = L - 1
        ops = [id2] * L
        ops[L - 1] = sy
        ops[0] = sy
        H -= J_xy*(1-(-1)**i*delta)*(1-gamma)/2 * kron_n(ops)

    
    return H

def load_eigen(filename, H):
    if os.path.exists(filename) :
        data = np.load(filename)
        eigenvalues = data['eigenvalues']
        eigenvectors = data['eigenvectors']
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(H.toarray())
        np.savez_compressed(filename, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

    return eigenvalues, eigenvectors

def get_operator(N, sites, ops):
    ops_obj = [id2] * N

    if not isinstance(sites, list):
        sites = [sites]
    if not isinstance(ops, list):
        ops = [ops]
        
    for site, op in zip(sites, ops):
        ops_obj[site] = op
        
    obj_op = kron_n(ops_obj)
    
    return obj_op

def get_expectation(N, sites, ops, eigenvalues, eigenvectors):
    
    expectation_values = []
    obj_op = get_operator(N, sites, ops)
    
    for i in range(len(eigenvalues)):
        psi = eigenvectors[:, i]
        O_psi = obj_op.dot(psi) 
        exp_val = np.vdot(psi, O_psi)
        expectation_values.append(exp_val.real)
        
    return expectation_values

def get_expectation_TI(N, sites, ops, eigenvalues, eigenvectors, tol=1e-8, method='pure'):
    # sum: 1/sqrt(N) * (|psi_1> + |psi_2> + ... |psi_N>)
    #mixed: 1/sqrt(N) * sum |psi_i><psi_i|
    expectation_values = []
    obj_op = get_operator(N, sites, ops)
    
    degenerate_groups = []
    current_group = [0]
    
    for i in range(1, len(eigenvalues)):
        if abs(eigenvalues[i] - eigenvalues[i-1]) < tol:
            current_group.append(i)
        else:
            degenerate_groups.append(current_group)
            current_group = [i]
    degenerate_groups.append(current_group)
    
    for group in degenerate_groups:
        if len(group) == 1:
            psi = eigenvectors[:, group[0]]
            exp_val = np.vdot(psi, obj_op.dot(psi)).real
            expectation_values.append(exp_val)
        else:
            if method == 'pure':
                psi_sum = np.sum(eigenvectors[:, group], axis=1)
                norm = np.linalg.norm(psi_sum)
                
                if norm > 1e-12:
                    psi_sum = psi_sum / norm
                else:
                    psi_sum = eigenvectors[:, group[0]] 
                    
                exp_val = np.vdot(psi_sum, obj_op.dot(psi_sum)).real
            else:
                exp_val_sum = 0
                for idx in group:
                    psi = eigenvectors[:, idx]
                    exp_val_sum += np.vdot(psi, obj_op.dot(psi)).real
                exp_val = exp_val_sum / len(group)
            
            for _ in group:
                expectation_values.append(exp_val)
                
    return expectation_values

def density_matrix_sdp(N, H, energy, eps, sites, ops, flag, details=False):
    X = cp.Variable((2**N, 2**N), hermitian=True)  #

    if hasattr(H, 'toarray'):
        H = H.toarray()


    constraints = []
    constraints += [X >> 0]
    constraints += [cp.trace(X) == 1]
    # constraints += [X@H == (energy)*X]
    constraints += [X@H == H@X]
    constraints += [X@H >> (energy-eps)*X]
    constraints += [(energy+eps)*X >> X@H ]
    
    
    
    obj_op = get_operator(N, sites, ops)

    if hasattr(obj_op, 'toarray'):
        obj_op = obj_op.toarray()
    
    if flag:
        objective=cp.Minimize(cp.real(cp.trace(obj_op@X)))
    else:
        objective=cp.Maximize(cp.real(cp.trace(obj_op@X)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
    
    if details:
        print("problem status = ",prob.status)
    
    return prob

def moment_sdp(N, symbolic_H, basis, symbolic_op, energy, eps, flag, details=False):


    var_dict = {}
    E_target = energy
    E_min = E_target - eps
    E_max = E_target + eps
    
    M_expr, var_dict, _ = create_moment_matrix(basis)

    basis_set = {Monomial(b.op, b.site, 1.0) for b in basis}
    missing_terms = []
    
    for _, h_term in symbolic_H:
        clean_h = Monomial(h_term.op, h_term.site, 1.0)
        
        
        if clean_h not in var_dict and clean_h.op != 'I':
            if clean_h not in basis_set: 
                basis.append(clean_h)
                basis_set.add(clean_h)
                missing_terms.append(clean_h)

    if missing_terms:
        print(f"Forced {len(missing_terms)} missing terms into the basis. {missing_terms}")

    M_H, var_dict, _ = create_localizing_matrix(basis, symbolic_H, var_dict)
    
    
    # print("Computing objective function...")
    objective_expr = create_objective_function(symbolic_op, var_dict)
    

    if details:
        print(f"Imposing Window: [{E_min:.4f}; {E_max:.4f}]")
    

    
    A_ops_list = [op for op in var_dict.keys() if op.op != 'I']
    comm_constraint_expr = create_commutator_constraint(A_ops_list, symbolic_H, var_dict)
    
    
    constraints = [M_expr >> 0]
    constraints.append(M_H - E_min * M_expr >> 0) 
    constraints.append(-M_H + E_max * M_expr >> 0)
    
    if isinstance(comm_constraint_expr, list):
        for expr in comm_constraint_expr:
            constraints.append(expr == 0)
    else:
        constraints.append(comm_constraint_expr == 0)
    
    id_key = None
    for key in var_dict.keys():
        if key.op == 'I':
            id_key = key
            break
    
    if id_key is not None:
        constraints.append(var_dict[id_key] == 1.0)

    if flag: 
        prob = cp.Problem(cp.Minimize(objective_expr), constraints)
    else:
        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    print(f"Total optimization variables: {prob.size_metrics.num_scalar_variables}")
    
    class FailedProblem:
        
        def __init__(self):
            self.value = np.nan
            self.status = "SolverError"

    try:
        
        prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
        
        if details:
            print("problem status = ", prob.status)
            
        return prob

    except cp.error.SolverError:
        print("  [!] MOSEK crashed. Assigning NaN.")
        return FailedProblem()


def create_moment_matrix_vectorized(basis_list: list[Monomial]):
    size = len(basis_list)
    
    symbolic_matrix = [[None for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        symbolic_matrix[0][i] = basis_list[i]
        symbolic_matrix[i][0] = basis_list[i]

    for i in range(1, size):
        for j in range(1, size):
            symbolic_matrix[i][j] = spin_multiplication(symbolic_matrix[i][0].dagger(), symbolic_matrix[0][j])

    groups = defaultdict(list)
    for i in range(size):
        for j in range(size):
            # monomial = symbolic_matrix[i][j]
            clean_monomial = Monomial(symbolic_matrix[i][j].op, symbolic_matrix[i][j].site, 1.0)
            groups[clean_monomial].append((i, j, symbolic_matrix[i][j].phase))

    N = size
    var_dict = {}

    
    non_id_keys = [k for k in groups.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    
    expr_list = []
    for key in non_id_keys:
        var = cp.Variable(name=f"<{key.op}{key.site}>")
        var_dict[key] = var
        expr_list.append(var)

    
    # column-major (F-order) flattening: flat_idx = col * N + row
    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N * N, dtype=complex)

    for key, locations in groups.items():
        if key.op == 'I':
            for r, c, phase in locations:
                flat_idx = c * N + r
                b[flat_idx] += phase
        else:
            col_idx = key_to_col[key]
            for r, c, phase in locations:
                flat_idx = c * N + r
                A_rows.append(flat_idx)
                A_cols.append(col_idx)
                A_data.append(phase)

  
    if len(expr_list) > 0:
        X = cp.hstack(expr_list) 
        
         
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N * N, len(non_id_keys)))
        flat_M = A @ X + b
    else:
        flat_M = b

  
    M_expr = cp.reshape(flat_M, (N, N), order='F')

    return M_expr, var_dict, symbolic_matrix, flat_M

def create_moment_matrix_vectorized_TI(basis_list: list[Monomial], lattice_size: int):
    size = len(basis_list)
    
    symbolic_matrix = [[None for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        symbolic_matrix[0][i] = basis_list[i]
        symbolic_matrix[i][0] = basis_list[i]

    for i in range(1, size):
        for j in range(1, size):
            raw_product = spin_multiplication(symbolic_matrix[i][0].dagger(), symbolic_matrix[0][j])
            symbolic_matrix[i][j] = get_canonical_ti(raw_product, lattice_size) #TI reduction


    for i in range(size):
        symbolic_matrix[0][i] = get_canonical_ti(basis_list[i], lattice_size)
        symbolic_matrix[i][0] = get_canonical_ti(basis_list[i], lattice_size)

    groups = defaultdict(list)
    for i in range(size):
        for j in range(size):
            # monomial = symbolic_matrix[i][j]
            clean_monomial = Monomial(symbolic_matrix[i][j].op, symbolic_matrix[i][j].site, 1.0)
            groups[clean_monomial].append((i, j, symbolic_matrix[i][j].phase))

    N = size
    var_dict = {}

    
    non_id_keys = [k for k in groups.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    
    expr_list = []
    for key in non_id_keys:
        var = cp.Variable(name=f"<{key.op}{key.site}>")
        var_dict[key] = var
        expr_list.append(var)

    
    # column-major (F-order) flattening: flat_idx = col * N + row
    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N * N, dtype=complex)

    for key, locations in groups.items():
        if key.op == 'I':
            for r, c, phase in locations:
                flat_idx = c * N + r
                b[flat_idx] += phase
        else:
            col_idx = key_to_col[key]
            for r, c, phase in locations:
                flat_idx = c * N + r
                A_rows.append(flat_idx)
                A_cols.append(col_idx)
                A_data.append(phase)

  
    if len(expr_list) > 0:
        X = cp.hstack(expr_list) 
        
         
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N * N, len(non_id_keys)))
        flat_M = A @ X + b
    else:
        flat_M = b

  
    M_expr = cp.reshape(flat_M, (N, N), order='F')

    return M_expr, var_dict, symbolic_matrix, flat_M

def create_localizing_matrix_vectorized(basis_list: list[Monomial], symbolic_H, var_dict):
    size = len(basis_list)
    groups = defaultdict(list)

    for i in range(size):
        Li_dag = basis_list[i].dagger()
        for j in range(size):
            Lj = basis_list[j]
            for coeff, h_term in symbolic_H:
                #  Li_dag * Lj * H
                product = spin_multiplication(Li_dag, spin_multiplication(Lj, h_term))
                product.phase *= coeff
                groups[product].append((i, j, product.phase))

    N = size


    non_id_keys = [k for k in groups.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}


    expr_list = []
    for key in non_id_keys:
        if key in var_dict:
            var = var_dict[key]
        else:
            var = cp.Variable(name=f"<{key.op}{key.site}>")
            var_dict[key] = var
        expr_list.append(var)

    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N * N, dtype=complex)

    for key, locations in groups.items():
        if key.op == 'I':
            for r, c, phase in locations:
                flat_idx = c * N + r
                b[flat_idx] += phase
        else:
            col_idx = key_to_col[key]
            for r, c, phase in locations:
                flat_idx = c * N + r
                A_rows.append(flat_idx)
                A_cols.append(col_idx)
                A_data.append(phase)


    if len(expr_list) > 0:
        X = cp.hstack(expr_list)

        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N * N, len(non_id_keys)))
        flat_M = A @ X + b
    else:
        flat_M = b

    M_expr = cp.reshape(flat_M, (N, N), order='F')

    return M_expr, var_dict, None, flat_M

def create_localizing_matrix_vectorized_TI(basis_list: list[Monomial], symbolic_H, var_dict, N: int):
    size = len(basis_list)
    groups = defaultdict(list)

    for i in range(size):
        Li_dag = basis_list[i].dagger()
        for j in range(size):
            Lj = basis_list[j]
            for coeff, h_term in symbolic_H:
                product = spin_multiplication(Li_dag, spin_multiplication(Lj, h_term))

                canonical_product = get_canonical_ti(product, N)

                total_phase = product.phase * coeff
                clean_canonical = Monomial(canonical_product.op, canonical_product.site, 1.0)

                groups[clean_canonical].append((i, j, total_phase))

    N_dim = size
    non_id_keys = [k for k in groups.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    expr_list = []
    for key in non_id_keys:
        if key in var_dict:
            var = var_dict[key]
        else:
            var = cp.Variable(name=f"<{key.op}{key.site}>")
            var_dict[key] = var
        expr_list.append(var)

    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N_dim * N_dim, dtype=complex)

    for key, locations in groups.items():
        if key.op == 'I':
            for r, c, phase in locations:
                flat_idx = c * N_dim + r
                b[flat_idx] += phase
        else:
            col_idx = key_to_col[key]
            for r, c, phase in locations:
                flat_idx = c * N_dim + r
                A_rows.append(flat_idx)
                A_cols.append(col_idx)
                A_data.append(phase)

    if len(expr_list) > 0:
        X = cp.hstack(expr_list)
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N_dim * N_dim, len(non_id_keys)))
        flat_M = A @ X + b
    else:
        flat_M = b

    M_expr = cp.reshape(flat_M, (N_dim, N_dim), order='F')

    return M_expr, var_dict, None, flat_M

def create_localizing_matrix_vectorized_TI_cached(basis_list: list[Monomial], symbolic_H, var_dict, N: int):
    size = len(basis_list)
    groups = defaultdict(list)
    ti_cache = {}
    h_action_cache = {}


    for i in range(size):
        Li_dag = basis_list[i].dagger()
        for j in range(size):
            Lj = basis_list[j]
            base_prod = spin_multiplication(Li_dag, Lj)

            if base_prod not in ti_cache:
                ti_cache[base_prod] = get_canonical_ti(base_prod, N)
            canon_base = ti_cache[base_prod]

            if canon_base not in h_action_cache:
                action_terms = []
                for coeff, h_term in symbolic_H:
                    prod = spin_multiplication(canon_base, h_term)
                    canon_prod = get_canonical_ti(prod, N)
                    action_terms.append(Monomial(canon_prod.op, canon_prod.site, prod.phase * coeff))

                combined_action = defaultdict(complex)
                for m in action_terms:
                    clean_m = Monomial(m.op, m.site, 1.0)
                    combined_action[clean_m] += m.phase
                h_action_cache[canon_base] = combined_action

            phase_shift = base_prod.phase / canon_base.phase
            for clean_m, action_phase in h_action_cache[canon_base].items():
                if abs(action_phase) > 1e-9:
                    groups[clean_m].append((i, j, action_phase * phase_shift))

    N_dim = size
    valid_mask = np.ones(N_dim * N_dim, dtype=bool)


    for key, locations in groups.items():
        if key.op != 'I' and key not in var_dict:
            for r, c, _ in locations:
                flat_idx = c * N_dim + r
                valid_mask[flat_idx] = False


    non_id_keys = [k for k in groups.keys() if k.op != 'I' and k in var_dict]
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    expr_list = [var_dict[key] for key in non_id_keys]

    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N_dim * N_dim, dtype=complex)


    for key, locations in groups.items():
        if key.op == 'I':
            for r, c, phase in locations:
                flat_idx = c * N_dim + r
                b[flat_idx] += phase
        elif key in key_to_col:
            col_idx = key_to_col[key]
            for r, c, phase in locations:
                flat_idx = c * N_dim + r
                A_rows.append(flat_idx)
                A_cols.append(col_idx)
                A_data.append(phase)

    if len(expr_list) > 0:
        X = cp.hstack(expr_list)
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N_dim * N_dim, len(non_id_keys)))
        flat_M = A @ X + b
    else:
        flat_M = b

    M_expr = cp.reshape(flat_M, (N_dim, N_dim), order='F')


    return M_expr, var_dict, None, flat_M, valid_mask



def moment_sdp_flat(N, symbolic_H, basis, symbolic_op, energy, eps, flag, details=False, scalar_flag = True):

    
    var_dict = {}
    E_target = energy
    E_min = E_target - eps
    E_max = E_target + eps
    
    M_expr, var_dict, _, flat_M_expr = create_moment_matrix_vectorized(basis)

    basis_set = {Monomial(b.op, b.site, 1.0) for b in basis}
    missing_terms = []
    
    for _, h_term in symbolic_H:
        clean_h = Monomial(h_term.op, h_term.site, 1.0)
        
        
        if clean_h not in var_dict and clean_h.op != 'I':
            if clean_h not in basis_set: 
                basis.append(clean_h)
                basis_set.add(clean_h)
                missing_terms.append(clean_h)

    if missing_terms:
        print(f"Forced {len(missing_terms)} missing terms into the basis. {missing_terms}")

    M_H, var_dict, _, flat_M_H = create_localizing_matrix_vectorized(basis, symbolic_H, var_dict)
    

    
    # print("Computing objective function...")
    objective_expr = create_objective_function(symbolic_op, var_dict)
    

    if details:
        print(f"Imposing Window: [{E_min:.4f}; {E_max:.4f}]")
    

    
    A_ops_list = [op for op in var_dict.keys() if op.op != 'I']
    comm_constraint_expr = create_commutator_constraint(A_ops_list, symbolic_H, var_dict)
    
    if not scalar_flag :
        # matrix inequality constraints 
        constraints = [M_expr >> 0]
        constraints.append(M_H - E_min * M_expr >> 0) 
        constraints.append(-M_H + E_max * M_expr >> 0)

        
        flat_C1 = flat_M_H - E_min * flat_M_expr
        flat_C2 = -flat_M_H + E_max * flat_M_expr
        
        N_mat = len(basis)
        C1_mat = cp.reshape(flat_C1, (N_mat, N_mat), order='F')
        C2_mat = cp.reshape(flat_C2, (N_mat, N_mat), order='F')

        constraints = [M_expr >> 0]
        constraints.append(C1_mat >> 0) 
        constraints.append(C2_mat >> 0)
    else:
        #element wise(scalar) positivity constraint 
        flat_diff = flat_M_H - E_target * flat_M_expr
        constraints = [M_expr >> 0]
        eps = eps / np.sqrt(2)
        constraints.append(cp.real(flat_diff) >= -eps)
        constraints.append(cp.real(flat_diff) <= eps)
        constraints.append(cp.imag(flat_diff) >= -eps)
        constraints.append(cp.imag(flat_diff) <= eps)
    
    if isinstance(comm_constraint_expr, list):
        for expr in comm_constraint_expr:
            constraints.append(expr == 0)
    else:
        constraints.append(comm_constraint_expr == 0)
    
    id_key = None
    for key in var_dict.keys():
        if key.op == 'I':
            id_key = key
            break
    
    if id_key is not None:
        constraints.append(var_dict[id_key] == 1.0)

    if flag: 
        prob = cp.Problem(cp.Minimize(objective_expr), constraints)
    else:
        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    print(f"Total optimization variables: {prob.size_metrics.num_scalar_variables}")
    
    class FailedProblem:
        
        def __init__(self):
            self.value = np.nan
            self.status = "SolverError"

    try:
        prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND) 
        
        if details:
            print("problem status = ", prob.status)
            
        return prob

    except cp.error.SolverError:
        print("  [!] MOSEK crashed. Assigning NaN.")
        return FailedProblem()

def moment_sdp_flat_TI(N, symbolic_H, basis, symbolic_op, energy, eps, flag, details=False, scalar_flag = True):

    
    var_dict = {}
    E_target = energy
    E_min = E_target - eps
    E_max = E_target + eps
    
    basis = sort_basis(basis, N)
    M_expr, var_dict, symbolic_matrix, flat_M_expr = create_moment_matrix_vectorized_TI(basis, lattice_size=N)

    #circulant check
    size = len(symbolic_matrix)
    
    
   
    for row_i in range(1, size, N):
        row_f = min(row_i + N, size)
        
        for col_i in range(1, size, N):
            col_f = min(col_i + N, size)
            
            # Check the block!
            if is_block_circulant(symbolic_matrix, row_i, row_f, col_i, col_f):
                pass 
            else:
                print(f"Non-Circulant Block in Momemnt matrix: Block at rows {row_i}:{row_f}, cols {col_i}:{col_f}.")
                
                
    # if all_safe:
        # print("SUCCESS: The entire matrix is circulant by blocks!")
        # export_symbolic_matrix_to_csv(symbolic_matrix)

    basis_set = {Monomial(b.op, b.site, 1.0) for b in basis}
    missing_terms = []
    
    for _, h_term in symbolic_H:
        canonical_h = get_canonical_ti(h_term, N)
        clean_h = Monomial(canonical_h.op, canonical_h.site, 1.0)
        
        if clean_h not in var_dict and clean_h.op != 'I':
            if clean_h not in basis_set: 
                basis.append(clean_h)
                basis_set.add(clean_h)
                missing_terms.append(clean_h)

    if missing_terms:
        print(f"Forced {len(missing_terms)} missing terms into the basis. {missing_terms}")

    M_H, var_dict, _, flat_M_H = create_localizing_matrix_vectorized_TI(basis, symbolic_H, var_dict, N)
    

    
    # print("Computing objective function...")
    objective_expr = create_objective_function_TI(symbolic_op, var_dict, N)
    

    if details:
        print(f"Imposing Window: [{E_min:.4f}; {E_max:.4f}]")
    

    
    A_ops_list = [op for op in var_dict.keys() if op.op != 'I']
    comm_constraint_expr = create_commutator_constraint_TI(A_ops_list, symbolic_H, var_dict, N)
    constraints = [M_expr >> 0]
    if not scalar_flag :
        # matrix inequality constraints 
        constraints.append(M_H >> 0)
        constraints.append(M_H - E_min * M_expr >> 0) 
        constraints.append(-M_H + E_max * M_expr >> 0)
        

        
        flat_C1 = flat_M_H - E_min * flat_M_expr
        flat_C2 = -flat_M_H + E_max * flat_M_expr
        
        N_mat = len(basis)
        C1_mat = cp.reshape(flat_C1, (N_mat, N_mat), order='F')
        C2_mat = cp.reshape(flat_C2, (N_mat, N_mat), order='F')

    
        constraints.append(C1_mat >> 0) 
        constraints.append(C2_mat >> 0)
    else:
        #element wise(scalar) positivity constraint 
        flat_diff = flat_M_H - E_target * flat_M_expr
  
        eps = eps / np.sqrt(2)
        constraints.append(cp.real(flat_diff) >= -eps)
        constraints.append(cp.real(flat_diff) <= eps)
        constraints.append(cp.imag(flat_diff) >= -eps)
        constraints.append(cp.imag(flat_diff) <= eps)
    
    if isinstance(comm_constraint_expr, list):
        for expr in comm_constraint_expr:
            constraints.append(expr == 0)
    else:
        constraints.append(comm_constraint_expr == 0)
    
    id_key = None
    for key in var_dict.keys():
        if key.op == 'I':
            id_key = key
            break
    
    if id_key is not None:
        constraints.append(var_dict[id_key] == 1.0)

    if flag: 
        prob = cp.Problem(cp.Minimize(objective_expr), constraints)
    else:
        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    print(f"Total optimization variables: {prob.size_metrics.num_scalar_variables}")
    
    class FailedProblem:
        
        def __init__(self):
            self.value = np.nan
            self.status = "SolverError"

    try:
        prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND) 
        
        if details:
            print("problem status = ", prob.status)
            
        return prob

    except cp.error.SolverError:
        print("  [!] MOSEK crashed. Assigning NaN.")
        return FailedProblem()


def moment_sdp_block_diagonal_TI(N, symbolic_H, basis, symbolic_op, energy, eps, flag, details=False, scalar_flag=True):
    var_dict = {}
    E_target = energy
    E_min = E_target - eps
    E_max = E_target + eps

    basis = sort_basis(basis, N)

    M_expr_list, var_dict, symbolic_matrix = create_block_diagonal_moment_matrix(basis, N)

    N_mat = len(basis)

    non_id_keys = [k for k in var_dict.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N_mat * N_mat, dtype=complex)

    for c in range(N_mat):
        for r in range(N_mat):
            m = symbolic_matrix[r][c]

            if r == 0 or c == 0:
                m = get_canonical_ti(m, N)

            flat_idx = c * N_mat + r

            if m.op == 'I':
                b[flat_idx] += m.phase
            else:
                clean_key = Monomial(m.op, m.site, 1.0)
                if clean_key in key_to_col:
                    A_rows.append(flat_idx)
                    A_cols.append(key_to_col[clean_key])
                    A_data.append(m.phase)

    if len(non_id_keys) > 0:
        X = cp.hstack([var_dict[k] for k in non_id_keys])
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N_mat * N_mat, len(non_id_keys)))
        flat_M_expr = A @ X + b
    else:
        flat_M_expr = b

    M_expr = cp.reshape(flat_M_expr, (N_mat, N_mat), order='F')

    for row_i in range(1, N_mat, N):
        row_f = min(row_i + N, N_mat)
        for col_i in range(1, N_mat, N):
            col_f = min(col_i + N, N_mat)
            if not is_block_circulant(symbolic_matrix, row_i, row_f, col_i, col_f):
                print(f"Non-Circulant Block in Moment matrix: Block at rows {row_i}:{row_f}, cols {col_i}:{col_f}.")

    basis_set = {Monomial(b.op, b.site, 1.0) for b in basis}
    missing_terms = []

    for _, h_term in symbolic_H:
        canonical_h = get_canonical_ti(h_term, N)
        clean_h = Monomial(canonical_h.op, canonical_h.site, 1.0)

        if clean_h not in var_dict and clean_h.op != 'I':
            if clean_h not in basis_set:
                basis.append(clean_h)
                basis_set.add(clean_h)
                missing_terms.append(clean_h)

    if missing_terms:
        print(f"Found {len(missing_terms)} missing terms from the basis. {missing_terms}")




    objective_expr = create_objective_function_TI(symbolic_op, var_dict, N)

    if details:
        print(f"Imposing Window: [{E_min:.4f}; {E_max:.4f}]")

    # A_ops_list = [op for op in var_dict.keys() if op.op != 'I']

    constraints = []

    # M_comm, var_dict, _, flat_M_comm = create_conjugate_commutator_matrix_vectorized(basis, symbolic_H, var_dict, N)

    # constraints.append(0.5 * cp.real(flat_M_comm) >= 0)
    # constraints.append(cp.imag(flat_M_comm) == 0)

    for Mk in M_expr_list:
        constraints.append(Mk >> 0)

    if not scalar_flag:
        # M_H, var_dict, _, flat_M_H = create_localizing_matrix_vectorized_TI(basis, symbolic_H, var_dict, N)
        # # constraints.append(M_H >> 0)
        # # constraints.append(M_H - E_min * M_expr >> 0)
        # # constraints.append(-M_H + E_max * M_expr >> 0)
        #
        # flat_C1 = flat_M_H - E_min * flat_M_expr
        # flat_C2 = -flat_M_H + E_max * flat_M_expr
        #
        # C1_mat = cp.reshape(flat_C1, (N_mat, N_mat), order='F')
        # C2_mat = cp.reshape(flat_C2, (N_mat, N_mat), order='F')
        #
        # constraints.append(C1_mat >> 0)
        # constraints.append(C2_mat >> 0)

        M_H_expr_list, var_dict, _, flat_M_H = create_block_diagonal_localizing_matrix(basis, symbolic_H, var_dict, N)

        for Mk, Mk_H in zip(M_expr_list, M_H_expr_list):
            # M_H - E_min * M >= 0 (for this specific momentum block)
            constraints.append(Mk_H - E_min * Mk >> 0)

            # E_max * M - M_H >= 0 (for this specific momentum block)
            constraints.append(-Mk_H + E_max * Mk >> 0)

    else:

        _, var_dict, _, flat_M_H, valid_mask = create_localizing_matrix_vectorized_TI_cached(basis, symbolic_H, var_dict, N)

        flat_diff = flat_M_H - E_target * flat_M_expr


        constraints.append(cp.real(flat_diff)[valid_mask] >= -eps / np.sqrt(2))
        constraints.append(cp.real(flat_diff)[valid_mask] <= eps / np.sqrt(2))
        constraints.append(cp.imag(flat_diff)[valid_mask] >= -eps / np.sqrt(2))
        constraints.append(cp.imag(flat_diff)[valid_mask] <= eps / np.sqrt(2))

    A_ops_list = [op for op in basis if op.op != 'I']
    comm_constraint_expr = create_commutator_constraint_TI(A_ops_list, symbolic_H, var_dict, N)

    if isinstance(comm_constraint_expr, list):
        for expr in comm_constraint_expr:
            constraints.append(expr == 0)
    else:
        constraints.append(comm_constraint_expr == 0)

    id_key = None
    for key in var_dict.keys():
        if key.op == 'I':
            id_key = key
            break

    if id_key is not None:
        constraints.append(var_dict[id_key] == 1.0)

    if flag:
        prob = cp.Problem(cp.Minimize(objective_expr), constraints)
    else:
        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    print(f"Total optimization variables: {prob.size_metrics.num_scalar_variables}")

    class FailedProblem:
        def __init__(self):
            self.value = np.nan
            self.status = "SolverError"

    try:
        prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
        # prob.solve(solver=cp.SCS, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND) #,alpha = 1.8,scale=1.0, warm_start=True, eps=1e-5
        if details:
            print("problem status = ", prob.status)
        return prob

    except cp.error.SolverError:
        print("  [!] Solver crashed. Assigning NaN.")
        return FailedProblem()


def moment_sdp_TONI(N, symbolic_H, basis, symbolic_op, energy, eps, flag, details=False, scalar_flag=True):
    var_dict = {}
    E_target = energy
    E_min = E_target - eps
    E_max = E_target + eps



    basis = sort_basis(basis, N)

    M_expr_list, var_dict, symbolic_matrix = create_block_diagonal_moment_matrix(basis, N)

    N_mat = len(basis)

    non_id_keys = [k for k in var_dict.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N_mat * N_mat, dtype=complex)

    for c in range(N_mat):
        for r in range(N_mat):
            m = symbolic_matrix[r][c]

            if r == 0 or c == 0:
                m = get_canonical_ti(m, N)

            flat_idx = c * N_mat + r

            if m.op == 'I':
                b[flat_idx] += m.phase
            else:
                clean_key = Monomial(m.op, m.site, 1.0)
                if clean_key in key_to_col:
                    A_rows.append(flat_idx)
                    A_cols.append(key_to_col[clean_key])
                    A_data.append(m.phase)

    if len(non_id_keys) > 0:
        X = cp.hstack([var_dict[k] for k in non_id_keys])
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N_mat * N_mat, len(non_id_keys)))
        flat_M_expr = A @ X + b
    else:
        flat_M_expr = b

    M_expr = cp.reshape(flat_M_expr, (N_mat, N_mat), order='F')

    for row_i in range(1, N_mat, N):
        row_f = min(row_i + N, N_mat)
        for col_i in range(1, N_mat, N):
            col_f = min(col_i + N, N_mat)
            if not is_block_circulant(symbolic_matrix, row_i, row_f, col_i, col_f):
                print(f"Non-Circulant Block in Moment matrix: Block at rows {row_i}:{row_f}, cols {col_i}:{col_f}.")

    objective_expr = create_objective_function_TI(symbolic_op, var_dict, N)

    if details:
        print(f"Imposing Window: [{E_min:.4f}; {E_max:.4f}]")

    constraints = []


    for Mk in M_expr_list:
        constraints.append(Mk >> 0)


    H_expr = create_objective_function_TI(symbolic_H, var_dict, N)

    constraints.append(cp.real(H_expr) <= E_max)
    constraints.append(cp.real(H_expr) >= E_min)





    # A_ops_list = [op for op in basis if op.op != 'I']
    # comm_constraint_expr = create_commutator_constraint_TI(A_ops_list, symbolic_H, var_dict, N)
    #
    # if isinstance(comm_constraint_expr, list):
    #     for expr in comm_constraint_expr:
    #         constraints.append(expr == 0)
    # else:
    #     constraints.append(comm_constraint_expr == 0)


    id_key = None
    for key in var_dict.keys():
        if key.op == 'I':
            id_key = key
            break

    if id_key is not None:
        constraints.append(var_dict[id_key] == 1.0)

    if flag:
        prob = cp.Problem(cp.Minimize(objective_expr), constraints)
    else:
        prob = cp.Problem(cp.Maximize(objective_expr), constraints)

    print(f"Total optimization variables: {prob.size_metrics.num_scalar_variables}")

    class FailedProblem:
        def __init__(self):
            self.value = np.nan
            self.status = "SolverError"

    try:
        prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
        if details:
            print("problem status = ", prob.status)
        return prob

    except cp.error.SolverError:
        print("  [!] Solver crashed. Assigning NaN.")
        return FailedProblem()


def GS_sdp_LB(N, symbolic_H, basis, details=False):
    var_dict = {}

    basis = sort_basis(basis, N)

    M_expr_list, var_dict, symbolic_matrix = create_block_diagonal_moment_matrix(basis, N)

    N_mat = len(basis)

    non_id_keys = [k for k in var_dict.keys() if k.op != 'I']
    key_to_col = {key: col for col, key in enumerate(non_id_keys)}

    A_rows, A_cols, A_data = [], [], []
    b = np.zeros(N_mat * N_mat, dtype=complex)

    for c in range(N_mat):
        for r in range(N_mat):
            m = symbolic_matrix[r][c]

            if r == 0 or c == 0:
                m = get_canonical_ti(m, N)

            flat_idx = c * N_mat + r

            if m.op == 'I':
                b[flat_idx] += m.phase
            else:
                clean_key = Monomial(m.op, m.site, 1.0)
                if clean_key in key_to_col:
                    A_rows.append(flat_idx)
                    A_cols.append(key_to_col[clean_key])
                    A_data.append(m.phase)

    if len(non_id_keys) > 0:
        X = cp.hstack([var_dict[k] for k in non_id_keys])
        A = csc_matrix((A_data, (A_rows, A_cols)), shape=(N_mat * N_mat, len(non_id_keys)))
        flat_M_expr = A @ X + b
    else:
        flat_M_expr = b

    M_expr = cp.reshape(flat_M_expr, (N_mat, N_mat), order='F')

    for row_i in range(1, N_mat, N):
        row_f = min(row_i + N, N_mat)
        for col_i in range(1, N_mat, N):
            col_f = min(col_i + N, N_mat)
            if not is_block_circulant(symbolic_matrix, row_i, row_f, col_i, col_f):
                print(f"Non-Circulant Block in Moment matrix: Block at rows {row_i}:{row_f}, cols {col_i}:{col_f}.")

    objective_expr = create_objective_function_TI(symbolic_H, var_dict, N)

    constraints = []

    for Mk in M_expr_list:
        constraints.append(Mk >> 0)

    id_key = None
    for key in var_dict.keys():
        if key.op == 'I':
            id_key = key
            break

    if id_key is not None:
        constraints.append(var_dict[id_key] == 1.0)

    prob = cp.Problem(cp.Minimize(cp.real(objective_expr)), constraints)

    print(f"Total optimization variables: {prob.size_metrics.num_scalar_variables}")

    class FailedProblem:
        def __init__(self):
            self.value = np.nan
            self.status = "SolverError"

    try:
        prob.solve(solver=cp.MOSEK, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
        if details:
            print("problem status = ", prob.status)
        return prob

    except cp.error.SolverError:
        print("  [!] Solver crashed. Assigning NaN.")
        return FailedProblem()