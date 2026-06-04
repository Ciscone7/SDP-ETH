from itertools import product
import re
from collections import defaultdict
import numpy as np
from scipy.sparse import kron, identity, csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
import time
from qutip import *
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.sparse import csc_matrix

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

    def dagger(self):
        return Monomial(self.op, self.site, self.phase.conjugate())

def spin_multiplication(a: Monomial, b: Monomial) -> Monomial:
    # 1. Pauli Algebra Rules
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
            clean_monomial = Monomial("I", (0,0,0), 1.0) 
        else:
            clean_monomial = Monomial(current_product.op, current_product.site, 1.0)

        if clean_monomial not in seen_keys:
            seen_keys.add(clean_monomial)
            result_monomials.append(clean_monomial)

    return result_monomials

def generate_neighbour_monomials(initial_list, m, pbc=False):
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
            site_idx = i + j
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

    return result_monomials

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

def kron_n(ops):
    
    result = ops[0]
    for op in ops[1:]:
        result = kron(result, op, format='csr')
    return result

def Hamiltonian(N, j, g):
    H = np.zeros((2**N, 2**N), dtype=complex)

    for i in range (N - 1):
        ops = [id2] * N
        ops[i] = sz
        ops[(i + 1)] = sz
        H -= j * kron_n(ops)

    for i in range(N):
        ops = [id2] * N
        ops[i] = sx
        H -= g * kron_n(ops)

    H = Qobj(H, dims=[[2]*N, [2]*N])
    return H

def get_symbolic_hamiltonian(N, J, g):
    terms = []
    

    for i in range(1, N):  
        m = Monomial("ZZ", [(i, 0, 0), (i+1, 0, 0)], 1.0)
        terms.append((-J, m)) 

    for i in range(1, N + 1):
        m = Monomial("X", [(i, 0, 0)], 1.0)
        terms.append((-g, m)) 
        
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

    # Build -J2 * sum sigma^z_i sigma^z_{i+2}
    for i in range(L-2):
        ops = [id2] * L
        ops[i] = sz
        ops[(i + 2)] = sz
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

    for i in range(L-1):
        ops = [id2] * L
        ops[i] = sy
        ops[(i + 1)] = sy
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
            monomial = symbolic_matrix[i][j]
            groups[monomial].append((i, j, monomial.phase))

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


