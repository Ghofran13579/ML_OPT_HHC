import os
import re
import math
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import traceback
import json
import traceback
import time
# =========================
# PATHS
# =========================
INPUT_FOLDER  = r"C:\Users\Ghofran MASSAOUDI\Desktop\output_equitable_Correct_stucture"
OUTPUT_FOLDER = r"C:\Users\Ghofran MASSAOUDI\Desktop\Q-Learnung_NCRO_PE_Based_test"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# MODEL-CONSISTENT PARAM BUILDERS
# =========================
def _vflag_for_nurse(df: pd.DataFrame, nurse_id: int) -> int:
    """Vehicle flag v_n: 1 = shared (Depot), 0 = personal (Nurse House)."""
    dep = df.loc[df[COL_NURSE_ID] == nurse_id, COL_DEPOT].dropna()
    if len(dep) and parse_xy(str(dep.iloc[0])) is not None:
        return 1
    home = df.loc[df[COL_NURSE_ID] == nurse_id, COL_NURSE_HOUSE].dropna()
    return 0 if len(home) and parse_xy(str(home.iloc[0])) is not None else 1


def _coords_for_patient_nodes(df: pd.DataFrame, pts: list[int]) -> dict[int, tuple[float, float]]:
    coords = {}
    for i, pid in enumerate(pts, start=1):
        sub = df[df[COL_PATIENT].astype("Int64") == pid]
        if sub.empty:
            coords[i] = (0.0, 0.0)
        else:
            xy = parse_xy(sub.iloc[0].get(COL_PL, None))
            coords[i] = xy if xy else (0.0, 0.0)
    return coords


def _nurse_depot_or_home(df: pd.DataFrame, nurse_id: int) -> tuple[float, float]:
    """Return depot/home coordinates following v_n flag."""
    v = _vflag_for_nurse(df, nurse_id)
    if v == 1:  # shared
        dep = df.loc[df[COL_NURSE_ID] == nurse_id, COL_DEPOT].dropna()
        if len(dep):
            xy = parse_xy(str(dep.iloc[0]))
            if xy:
                return xy
        home = df.loc[df[COL_NURSE_ID] == nurse_id, COL_NURSE_HOUSE].dropna()
        if len(home):
            xy = parse_xy(str(home.iloc[0]))
            if xy:
                return xy
    else:  # personal
        home = df.loc[df[COL_NURSE_ID] == nurse_id, COL_NURSE_HOUSE].dropna()
        if len(home):
            xy = parse_xy(str(home.iloc[0]))
            if xy:
                return xy
        dep = df.loc[df[COL_NURSE_ID] == nurse_id, COL_DEPOT].dropna()
        if len(dep):
            xy = parse_xy(str(dep.iloc[0]))
            if xy:
                return xy
    return (50.0, 50.0)


def _service_time(df: pd.DataFrame, pid: int) -> float:
    row = df[df[COL_PATIENT].astype("Int64") == pid]
    return float(row.iloc[0].get(COL_SERVICE, 0.0)) if len(row) else 0.0


def _qual_pref(df: pd.DataFrame, pid: int, nurse_id: int) -> tuple[int, int]:
    """Return (q_i^n, S_i^n) ∈ [1..5]; defaults (3,3)."""
    sub = df[(df[COL_PATIENT].astype("Int64") == pid) & (df[COL_NURSE_ID] == nurse_id)]
    if len(sub):
        qv = int(sub.iloc[0].get(COL_QUAL, 3) or 3)
        sv = int(sub.iloc[0].get(COL_PREF, 3) or 3)
    else:
        qv, sv = 3, 3
    return max(1, min(5, qv)), max(1, min(5, sv))
# --- Target F2 bands from MILP, used only for calibration intuition ---
# ==== Targets (from your MILP) ====
F2_TARGET = {
    "c": 1400.0,   # clustered
    "r": 410.0,    # random
    "rc": 500.0,   # mixed
    "other": 1300.0
}

def _family_of(csv_name: str) -> str:
    n = (csv_name or "").lower()
    if n.startswith("rc"): return "rc"
    if n.startswith("c"):  return "c"
    if n.startswith("r"):  return "r"
    return "other"

def _estimate_total_service(df: pd.DataFrame) -> float:
    """Sum of service times over unique patients (one visit each)."""
    p = df["Patient"].dropna().astype("Int64").drop_duplicates()
    sv = 0.0
    for pid in p:
        sv += float(df.loc[df["Patient"].astype("Int64") == pid, "Service Time"].iloc[0])
    return sv

def _service_scale_for_instance(csv_name: str) -> float:
    """
    Service times should NOT be heavily scaled down.
    MILP uses actual service times from data.
    """
    name = csv_name.lower()
    
    # Use 90-100% of actual service times for all families
    if name.startswith("c"):
        scale = 0.95
    elif name.startswith("r") and not name.startswith("rc"):
        scale = 0.98   # ← Changed from 0.48
    elif name.startswith("rc"):
        scale = 0.96   # ← Changed from 0.52
    else:
        scale = 0.95
        
    return float(scale)
# =========================
# PARAMETERS
# =========================
POP_SIZE      = 100
MAX_ITERS     = 100
INITIAL_KE    = 1000.0
KE_LOSS_RATE  = 0.2
MOLE_COLL     = 0.8
ALPHA         = 15
BETA          = 10.0
BUFFER_INIT   = 0.0
FELIMIT       = 25000
DN_PER_TIME   = 1.0
random.seed(42)
np.random.seed(42)

# =========================
# CSV SCHEMA
# =========================
COL_NURSE       = "Nurse"
COL_NURSE_HOUSE = "Nurse House"
COL_PATIENT     = "Patient"
COL_PL          = "Patient Location"
COL_SERVICE     = "Service Time"
COL_COST        = "Cost"
COL_PREF        = "Preferences of the Patients"
COL_QUAL        = "Qualification of the Nurses"
COL_LN          = "Maximum Regular Time"
COL_DEPOT       = "Depot"
COL_NURSE_ID    = "NurseID"
COORD_PATTERN = re.compile(r"\(?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)?")

def parse_xy(val: Any) -> Optional[Tuple[float, float]]:
    if isinstance(val, str):
        m = COORD_PATTERN.match(val.strip())
        if m:
            return float(m.group(1)), float(m.group(2))
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            return float(val[0]), float(val[1])
        except Exception:
            return None
    return None

def euclid(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])

# =========================
# MODEL-CONSISTENT EVALUATION
# =========================
TRAVEL_COST_RATE = 1.0
OVERTIME_RATE_FALLBACK = DN_PER_TIME
LAMBDA_ASSIGN = 500.0
LAMBDA_FLOW = 200.0
LAMBDA_2CYCLE = 200.0


# =========================
# OBJECTIVE HELPERS
# =========================
def nurse_Ln(df: pd.DataFrame, nurse_id: int, default_ln: float = 480.0) -> float:
    """
    Return L_n: maximum regular working time allowed for nurse n before overtime.
    If not found in the dataset, return default (480 minutes = 8h).
    """
    COL_NURSE_ID = "NurseID"
    COL_LN = "Maximum Regular Time"

    if COL_LN in df.columns:
        vals = df.loc[df[COL_NURSE_ID] == nurse_id, COL_LN].dropna()
        if len(vals):
            try:
                return float(vals.iloc[0])
            except Exception:
                pass
    return float(default_ln)

def extract_route_sequence(solution: List[int], nurse_id: int, df: pd.DataFrame) -> List[int]:
    """
    Extract the ACTUAL patient visit sequence for a nurse from the solution vector.
    
    Returns: List of patient IDs in visit order (not including depot nodes)
    """
    routes = decode_routes(solution, df)
    return routes.get(nurse_id, [])


# =====================================================
# CORRECTED: Build edges from actual route sequence
# =====================================================
def _edges_from_sequence(patient_sequence: List[int], J: int) -> set[tuple[int, int]]:
    """
    Convert patient visit sequence to edge set with depot start/end.
    
    Args:
        patient_sequence: Actual order of patient visits [p1, p3, p5, ...]
        J: Total number of patients this nurse serves
    
    Returns:
        Set of edges (i,j) where nodes are indexed 0..J+1:
        - Node 0: depot start
        - Nodes 1..J: patients in visit order
        - Node J+1: depot end
    """
    if not patient_sequence:
        return set()
    
    edges = set()
    
    # Start: depot → first patient
    edges.add((0, 1))
    
    # Patient-to-patient transitions
    for k in range(len(patient_sequence) - 1):
        edges.add((k + 1, k + 2))
    
    # End: last patient → depot
    edges.add((J, J + 1))
    
    return edges


# =====================================================
# CORRECTED: Build coordinate mapping from ACTUAL sequence
# =====================================================
def _coords_from_sequence(df: pd.DataFrame, nurse_id: int, 
                          patient_sequence: List[int]) -> Dict[int, Tuple[float, float]]:
    """
    Map node indices to coordinates based on actual visit order.
    
    Returns:
        coords[0] = depot start
        coords[1..J] = patients in visit order
        coords[J+1] = depot end
    """
    J = len(patient_sequence)
    coords = {}
    
    # Depot coordinates (both start and end)
    depot_xy = _nurse_depot_or_home(df, nurse_id)
    coords[0] = depot_xy
    coords[J + 1] = depot_xy
    
    # Patient coordinates in actual visit order
    for idx, patient_id in enumerate(patient_sequence, start=1):
        sub = df[df[COL_PATIENT].astype("Int64") == patient_id]
        if sub.empty:
            coords[idx] = (0.0, 0.0)
        else:
            xy = parse_xy(sub.iloc[0].get(COL_PL, None))
            coords[idx] = xy if xy else (0.0, 0.0)
    
    return coords
# =====================================================
# CALIBRATED TIME–COST MATRIX BUILDER (aligned with MILP)
# =====================================================
def _build_TC_for_actual_route(df: pd.DataFrame, nurse_id: int,
                               patient_sequence: List[int],
                               csv_name: str = "") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build time and cost matrices reflecting the ACTUAL visit order.
    
    This replaces _build_T_C_for_nurse to use real route structure.
    """
    J = len(patient_sequence)
    if J == 0:
        return np.zeros((2, 2)), np.zeros((2, 2)), {0: (0, 0), 1: (0, 0)}
    
    # Get coordinates in actual visit order
    coords = _coords_from_sequence(df, nurse_id, patient_sequence)
    
    # Initialize matrices
    Tn = np.zeros((J + 2, J + 2))
    Cn = np.zeros((J + 2, J + 2))
    
    # Family-specific calibration
    fam = _family_of(csv_name)
    
    if fam == "c":
        TIME_FACTOR = 0.50
        COST_PER_MIN = 1.0
    elif fam == "r":
        TIME_FACTOR = 0.60  # Critical for random instances
        COST_PER_MIN = 1.0
    elif fam == "rc":
        TIME_FACTOR = 0.55
        COST_PER_MIN = 1.0
    else:
        TIME_FACTOR = 0.50
        COST_PER_MIN = 1.0
    
    # Compute travel times and costs
    for i in range(J + 2):
        for j in range(J + 2):
            if i == j:
                continue
            dist = euclid(coords[i], coords[j])
            travel_time = dist * TIME_FACTOR
            Tn[i, j] = travel_time
            Cn[i, j] = travel_time * COST_PER_MIN
    
    return Tn, Cn, coords
    

# =========================
# MODEL CONSISTENCY LAYER (enforce MILP-style constraints)
# =========================

PEN_ASSIGN      = 500.0   # patient coverage violation
PEN_FLOW        = 200.0   # flow conservation violation
PEN_DEPOT       = 200.0   # depot start/end violation
PEN_2CYCLE      = 200.0   # 2-node subtour violation
PEN_TIME        = 200.0   # time propagation violation (t >= a, etc.)
BIG_M = 10000.0            # large constant for precedence and timing
def _vn_flag(df: pd.DataFrame, nurse_id: int) -> Optional[int]:
    """
    v_n ∈ {0,1} decision surrogate:
    1 = shared vehicle (depot start), 0 = personal vehicle (home start).
    If not provided, defaults to data-driven estimate.
    """
    if "Vehicle" in df.columns:
        vals = df.loc[df[COL_NURSE_ID] == nurse_id, "Vehicle"].dropna()
        if len(vals):
            try:
                return int(vals.iloc[0])
            except Exception:
                pass
    # fallback: detect from coordinates (legacy behaviour)
    return _vflag_for_nurse(df, nurse_id)
def _coords_and_TC(df, nurse_id, local_pids, csv_name):
    """Convenience: build coords, Tn, Cn, indices."""
    J = len(local_pids)
    Tn, Cn, coords = _build_TC_for_actual_route(df, nurse_id, local_pids, csv_name=csv_name)
    V = list(range(0, J + 2))  # 0 .. J .. J+1
    return J, V, Tn, Cn, coords

def _sequential_path_edges(J: int):
    """Hamiltonian path 0->1->2->...->J->J+1 (no cycles)."""
    edges = {(0, 1)}
    edges |= {(k, k + 1) for k in range(1, J)}
    edges |= {(J, J + 1)}
    return edges

def _flow_degrees(J: int, edges: set[tuple[int,int]]) -> tuple[dict,int]:
    """Return in/out degrees per node in 0..J+1 and number of violations."""
    indeg = {k: 0 for k in range(0, J + 2)}
    outdeg = {k: 0 for k in range(0, J + 2)}
    for (i, j) in edges:
        if i == j: 
            continue
        outdeg[i] += 1
        indeg[j]  += 1

    viol = 0
    # depot start: outdeg[0] == 1, indeg[0] == 0
    if outdeg[0] != 1 or indeg[0] != 0:
        viol += 1
    # depot end: indeg[J+1] == 1, outdeg[J+1] == 0
    if indeg[J+1] != 1 or outdeg[J+1] != 0:
        viol += 1
    # each job 1..J : indeg == outdeg == 1
    for k in range(1, J + 1):
        if indeg[k] != 1 or outdeg[k] != 1:
            viol += 1
    return ({"in": indeg, "out": outdeg}, viol)

def _eliminate_two_cycles(J: int, edges: set[tuple[int,int]]) -> tuple[set[tuple[int,int]], int]:
    """
    Remove 2-node cycles among patient nodes (i<->j).
    Since we ultimately use a sequential path, this is mostly a checker,
    but we clean any found pairs just in case.
    """
    to_remove = set()
    viol = 0
    for i in range(1, J + 1):
        for j in range(1, J + 1):
            if i != j and (i, j) in edges and (j, i) in edges:
                viol += 1
                # arbitrarily remove (j,i)
                to_remove.add((j, i))
    if to_remove:
        edges = edges.difference(to_remove)
    return edges, viol

def _check_depot_counts(v_n: int, J: int, edges: set[tuple[int,int]]) -> float:
    """
    Enforce depot selection constraints:
      ∑_j x_{0j}^n ≤ v_n   (shared car)
      ∑_i x_{i,J+1}^n ≤ 1 - v_n   (personal car)
    """
    out0 = sum(1 for (i, j) in edges if i == 0)
    inEnd = sum(1 for (i, j) in edges if j == J + 1)
    pen = 0.0
    if out0 > v_n:
        pen += PEN_DEPOT * (out0 - v_n)
    if inEnd > (1 - v_n):
        pen += PEN_DEPOT * (inEnd - (1 - v_n))
    return pen

def _timeline_and_overtime_with_arrival(J, Tn, s, Ln):
    """
    Build explicit arrival a[i], start t[i], end[i],
    and compute overtime if total work > L_n.
    
    This function is already correct - no changes needed.
    """
    a = np.zeros(J + 2)
    t = np.zeros(J + 2)
    endt = np.zeros(J + 2)
    viol = 0

    # Depot start (node 0): no service
    t[0] = 0.0
    endt[0] = 0.0

    # First patient (node 1)
    a[1] = Tn[0, 1]
    t[1] = a[1]
    endt[1] = t[1] + s[1]

    # Subsequent patients
    for i in range(2, J + 1):
        a[i] = endt[i - 1] + Tn[i - 1, i]
        t[i] = a[i]
        endt[i] = t[i] + s[i]
        
        if endt[i] < t[i]:
            viol += 1

    # Final return to depot
    a[J + 1] = endt[J] + Tn[J, J + 1]
    t[J + 1] = a[J + 1]
    endt[J + 1] = t[J + 1]

    # Total working time = final return time
    total_time = endt[J + 1]
    
    # Overtime = excess beyond regular hours
    OT_n = max(0.0, total_time - Ln)
    
    return a, t, endt, total_time, OT_n, viol

def enforce_and_score_one_nurse_CORRECTED(df, nurse_id, patient_sequence, csv_name,
                                         SERVICE_SCALE, d_n_fallback) -> Tuple[float, float, float, float]:
    """
    Evaluate one nurse's route using ACTUAL patient visit order.
    
    Args:
        patient_sequence: Actual ordered list of patient IDs to visit
    
    Returns:
        (Z1, Z2, Z3, penalties)
    """
    if not patient_sequence:
        return 0.0, 0.0, 0.0, 0.0
    
    J = len(patient_sequence)
    Ln = nurse_Ln(df, nurse_id)
    d_n = get_overtime_rate(df, nurse_id) or d_n_fallback
    v_n = _vn_flag(df, nurse_id)
    
    # Build matrices for ACTUAL route
    Tn, Cn, coords = _build_TC_for_actual_route(df, nurse_id, patient_sequence, csv_name)
    
    # Build edges from actual sequence
    edges = _edges_from_sequence(patient_sequence, J)
    
    penalties = 0.0
    
    # Depot constraints
    penalties += _check_depot_counts(v_n, J, edges)
    
    # Flow validation (should be perfect for sequential routes)
    _, flow_viol = _flow_degrees(J, edges)
    if flow_viol > 0:
        penalties += PEN_FLOW * flow_viol
    
    # Service times (scaled)
    s = np.zeros(J + 2)
    for i in range(1, J + 1):
        s[i] = SERVICE_SCALE * _service_time(df, patient_sequence[i - 1])
    
    # Timeline calculation
    a, t, endt, work_n, OT_n, time_viol = _timeline_and_overtime_with_arrival(J, Tn, s, Ln)
    if time_viol > 0:
        penalties += PEN_TIME * time_viol
    
    # ===== OBJECTIVES =====
    # Z2: Total working time (including travel + service + return)
    Z2_n = work_n
    
    # Z1: Travel cost + overtime cost
    Z1_travel = sum(Cn[i, j] for (i, j) in edges)
    Z1_n = Z1_travel + d_n * OT_n
    
    # Z3: Qualification-preference mismatch
    # inside enforce_and_score_one_nurse_CORRECTED
    Z3_n = 0.0
    for patient_id in patient_sequence:
        qv, sv = _qual_pref(df, patient_id, nurse_id)
        Z3_n += abs(qv - sv)  # smaller=better
# and in q_ndsa_rank caller use objs_min = [F1, F2, +F3]  # all are min

    
    return Z1_n, Z2_n, Z3_n, penalties

def check_inter_nurse_precedence(routes: dict[int, list[int]],
                                 all_timeline: dict[int, dict[int, float]],
                                 service_times: dict[int, float],
                                 M: float = BIG_M) -> float:
    """
    Enforce cross-nurse precedence:
      t_i^n + s_i ≤ t_i^m + M (1 - y_i^n y_i^m)
    For each patient visited by two nurses (rare).
    """
    pen = 0.0
    visits = {}
    for n, seq in routes.items():
        for p in seq:
            visits.setdefault(p, []).append(n)

    for p, nurses in visits.items():
        if len(nurses) <= 1:
            continue
        for i in range(len(nurses)):
            for j in range(i + 1, len(nurses)):
                n1, n2 = nurses[i], nurses[j]
                t1 = all_timeline[n1].get(p, 0)
                t2 = all_timeline[n2].get(p, 0)
                s_i = service_times.get(p, 0)
                if (t1 + s_i) > (t2 + BIG_M * (1 - 1)):  # both assigned
                    pen += PEN_TIME
    return pen

# =====================================================
# CORRECTED EVALUATION FUNCTION (boosted scaling)
# =====================================================
def evaluate_solution_CORRECTED(solution: List[int], df: pd.DataFrame, 
                               csv_name: str = "") -> Tuple[float, float, float]:
    """
    Evaluate NCRO solution with proper route structure handling.
    """
    # Normalize to ensure valid structure
    sln_norm = normalize_solution(solution, df)
    routes = decode_routes(sln_norm, df)
    all_pids = patient_list(df)
    
    # Family-specific parameters
    fam = _family_of(csv_name)
    
    # Moderate overtime penalties
    if fam == "c":      
        DN_PER_TIME = 0.5
    elif fam == "r":
        DN_PER_TIME = 0.5  # Keep moderate for random
    elif fam == "rc":
        DN_PER_TIME = 0.5
    else:               
        DN_PER_TIME = 0.5
    
    # Service scaling (use near-full service times)
    if fam == "c":
        SERVICE_SCALE = 0.95
    elif fam == "r":
        SERVICE_SCALE = 0.98  # 98% of actual service time
    elif fam == "rc":
        SERVICE_SCALE = 0.96
    else:
        SERVICE_SCALE = 0.95
    
    # Patient coverage penalty
    penalties_global = 0.0
    seen = {}
    for _, seq in routes.items():
        for p in seq:
            seen[p] = seen.get(p, 0) + 1
    
    for p in all_pids:
        if seen.get(p, 0) != 1:
            penalties_global += PEN_ASSIGN
    
    # Per-nurse accumulation with CORRECTED evaluation
    Z1_total, Z2_total, Z3_total = 0.0, 0.0, 0.0
    
    for nurse_id, patient_sequence in routes.items():
        if not patient_sequence:
            continue
            
        Z1_n, Z2_n, Z3_n, pen_n = enforce_and_score_one_nurse_CORRECTED(
            df, nurse_id, patient_sequence, csv_name, SERVICE_SCALE, DN_PER_TIME)
        
        Z1_total += Z1_n
        Z2_total += Z2_n
        Z3_total += Z3_n
        penalties_global += pen_n
    
    # Add light penalty influence
    Z1_total += 0.01 * penalties_global
    Z2_total += 0.01 * penalties_global
    
    return Z1_total, Z2_total, Z3_total

# =========================
# ENCODING HELPERS
# =========================
def generate_solution(df: pd.DataFrame, num_nurses: int) -> List[int]:
    patients = df[COL_PATIENT].dropna().astype(int).drop_duplicates().tolist()
    random.shuffle(patients)

    nurse_ids = df[COL_NURSE_ID].drop_duplicates().astype(int).tolist()
    random.shuffle(nurse_ids)
    nurse_ids = nurse_ids[:min(num_nurses, len(nurse_ids))]

    P, N = len(patients), len(nurse_ids)
    if N == 0:
        return patients[:]

    # split patients into N parts
    if P <= 1:
        parts = [patients[:]]
    else:
        k = max(1, N-1)
        splits = sorted(random.sample(range(1, P), k=min(k, P-1)))
        parts, prev = [], 0
        for sp in splits:
            parts.append(patients[prev:sp]); prev = sp
        parts.append(patients[prev:])

    sol = []
    for nid, group in zip(nurse_ids, parts):
        sol.append(-int(nid))
        sol.extend(group)
    return sol

def detect_num_nurses(df: pd.DataFrame) -> int:
    return int(max(1, df[COL_NURSE_ID].nunique()))


# =========================
# NORMALIZATION HELPERS (to keep nurse delimiters valid)
# =========================
def _is_delim(x):
    """Return True if x is a nurse delimiter (negative int)."""
    return isinstance(x, (int, float)) and x < 0

def nurse_ids_list(df: pd.DataFrame) -> list[int]:
    ids = df[COL_NURSE_ID].dropna().astype(int).drop_duplicates().tolist()
    ids.sort()
    return ids

def patient_list(df: pd.DataFrame) -> list[int]:
    return df[COL_PATIENT].dropna().astype(int).drop_duplicates().tolist()

def normalize_solution(sln: list[int], df: pd.DataFrame) -> list[int]:
    """Ensure each patient appears once, keep nurse blocks intact."""
    nurses = nurse_ids_list(df)
    pats = patient_list(df)
    
    # Parse existing structure
    routes = {nid: [] for nid in nurses}
    cur_nurse = None
    seen = set()
    
    for x in sln:
        if _is_delim(x):
            cur_nurse = abs(int(x))
            if cur_nurse not in routes:
                routes[cur_nurse] = []
        else:
            p = int(x)
            if p in pats and p not in seen:
                if cur_nurse is None:
                    cur_nurse = nurses[0]
                routes[cur_nurse].append(p)
                seen.add(p)
    
    # Add missing patients to least-loaded nurse
    for p in pats:
        if p not in seen:
            min_nurse = min(routes.keys(), key=lambda k: len(routes[k]))
            routes[min_nurse].append(p)
    
    # Rebuild solution
    out = []
    for nid in nurses:
        if routes[nid]:  # only add delimiter if nurse has patients
            out.append(-int(nid))
            out.extend(routes[nid])
    
    return out

# =========================
# STRICT MODEL-COMPLIANT EVALUATION
# =========================

TRAVEL_COST_RATE = 1.0
OVERTIME_RATE_FALLBACK = DN_PER_TIME  # global fallback if nurse-specific not given

COL_DN_OVERTIME = "Overtime Cost"   # optional: d_n per nurse
COL_VEHICLE_USE = "Vehicle"         # optional: v_n flag

def get_overtime_rate(df: pd.DataFrame, nurse_id: int) -> float:
    """Return d_n: per-unit overtime cost for nurse n."""
    if COL_DN_OVERTIME in df.columns:
        s = df.loc[df[COL_NURSE_ID] == nurse_id, COL_DN_OVERTIME].dropna()
        if len(s):
            try:
                return float(s.iloc[0])
            except:
                pass
    return float(OVERTIME_RATE_FALLBACK)

def decode_routes(sln: list[int], df: pd.DataFrame) -> dict[int, list[int]]:
    """Decode solution vector into nurse→route dictionary (unique patients)."""
    nurses = nurse_ids_list(df)
    pats   = patient_list(df)

    routes = {nid: [] for nid in nurses}
    cur = None
    for x in sln:
        if _is_delim(x):  # new nurse block
            cur = abs(int(x))
            if cur not in routes:
                routes[cur] = []
        else:
            p = int(x)
            if cur is None:  # assign to least-loaded if no nurse yet
                cur = min(routes.keys(), key=lambda k: len(routes[k]))
            routes[cur].append(p)

    # make assignments unique (keep first occurrence)
    used = set()
    for nid in nurses:
        new_seq = []
        for p in routes[nid]:
            if p not in used and p in pats:
                new_seq.append(p)
                used.add(p)
        routes[nid] = new_seq

    # add missing patients to least-loaded nurse
    for p in pats:
        if p not in used:
            nid = min(routes.keys(), key=lambda k: len(routes[k]))
            routes[nid].append(p)
            used.add(p)

    return routes

# =============================
# EVALUATION FUNCTIONS
# =============================

DISTANCE_TO_TIME_FACTOR = 1.0  # ← Adjust this
TRAVEL_COST_PER_UNIT = 2.0   # cost per travel minute
OVERTIME_COST_RATE = 2.0 # d_n per overtime minute

# =========================
# VARIATION OPERATORS (reaction kernels only)
# =========================
class Ops:
    r = random.Random()
   
    @staticmethod
    def _patient_positions(sln: List[int]) -> List[int]:
        return [i for i, v in enumerate(sln) if not (isinstance(v, (int,float)) and v < 0)]

    @staticmethod
    def swap_mutation(sln: List[int]) -> None:
        idx = Ops._patient_positions(sln)
        if len(idx) >= 2:
            i1, i2 = Ops.r.sample(idx, 2)
            sln[i1], sln[i2] = sln[i2], sln[i1]

    @staticmethod
    def insert_mutation(sln: List[int]) -> None:
        idx = Ops._patient_positions(sln)
        if len(idx) >= 2:
            i1 = Ops.r.choice(idx)
            idx2 = [j for j in idx if j != i1]
            if not idx2: return
            i2 = Ops.r.choice(idx2)
            val = sln[i1]
            sln.pop(i1)
            insert_pos = i2 if i2 < i1 else i2 - 1
            sln.insert(insert_pos, val)

    @staticmethod
    def inversion_mutation(sln: List[int]) -> None:
        idx = Ops._patient_positions(sln)
        if len(idx) >= 2:
            i1, i2 = sorted(Ops.r.sample(idx, 2))
            sln[i1:i2+1] = sln[i1:i2+1][::-1]

    @staticmethod
    def recombine_and_scramble(p1: List[int], p2: List[int]) -> List[int]:
        if not p1: return p2[:]
        if not p2: return p1[:]
        c1 = Ops.r.randint(1, max(1, len(p1)-1))
        c2 = Ops.r.randint(1, max(1, len(p2)-1))
        child = p1[:c1] + p2[c2:]

        idx = Ops._patient_positions(child)
        if len(idx) >= 3:
            k = Ops.r.randint(2, min(4, len(idx)))
            subset = Ops.r.sample(idx, k)
            vals = [child[i] for i in subset]
            Ops.r.shuffle(vals)
            for i, v in zip(subset, vals):
                child[i] = v
        return child
# =========================
# Q-NDSA + CROWDING + PE
# =========================
def q_ndsa_rank(pop_objs: List[List[float]]) -> List[int]:
    """
    pop_objs: list of [f1,f2,f3] in MIN form (we will pass [-F3] as + for min).
    Returns integer ranks starting at 1.
    """
    N = len(pop_objs)
    idx = list(range(N))
    # sort by f1
    idx.sort(key=lambda i: pop_objs[i][0])
    ranks = [None]*N
    remaining = idx[:]
    current = 1

    while remaining:
        W = float("inf")
        F: List[int] = []
        keep: List[int] = []
        for i in remaining:
            f2 = pop_objs[i][1]
            if f2 < W:
                W = f2
                ranks[i] = current
                F.append(i)
            else:
                keep.append(i)

        if keep:
            # correction by f3
            E = sorted(F, key=lambda i: pop_objs[i][2])
            new_E = E[:]
            for s in keep[:]:
                f1s, f2s, f3s = pop_objs[s]
                # binary search by f3
                lo, hi = 0, len(new_E)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if pop_objs[new_E[mid]][2] < f3s:
                        lo = mid + 1
                    else:
                        hi = mid
                left = max(0, lo-3); right = min(len(new_E), lo+3)
                PDS = new_E[left:right]

                dominated = False
                for j in PDS:
                    f1j, f2j, f3j = pop_objs[j]
                    if (f1j <= f1s and f2j <= f2s and f3j <= f3s) and (f1j < f1s or f2j < f2s or f3j < f3s):
                        dominated = True
                        break
                if not dominated:
                    ranks[s] = current
                    ins = 0
                    while ins < len(new_E) and pop_objs[new_E[ins]][2] <= f3s:
                        ins += 1
                    new_E.insert(ins, s)

            remaining = [i for i in remaining if ranks[i] is None]
        else:
            remaining = [i for i in remaining if ranks[i] is None]

        current += 1

    return [int(r) for r in ranks]

def crowding_distance(front_indices: List[int], objs: List[List[float]]) -> Dict[int, float]:
    if not front_indices:
        return {}
    if len(front_indices) <= 2:
        return {idx: float('inf') for idx in front_indices}

    distances = {idx: 0.0 for idx in front_indices}
    M = len(objs[0])

    for m in range(M):
        sorted_idx = sorted(front_indices, key=lambda i: objs[i][m])
        f_min = objs[sorted_idx[0]][m]
        f_max = objs[sorted_idx[-1]][m]

        distances[sorted_idx[0]] = float('inf')
        distances[sorted_idx[-1]] = float('inf')

        denom = (f_max - f_min)
        if denom == 0:
            continue

        for k in range(1, len(sorted_idx) - 1):
            prev_val = objs[sorted_idx[k - 1]][m]
            next_val = objs[sorted_idx[k + 1]][m]
            distances[sorted_idx[k]] += (next_val - prev_val) / denom

    return distances

# =========================
# MOLECULE
# =========================
class Molecule:
    __slots__ = ("structure","obj","PE","KE","NumHit","MinHit","type","parents","children","Rank","CD","id_tag")
    def __init__(self, structure: List[int], obj: Tuple[float,float,float], KE: float,
                 type_: str = "", parents: Tuple[Optional[int],Optional[int]] = (None,None),
                 children: Tuple[Optional[int],Optional[int]] = (None,None)):
        self.structure = structure[:]
        self.obj       = (float(obj[0]), float(obj[1]), float(obj[2]))  # (F1,F2,F3)
        self.PE        = 0.0
        self.KE        = float(KE)
        self.NumHit    = 0
        self.MinHit    = 0
        self.type      = type_     # "wall" | "dec" | "inter" | "syn" | ""
        self.parents   = parents   # indices in parent list (logical), used inside one iteration
        self.children  = children  # sibling references (logical)
        self.Rank      = 0
        self.CD        = 0.0
        self.id_tag    = None      # optional for debugging

# =========================
# REACTIONS (generate ONLY)
# =========================
def op_wall(p: Molecule, df: pd.DataFrame) -> Molecule:
    child_struct = p.structure[:]
    Ops.swap_mutation(child_struct)
    # ✅ normalize structure before evaluating
    child_struct = normalize_solution(child_struct, df)
    f1, f2, f3 = evaluate_solution_CORRECTED(child_struct, df)
    return Molecule(child_struct, (f1, f2, f3), KE=p.KE, type_="wall")

def op_dec(p: Molecule, df: pd.DataFrame) -> Tuple[Molecule, Molecule]:
    mid = max(1, len(p.structure) // 2)
    s1, s2 = p.structure[:mid], p.structure[mid:]

    Ops.insert_mutation(s1)
    Ops.insert_mutation(s2)

    # ✅ Normalize both
    s1 = normalize_solution(s1, df)
    s2 = normalize_solution(s2, df)

    f1, f2, f3 = evaluate_solution_CORRECTED(s1, df)
    c1 = Molecule(s1, (f1, f2, f3), KE=p.KE, type_="dec")
    f1, f2, f3 = evaluate_solution_CORRECTED(s2, df)
    c2 = Molecule(s2, (f1, f2, f3), KE=p.KE, type_="dec")
    
    return c1, c2


def op_inter(p1: Molecule, p2: Molecule, df: pd.DataFrame) -> Tuple[Molecule, Molecule]:
    s1, s2 = p1.structure[:], p2.structure[:]

    Ops.inversion_mutation(s1)
    Ops.inversion_mutation(s2)

    # ✅ Normalize both
    s1 = normalize_solution(s1, df)
    s2 = normalize_solution(s2, df)

    f1, f2, f3 = evaluate_solution_CORRECTED(s1, df)
    c1 = Molecule(s1, (f1, f2, f3), KE=p1.KE, type_="inter")
    f1, f2, f3 = evaluate_solution_CORRECTED(s2, df)
    c2 = Molecule(s2, (f1, f2, f3), KE=p2.KE, type_="inter")

    return c1, c2


def op_syn(p1: Molecule, p2: Molecule, df: pd.DataFrame) -> Molecule:
    s = Ops.recombine_and_scramble(p1.structure, p2.structure)

    # ✅ Normalize combined structure
    s = normalize_solution(s, df)

    f1, f2, f3 = evaluate_solution_CORRECTED(s, df)
    c = Molecule(s, (f1, f2, f3), KE=p1.KE + p2.KE, type_="syn")

    return c

# =========================
# HYPERVOLUME
# =========================
class MetricsUtil:
    """Faithful Python adaptation of Mme Abir’s MetricsUtil."""
    @staticmethod
    def get_maximum_values(front: np.ndarray, no_objectives: int) -> np.ndarray:
        return np.max(front[:, :no_objectives], axis=0)

    @staticmethod
    def get_minimum_values(front: np.ndarray, no_objectives: int) -> np.ndarray:
        return np.min(front[:, :no_objectives], axis=0)

    @staticmethod
    def get_normalized_front(front: np.ndarray,
                             max_values: np.ndarray,
                             min_values: np.ndarray) -> np.ndarray:
        norm = (front - min_values) / (max_values - min_values + 1e-12)
        return np.clip(norm, 0.0, 1.0)

    @staticmethod
    def inverted_front(front: np.ndarray) -> np.ndarray:
        # for minimization problems → invert values
        return 1.0 - front
    
class Hypervolume:
    """Faithful translation of Mme Abir’s Hypervolume.java implementation."""
    def __init__(self):
        self.utils_ = MetricsUtil()

    # --- same dominate relation ---
    def dominates(self, p1: np.ndarray, p2: np.ndarray, nobj: int) -> bool:
        better = False
        for i in range(nobj):
            if p1[i] > p2[i]:
                return False
            elif p1[i] < p2[i]:
                better = True
        return better

    # --- swap two points ---
    def swap(self, front, i, j):
        front[[i, j]] = front[[j, i]]

    # --- filter nondominated points ---
    def filter_nondominated_set(self, front, n_points, n_obj):
        n = n_points
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                if self.dominates(front[i], front[j], n_obj):
                    n -= 1
                    self.swap(front, j, n)
                elif self.dominates(front[j], front[i], n_obj):
                    n -= 1
                    self.swap(front, i, n)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1
        return n

    # --- smallest value on a given objective ---
    def surface_unchanged_to(self, front, n_points, obj):
        return np.min(front[:n_points, obj])

    # --- remove points <= threshold on given objective ---
    def reduce_nondominated_set(self, front, n_points, obj, threshold):
        n = n_points
        i = 0
        while i < n:
            if front[i, obj] <= threshold:
                n -= 1
                self.swap(front, i, n)
            else:
                i += 1
        return n

    # --- recursive HV calculation ---
    def calculate_hypervolume(self, front: np.ndarray, n_points: int, n_obj: int) -> float:
        volume = 0.0
        distance = 0.0
        n = n_points
        while n > 0:
            n_nd = self.filter_nondominated_set(front, n, n_obj - 1)
            temp_volume = (front[0, 0] if n_obj < 3 else
                           self.calculate_hypervolume(front, n_nd, n_obj - 1))
            temp_distance = self.surface_unchanged_to(front, n, n_obj - 1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = self.reduce_nondominated_set(front, n, n_obj - 1, distance)
        return volume

    # --- main entry point (same as Java hypervolume()) ---
    def hypervolume(self, pareto_front: np.ndarray,
                    pareto_true_front: np.ndarray,
                    number_of_objectives: int) -> float:
        max_vals = self.utils_.get_maximum_values(pareto_true_front, number_of_objectives)
        min_vals = self.utils_.get_minimum_values(pareto_true_front, number_of_objectives)
        norm_front = self.utils_.get_normalized_front(pareto_front, max_vals, min_vals)
        inv_front = self.utils_.inverted_front(norm_front)
        return self.calculate_hypervolume(inv_front.copy(), len(inv_front), number_of_objectives)

#========#

def read_instance(file_path: str, num_patients: int) -> pd.DataFrame:
    """
    Load a single HHC instance CSV file, clean its columns, and sample up to num_patients.
    Returns a standardized DataFrame with numeric NurseID and Patient columns.
    """

    # Try comma first, fallback to semicolon
    try:
        df = pd.read_csv(file_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=";")
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV file {file_path}: {e}")

    # Clean up column names
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)

    # Verify required columns
    required = ["Nurse", "Patient", "Patient Location"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in {file_path}. Found: {df.columns.tolist()}")

    # Create numeric NurseID
    df["NurseID"] = df["Nurse"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    df["Patient"] = pd.to_numeric(df["Patient"], errors="coerce").astype("Int64")

    # Randomly sample patients if needed
    patients = df["Patient"].dropna().drop_duplicates()
    if len(patients) < num_patients:
        num_patients = len(patients)

    sampled = patients.sample(n=num_patients, random_state=random.randint(1, 9999)).tolist()
    df = df[df["Patient"].isin(sampled)].reset_index(drop=True)

    return df

def fitness(m: Molecule) -> float:
    """Composite minimization fitness (lower = better)."""
    f1, f2, f3 = m.obj
    return f1 + f2 - f3
# =========================
# NCRO LOOP 
# =========================
def run_ncro_one_file(csv_path: Path, pat_count: int) -> pd.DataFrame:
    df = read_instance(str(csv_path), num_patients=pat_count)
    num_nurses = detect_num_nurses(df)

    # ---- Initial population ----
    pop: List[Molecule] = []
    for _ in range(POP_SIZE):
        sol = generate_solution(df, num_nurses)
        f1, f2, f3 = evaluate_solution_CORRECTED(sol, df, csv_path.name)
        pop.append(Molecule(sol, (f1, f2, f3), INITIAL_KE))

    # compute initial PE = rank + exp(-CD) on pop
    assign_rank_cd_pe(pop)

    enBuff = float(BUFFER_INIT)
    FE = len(pop)
    it = 0

    # === Q-LEARNING PARAMETERS (Operator Selection) ===
    ACTIONS = ["wall", "dec", "inter", "syn"]          # available CRO reactions
    Q_table = np.zeros((len(ACTIONS), len(ACTIONS)))   # state = previous action
    QL_ALPHA, QL_GAMMA, QL_EPSILON = 0.4, 0.9, 0.4
    prev_action = 0  # start assuming 'wall' was last operator
    op_rewards = {a: [] for a in ACTIONS}
# --- Tag children with their operator for tracking ---
    def compute_reward(parents, children, action_label=""):
        """
        Reward based on Potential Energy (PE) improvement.
        Reward = max(0, mean(PE_parents) - mean(PE_children))
        (Lower PE means better solutions.)
        """
        if not parents or not children:
            return 0.0

        try:
            pe_parents = np.mean([p.PE for p in parents])
            pe_children = np.mean([c.PE for c in children])
            delta_pe = pe_children - pe_parents
            reward = max(0.0, -delta_pe)  # positive only when children improved

        except Exception as e:
            print(f"⚠️ PE reward computation error ({action_label}): {e}")
            reward = 0.0

        return reward



 # === Q-LEARNING MAIN LOOP (one operator per iteration + Q-table tracking) ===    
    while it < MAX_ITERS and FE < FELIMIT:
        it += 1

        # === Operator selection purely by Q-Learning (no energy condition) ===
        state = prev_action
        if random.random() < QL_EPSILON:
            action_idx = random.randint(0, len(ACTIONS) - 1)
        else:
            action_idx = int(np.argmax(Q_table[state]))
        action = ACTIONS[action_idx]

        parents_snapshot = pop[:]
        Q = []

        # === Generate offspring population purely based on Q-learning decision ===
        for _ in range(POP_SIZE):
            if random.random() < MOLE_COLL and len(pop) >= 2:
                # --- Bimolecular reactions ---
                p1, p2 = random.sample(pop, 2)

                if action == "syn":
                    child = op_syn(p1, p2, df)
                    child.parents = (pop.index(p1), pop.index(p2))
                    Q.append(child)

                elif action == "inter":
                    c1, c2 = op_inter(p1, p2, df)
                    c1.parents = (pop.index(p1), pop.index(p2))
                    c2.parents = (pop.index(p1), pop.index(p2))
                    Q.extend([c1, c2])

                elif action == "dec":
                    base = random.choice([p1, p2])
                    c1, c2 = op_dec(base, df)
                    c1.parents = (pop.index(base), None)
                    c2.parents = (pop.index(base), None)
                    Q.extend([c1, c2])

                else:  # "wall"
                    base = random.choice([p1, p2])
                    c = op_wall(base, df)
                    c.parents = (pop.index(base), None)
                    Q.append(c)

            else:
                # --- Unimolecular reactions ---
                p = random.choice(pop)

                if action == "dec":
                    c1, c2 = op_dec(p, df)
                    c1.parents = (pop.index(p), None)
                    c2.parents = (pop.index(p), None)
                    Q.extend([c1, c2])

                elif action == "inter" and len(pop) >= 2:
                    # pair with a different random parent
                    p2 = random.choice([q for q in pop if q is not p])
                    c1, c2 = op_inter(p, p2, df)
                    c1.parents = (pop.index(p), pop.index(p2))
                    c2.parents = (pop.index(p), pop.index(p2))
                    Q.extend([c1, c2])

                elif action == "syn" and len(pop) >= 2:
                    p2 = random.choice([q for q in pop if q is not p])
                    c = op_syn(p, p2, df)
                    c.parents = (pop.index(p), pop.index(p2))
                    Q.append(c)

                else:  # "wall" or fallback
                    c = op_wall(p, df)
                    c.parents = (pop.index(p), None)
                    Q.append(c)

            if len(Q) >= POP_SIZE:
                Q = Q[:POP_SIZE]
                break

        FE += len(Q)

        # === Stage 2: Pareto evaluation and survival (no physics gating) ===
        pool = pop + Q
        assign_rank_cd_pe(pool)

        # 4) Energy gating (NCRO physics)
        SurvivorsQ = []
        removed_parents = set()
        q_iter = 0
        while q_iter < len(Q):
                child = Q[q_iter]
                typ = child.type
                pA, pB = child.parents
                p1 = pop[pA] if pA is not None else None
                p2 = pop[pB] if (pB is not None and pB < len(pop)) else None
    
                if typ == "wall":
                    tempBuff = p1.PE + p1.KE - child.PE
                    if tempBuff >= 0:
                        frac = random.random() * (1.0 - KE_LOSS_RATE) + KE_LOSS_RATE
                        child.KE = tempBuff * frac
                        leak = tempBuff - child.KE
                        enBuff += max(0.0, leak)
                        child.MinHit = p1.MinHit
                        child.NumHit = p1.NumHit + 1
                        SurvivorsQ.append(child)
                        removed_parents.add(pA)
                    q_iter += 1
    
                elif typ == "dec":
                    if q_iter + 1 >= len(Q) or Q[q_iter + 1].type != "dec" or Q[q_iter + 1].parents[0] != pA:
                        tempBuff = p1.PE + p1.KE - child.PE
                        if tempBuff >= 0:
                            frac = random.random() * (1.0 - KE_LOSS_RATE) + KE_LOSS_RATE
                            child.KE = tempBuff * frac
                            enBuff += max(0.0, tempBuff - child.KE)
                            SurvivorsQ.append(child)
                            removed_parents.add(pA)
                        q_iter += 1
                    else:
                        child2 = Q[q_iter + 1]
                        pe_kids = child.PE + child2.PE
                        tempBuff = p1.PE + p1.KE - pe_kids
                        if (tempBuff >= 0) or (tempBuff + enBuff >= 0):
                            if tempBuff < 0:
                                enBuff += tempBuff
                                tempBuff = 0.0
                            a = random.random()
                            child.KE = tempBuff * a
                            child2.KE = tempBuff * (1.0 - a)
                            child.MinHit = 0
                            child.NumHit = 0
                            child2.MinHit = 0
                            child2.NumHit = 0
                            SurvivorsQ.extend([child, child2])
                            removed_parents.add(pA)
                        q_iter += 2
    
                elif typ == "inter":
                    if q_iter + 1 >= len(Q) or Q[q_iter + 1].type != "inter" or Q[q_iter + 1].parents != (pA, pB):
                        sum_par = p1.PE + p1.KE + (p2.PE + p2.KE if p2 else 0.0)
                        tempBuff = sum_par - child.PE
                        if tempBuff >= 0:
                            child.KE = tempBuff
                            SurvivorsQ.append(child)
                            removed_parents.add(pA)
                            if pB is not None:
                                removed_parents.add(pB)
                        q_iter += 1
                    else:
                        child2 = Q[q_iter + 1]
                        pe_kids = child.PE + child2.PE
                        sum_par = p1.PE + p1.KE + p2.PE + p2.KE
                        tempBuff = sum_par - pe_kids
                        if tempBuff >= 0:
                            a = random.random()
                            child.KE = tempBuff * a
                            child2.KE = tempBuff * (1.0 - a)
                            child.MinHit = p1.MinHit
                            child2.MinHit = p2.MinHit
                            SurvivorsQ.extend([child, child2])
                            removed_parents.add(pA)
                            removed_parents.add(pB)
                        q_iter += 2
    
                elif typ == "syn":
                    sum_par = p1.PE + p1.KE + (p2.PE + p2.KE if p2 else 0.0)
                    tempBuff = sum_par - child.PE
                    if tempBuff >= 0:
                        child.KE = tempBuff
                        child.MinHit = 0
                        child.NumHit = 0
                        SurvivorsQ.append(child)
                        removed_parents.add(pA)
                        if pB is not None:
                            removed_parents.add(pB)
                    q_iter += 1
                else:
                    q_iter += 1
        # 5) Merge full parent + children, not only survivors
        pop_extended = pop + Q
        assign_rank_cd_pe(pop_extended)
        pop = sorted(pool, key=lambda m: (m.Rank, -m.CD))[:POP_SIZE]
        # ✅ Compute reward using PE difference (children - parents)
        reward = compute_reward(parents_snapshot, Q, action_label=action)
        op_rewards[action].append(reward)
        # --- Q-learning update (temporal) ---
        next_state = action_idx
        Q_table[state, action_idx] += QL_ALPHA * (
            reward + QL_GAMMA * float(np.max(Q_table[next_state])) - Q_table[state, action_idx]
        )
        # --- Move forward in the temporal chain ---
        prev_action = next_state
        QL_EPSILON = max(0.01, QL_EPSILON * 0.995)

        # --- Simple progress log ---
        if it % 10 == 0:
           mean_fit = np.mean([fitness(m) for m in pop])
           print(f"Iter {it:03d} | action={action:<5} | reward={reward:+.3f} "
                 f"| mean_fit={mean_fit:.3f} | ε={QL_EPSILON:.3f}")


    # ===== Final reporting =====
    rows = []
    for m in pop:
        F1, F2, F3 = m.obj
        rows.append({
            "Solution": m.structure,
            "F1": float(F1),
            "F2": float(F2),
            "F3": float(F3),
            "Rank": int(m.Rank),
            "PE": float(m.PE),
        })

    df_final = pd.DataFrame(rows)

    # ===== Final HV (Mme Abir adaptation) =====
    try:
        pareto_front = df_final.loc[df_final["Rank"] == 1, ["F1", "F2", "F3"]].to_numpy()
        if len(pareto_front) <= 1:
            print(f"⚠️ Pareto front too small for HV ({len(pareto_front)} point). HV = 0.")
            HV_value = 0.0
        else:
            ref_front = df_final[["F1", "F2", "F3"]].to_numpy()
            pareto_min = np.array([[f1, f2, -f3] for f1, f2, f3 in pareto_front])
            ref_min = np.array([[f1, f2, -f3] for f1, f2, f3 in ref_front])
            hv_calc = Hypervolume()
            HV_value = float(np.clip(hv_calc.hypervolume(pareto_min, ref_min, 3), 0.0, 1.0))
        print(f"   └─ HV for {csv_path.name}: {HV_value:.4f}")
    except Exception as e:
        print(f"⚠️ Hypervolume computation error for {csv_path.name}: {e}")
        HV_value = np.nan

    df_final["HV"] = HV_value
    return df_final

def assign_rank_cd_pe(pool: List[Molecule]) -> None:
    """
    Compute ranks, crowding and set PE = rank + exp(-CD) over 'pool'.
    For ranking we minimize [F1, F2, -F3] -> build objs_min accordingly.
    """
    if not pool:
        return
    objs_min = [[m.obj[0], m.obj[1], -m.obj[2]] for m in pool]
    ranks = q_ndsa_rank(objs_min)

    # gather fronts
    fronts: Dict[int, List[int]] = {}
    for i, r in enumerate(ranks):
        pool[i].Rank = r
        fronts.setdefault(r, []).append(i)

    # crowding per front
    CD = np.zeros(len(pool))
    for f_indices in fronts.values():
        cd_map = crowding_distance(f_indices, objs_min)
        for idx, cd in cd_map.items():
            CD[idx] = cd

    # set CD and PE
    for i, m in enumerate(pool):
        m.CD = float(CD[i])
        m.PE = float(m.Rank + math.exp(-max(0.0, m.CD)))
# =========================
# SMART 5-INSTANCE TEST EXECUTION
# =========================
NUM_RUNS = 10   # ← Change to 10 if you want 10 runs
CHECKPOINT_FILE = Path(OUTPUT_FOLDER) / "ncro_checkpoint.json"

def save_checkpoint(run_id, file_name, completed=False):
    """Save progress state for resume after crash/power loss."""
    data = {
        "run_id": run_id,
        "file_name": file_name,
        "completed": completed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return None

def clear_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    all_files = sorted(Path(INPUT_FOLDER).glob("*.csv"))

    # 🔍 Run only the first instance (e.g., c101)
    all_files = all_files[:1]  # this keeps only the first CSV file found

    if not all_files:
        print("⚠️ No CSV files found in INPUT_FOLDER.")
        raise SystemExit

    print(f"Found {len(all_files)} instance files to process.\n")

    checkpoint = load_checkpoint()
    start_run = 1
    start_file_idx = 0

    # === Resume point detection ===
    if checkpoint:
        print(f"🔄 Resuming from checkpoint: {checkpoint}")
        start_run = checkpoint.get("run_id", 1)
        last_file = checkpoint.get("file_name", None)
        if last_file:
            for i, f in enumerate(all_files):
                if f.name == last_file:
                    start_file_idx = i
                    break
    else:
        print("🚀 Starting fresh multi-run experiment.")

    # === Main loop ===
    for run_id in range(start_run, NUM_RUNS + 1):
        print(f"\n===============================")
        print(f"🚀 Starting NCRO Run {run_id} / {NUM_RUNS}")
        print(f"===============================\n")

        # Per-run output folder
        run_folder = Path(OUTPUT_FOLDER) / f"Run_{run_id:02d}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Controlled randomness per run
        random.seed(42 + run_id * 111)
        np.random.seed(42 + run_id * 111)

        # Continue from the correct file if resuming
        for fi, f in enumerate(all_files[start_file_idx:], start=start_file_idx):
            try:
                out_path = run_folder / f"{f.stem}_Run{run_id:02d}_ncro_results.csv"
                tmp_path = out_path.with_suffix(".tmp")

                # Skip files already completed
                if out_path.exists():
                    print(f"⏭️ Run {run_id}: Skipping {f.name} (already done).")
                    continue

                print(f"⚙️  Run {run_id}: Processing {f.name} ...")
                save_checkpoint(run_id, f.name, completed=False)

                # === Run NCRO optimization ===
                df_out = run_ncro_one_file(f, pat_count=25)

                # === Safe write (atomic) ===
                df_out.to_csv(tmp_path, sep=",", float_format="%.6f", index=False)
                tmp_path.replace(out_path)

                hv_value = float(df_out["HV"].iloc[0]) if "HV" in df_out.columns else float("nan")
                print(f"✅ Run {run_id}: saved {out_path.name} | HV = {hv_value:.6f}\n")

                save_checkpoint(run_id, f.name, completed=True)

            except Exception as e:
                print(f"❌ Run {run_id}: Error processing {f.name}: {e}")
                traceback.print_exc()
                save_checkpoint(run_id, f.name, completed=False)
                print("💾 Progress saved. You can safely restart later.\n")
                break  # Pause execution safely on error

        # After finishing this run, clear checkpoint
        clear_checkpoint()
        start_file_idx = 0  # reset for next run

    print("\n🎯 All runs completed successfully.")
    print(f"📁 Results saved under: {OUTPUT_FOLDER}")
