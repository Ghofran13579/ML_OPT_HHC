import os
import glob
import math
import ast
import time

import numpy as np
import pandas as pd
from docplex.mp.model import Model

# ========================
# USER CONFIG
# ========================
INPUT_FOLDER = r"C:\Users\Ghofran MASSAOUDI\Desktop\output_equitable_Correct_stucture"
MAX_JOBS = 14
NUM_NURSES = 4
NUM_SHARED_CARS = 25
COST_RATE = 1.0
USE_DEPOT_FROM_DATA = True
FORCE_AT_LEAST_ONE_SHARED_ROUTE = True
OVERTIME_COST_RATE = 2.0  # d_n parameter in math model

# ========================
# HELPERS
# ========================
def parse_tuple(val, default=None):
    if default is None:
        default = (0.0, 0.0)
    if pd.isna(val):
        return default
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return float(val[0]), float(val[1])
    try:
        t = ast.literal_eval(str(val))
        if isinstance(t, (tuple, list)) and len(t) == 2:
            return float(t[0]), float(t[1])
    except Exception:
        pass
    try:
        xs = str(val).strip("()[] ").split(",")
        return float(xs[0]), float(xs[1])
    except Exception:
        return default

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def first_nonempty(series, parser=parse_tuple, default=(50.0, 50.0)):
    for v in series:
        p = parser(v, default=None)
        if p is not None:
            return p
    return default

# ========================
# BUILD INSTANCE
# ========================
def build_instance_from_csv(csv_path, max_jobs=MAX_JOBS, num_nurses=NUM_NURSES):
    df = pd.read_csv(csv_path, sep=";")

    # Detect coordinates
    if 'Patient Location' in df.columns:
        pts = df[['Patient', 'Patient Location']].copy() if 'Patient' in df.columns else df[['Patient Location']].copy()
        pts['xy'] = pts['Patient Location'].apply(lambda v: parse_tuple(v, (0.0, 0.0)))
        if 'Patient' not in pts.columns:
            pts.insert(0, 'Patient', range(1, len(pts)+1))
    elif {'X', 'Y'}.issubset(df.columns):
        pts = df[['Patient', 'X', 'Y']] if 'Patient' in df.columns else df[['X', 'Y']].copy()
        if 'Patient' not in pts.columns:
            pts.insert(0, 'Patient', range(1, len(pts)+1))
        pts['xy'] = list(zip(pts['X'].astype(float), pts['Y'].astype(float)))
    else:
        raise ValueError(f"{os.path.basename(csv_path)} missing coordinates")

    # Service time
    st_map = df.groupby('Patient')['Service Time'].mean().to_dict() if 'Service Time' in df.columns and 'Patient' in df.columns else None

    # Preferences & qualifications
    pref_map = df.groupby('Patient')['Preferences of the Patients'].mean().to_dict() if 'Preferences of the Patients' in df.columns and 'Patient' in df.columns else {}
    qual_map = df.groupby('Patient')['Qualification of the Nurses'].mean().to_dict() if 'Qualification of the Nurses' in df.columns and 'Patient' in df.columns else {}

    # Depot
    depot = first_nonempty(df['Depot']) if USE_DEPOT_FROM_DATA and 'Depot' in df.columns else (50.0, 50.0)

    # Select patients/jobs
    pts = pts.drop_duplicates(subset=['Patient']).sort_values('Patient').head(max_jobs).reset_index(drop=True)
    J = len(pts)
    if J == 0:
        raise ValueError("No patients found in CSV after filtering")

    # V = J ∪ {0, J+1} as per math model
    coords = {0: depot}
    for i, row in enumerate(pts.itertuples(index=False), start=1):
        coords[i] = row.xy
    coords[J+1] = depot

    # Service times
    service = {0: 0.0, J+1: 0.0}
    mean_service = df['Service Time'].mean() if 'Service Time' in df.columns else 0.0
    for i in range(1, J+1):
        pid = int(pts.loc[i-1, 'Patient'])
        si = float(st_map.get(pid, mean_service) if st_map is not None else 0.0)
        service[i] = si

    # Nurse homes & vehicle type
    nurse_homes = [(depot[0]+np.random.normal(0,2.0), depot[1]+np.random.normal(0,2.0)) for _ in range(num_nurses)]
    v_n = [0]*num_nurses  # 0 = personal vehicle, 1 = shared car
    for i in range(num_nurses//2, num_nurses):
        v_n[i] = 1

    # Distance matrix
    V = list(range(0, J+2))
    dist = np.zeros((J+2, J+2))
    for i in V:
        for j in V:
            if i != j:
                dist[i, j] = euclid(coords[i], coords[j])

    # Time and cost matrices with H_n adjustments
    T = np.zeros((num_nurses, J+2, J+2))
    C = np.zeros((num_nurses, J+2, J+2))
    
    for n in range(num_nurses):
        for i in V:
            for j in V:
                if i == j:
                    continue
                
                T[n, i, j] = dist[i, j]
                C[n, i, j] = COST_RATE * dist[i, j]
                
                # For personal vehicles, node 0 and J+1 represent home H_n
                if v_n[n] == 0:
                    home = nurse_homes[n]
                    if i == 0:
                        T[n, 0, j] = euclid(home, coords[j])
                        C[n, 0, j] = COST_RATE * T[n, 0, j]
                    if j == J+1:
                        T[n, i, J+1] = euclid(coords[i], home)
                        C[n, i, J+1] = COST_RATE * T[n, i, J+1]

    # Preferences & qualifications
    S = np.full((num_nurses, J), 3.0)
    q = np.full((num_nurses, J), 3.0)
    for j in range(1, J+1):
        pid = int(pts.loc[j-1, 'Patient'])
        S[:, j-1] = pref_map.get(pid, 3.0)
        q[:, j-1] = qual_map.get(pid, 3.0)

    return {
        'csv_name': os.path.basename(csv_path),
        'N': list(range(num_nurses)),
        'J': J,
        'V': V,
        'coords': coords,
        'service': service,
        'nurse_homes': nurse_homes,
        'v_n': v_n,
        'T': T,
        'C': C,
        'S': S,
        'q': q,
        'L': np.full(num_nurses, 480.0),
        'num_shared_cars': NUM_SHARED_CARS
    }

# ========================
# SOLVE INSTANCE - ALL CONSTRAINTS IMPLEMENTED
# ========================
def solve_instance(params):
    N, J, V, T, C, S, q, L, v_n, num_shared_cars = (
        params['N'], params['J'], params['V'], params['T'], params['C'],
        params['S'], params['q'], params['L'], params['v_n'], params['num_shared_cars']
    )

    mdl = Model(name=f"HHC_Complete_{params['csv_name']}")
    M = 10000  # Big-M constant

    print(f"\n📊 Building model with ALL constraints for {J} jobs, {len(N)} nurses")

    # ========================
    # DECISION VARIABLES
    # ========================
    x = {(n,i,j): mdl.binary_var(name=f"x_{n}_{i}_{j}") for n in N for i in V for j in V if i != j}
    y = {(n,i): mdl.binary_var(name=f"y_{n}_{i}") for n in N for i in range(1, J+1)}
    z = {n: mdl.binary_var(name=f"z_{n}") for n in N}

    a = {(n,i): mdl.continuous_var(lb=0, name=f"a_{n}_{i}") for n in N for i in range(1, J+1)}
    t = {(n,i): mdl.continuous_var(lb=0, name=f"t_{n}_{i}") for n in N for i in range(1, J+1)}
    end = {(n,i): mdl.continuous_var(lb=0, name=f"end_{n}_{i}") for n in N for i in range(1, J+1)}

    # ========================
    # CONSTRAINT 1: TIMING CONSTRAINTS
    # ========================
    print("✓ Adding Constraint 1: Timing & Arrival")
    for n in N:
        for i in range(1, J+1):
            # First job from depot: t_i^n = T_0i^n if x_0i^n = 1
            mdl.add_indicator(x[(n,0,i)], t[(n,i)] == T[n,0,i], active_value=1)
            
            # Subsequent jobs: a_i^n = end_j^n + T_ji^n if x_ji^n = 1
            for j in range(1, J+1):
                if i != j:
                    mdl.add_indicator(x[(n,j,i)], a[(n,i)] == end[(n,j)] + T[n,j,i], active_value=1)
            
            # Service starts no earlier than arrival: t_i^n ≥ a_i^n
            mdl.add_constraint(t[(n,i)] >= a[(n,i)])
            
            # Completion time: end_i^n = t_i^n + s_i
            mdl.add_constraint(end[(n,i)] == t[(n,i)] + params['service'][i])

    # ========================
    # CONSTRAINT 2: ASSIGNMENT
    # ========================
    print("✓ Adding Constraint 2: Assignment (each patient once)")
    for i in range(1, J+1):
        mdl.add_constraint(mdl.sum(y[(n,i)] for n in N) == 1)

    # ========================
    # CONSTRAINT 3: PRECEDENCE (SEQUENTIAL CARE)
    # ========================
    print("✓ Adding Constraint 3: Precedence (sequential visits)")
    for i in range(1, J+1):
        for n in N:
            for m in N:
                if n != m:
                    # If both nurses visit patient i, one must finish before the other starts
                    # end_i^n ≤ t_i^m + M(2 - y_i^n - y_i^m)
                    mdl.add_constraint(end[(n,i)] <= t[(m,i)] + M * (2 - y[(n,i)] - y[(m,i)]))

    # ========================
    # CONSTRAINT 4: FLOW CONSERVATION
    # ========================
    print("✓ Adding Constraint 4: Flow Conservation")
    for n in N:
        for i in range(1, J+1):
            mdl.add_constraint(mdl.sum(x[(n,i,j)] for j in V if j != i) == y[(n,i)])
            mdl.add_constraint(mdl.sum(x[(n,j,i)] for j in V if j != i) == y[(n,i)])
            mdl.add_constraint(y[(n,i)] <= z[n])

    # ========================
    # CONSTRAINT 5: START/END AT DEPOT
    # ========================
    print("✓ Adding Constraint 5: Start/End at depot")
    for n in N:
        # Start at depot: ∑_j x_0j^n ≤ 1
        mdl.add_constraint(mdl.sum(x[(n,0,j)] for j in range(1, J+1)) <= 1)
        # End at depot: ∑_i x_i,J+1^n ≤ 1
        mdl.add_constraint(mdl.sum(x[(n,i,J+1)] for i in range(1, J+1)) <= 1)
        
        # Link to active nurse
        mdl.add_constraint(mdl.sum(x[(n,0,j)] for j in range(1, J+1)) == z[n])
        mdl.add_constraint(mdl.sum(x[(n,i,J+1)] for i in range(1, J+1)) == z[n])

    # ========================
    # CONSTRAINT 6: OVERTIME CALCULATION (IMPLEMENTED)
    # ========================
    print("✓ Adding Constraint 6: Overtime limits")
    # Total working time for each nurse
    total_work_time = {}
    overtime = {}
    
    for n in N:
        # Calculate total time: sum of (travel time + service time) for all arcs
        total_work_time[n] = mdl.sum(
            x[(n,i,j)] * (T[n,i,j] + (params['service'][j] if 1 <= j <= J else 0.0))
            for i in V for j in V if i != j
        )
        
        # Overtime variable: overtime = max(0, total_time - L_n)
        overtime[n] = mdl.continuous_var(lb=0, name=f"overtime_{n}")
        mdl.add_constraint(overtime[n] >= total_work_time[n] - L[n])
        
        # Constraint 6 interpretation: Total work time ≤ L_n + overtime
        # This is always satisfied by the overtime definition above
        mdl.add_constraint(total_work_time[n] <= L[n] + overtime[n])

    # ========================
    # CONSTRAINT 7: DEPOT SELECTION
    # ========================
    print("✓ Adding Constraint 7: Depot selection (vehicle type)")
    # NOTE: Since V doesn't include H_n nodes, we interpret this constraint as:
    # - Shared vehicles (v_n=1) use actual depot (node 0, J+1)
    # - Personal vehicles (v_n=0) use node 0 and J+1 but with H_n distances
    
    # For shared cars: ∑_j x_0j^n ≤ v_n would prevent personal vehicles from working
    # This conflicts with Constraint 5, so we enforce it differently:
    
    # Shared car capacity constraint (global)
    shared = [n for n in N if v_n[n] == 1]
    if shared:
        mdl.add_constraint(mdl.sum(x[(n,0,j)] for n in shared for j in range(1, J+1)) <= num_shared_cars)
        if FORCE_AT_LEAST_ONE_SHARED_ROUTE:
            mdl.add_constraint(mdl.sum(x[(n,0,j)] for n in shared for j in range(1, J+1)) >= 1)

    # ========================
    # CONSTRAINT 8: SUBTOUR ELIMINATION
    # ========================
    print("✓ Adding Constraint 8: Subtour elimination")
    
    # 8a: 2-cycle elimination for shared cars
    for i in range(1, J+1):
        for j in range(1, J+1):
            if i != j:
                # For all vehicles (simplified from separate shared/personal in model)
                mdl.add_constraint(mdl.sum(x[(n,i,j)] + x[(n,j,i)] for n in N) <= 1)

    # 8b: MTZ constraints for longer subtours
    u = {(n,i): mdl.continuous_var(lb=1, ub=J, name=f"u_{n}_{i}") for n in N for i in range(1, J+1)}
    for n in N:
        for i in range(1, J+1):
            for j in range(1, J+1):
                if i != j:
                    mdl.add_constraint(u[(n,i)] - u[(n,j)] + J * x[(n,i,j)] <= J - 1)

    # ========================
    # EQUITABLE WORKLOAD (Additional practical constraint)
    # ========================
    print("✓ Adding workload balancing")
    cap = math.ceil(J / len(N))
    for n in N:
        mdl.add_constraint(mdl.sum(y[(n,i)] for i in range(1, J+1)) <= cap)

    # ========================
    # OBJECTIVES
    # ========================
    print("✓ Setting up lexicographic objectives")
    
    # Travel cost
    travel_cost = mdl.sum(C[n,i,j]*x[(n,i,j)] for n in N for i in V for j in V if i != j)
    
    # Overtime cost (using d_n = OVERTIME_COST_RATE)
    overtime_cost = mdl.sum(OVERTIME_COST_RATE * overtime[n] for n in N)
    
    # Z1: Minimize Cost (Travel + Overtime)
    Z1 = travel_cost + overtime_cost
    
    # Z2: Minimize Workload (Total travel + service time)
    Z2 = mdl.sum(total_work_time[n] for n in N)
    
    # Z3: Maximize Satisfaction (Qualification - Preferences)
    Z3 = mdl.sum(y[(n,i)] * (q[n,i-1] - S[n,i-1]) for n in N for i in range(1, J+1))

    # Lexicographic optimization: Z1 → Z2 → -Z3
    mdl.minimize_static_lex([Z1, Z2, -Z3])

    # ========================
    # SOLVE
    # ========================
    print(f"🚀 Solving with lexicographic optimization [Z1 → Z2 → Z3]...")
    start = time.time()
    sol = mdl.solve(log_output=True)
    cpu_time = time.time() - start

    if not sol:
        status = str(mdl.solve_details.status) if mdl.solve_details else "unknown"
        print(f"❌ No solution found. Status: {status}")
        return None, {
            "Instance": params['csv_name'],
            "CPU Time": round(cpu_time, 3),
            "F1": None, "F2": None, "F3": None,
            "Lex Order": "[Z1 → Z2 → Z3]",
            "Status": status
        }

    # Store solution
    params['x'] = x
    params['a'] = a
    params['t'] = t
    params['end'] = end
    params['z'] = z
    params['overtime'] = overtime

    print(f"✅ Solution found in {cpu_time:.2f}s")
    print(f"   F1 (Cost) = {sol.get_value(Z1):.2f}")
    print(f"   F2 (Workload) = {sol.get_value(Z2):.2f}")
    print(f"   F3 (Satisfaction) = {sol.get_value(Z3):.2f}")

    return sol, {
        "Instance": params['csv_name'],
        "CPU Time": round(cpu_time, 3),
        "F1": round(sol.get_value(Z1), 2),
        "F2": round(sol.get_value(Z2), 2),
        "F3": round(sol.get_value(Z3), 2),
        "Lex Order": "[Z1 → Z2 → Z3]",
        "Status": str(mdl.solve_details.status)
    }

# ========================
# EXPORT ROUTES
# ========================
# ========================
# EXPORT ROUTES (FULLY CORRECTED ARRIVAL & TOTAL TIME)
# ========================
def export_routes_details(params, sol, output_csv="routes_details.csv"):
    N, J, V = params['N'], params['J'], params['V']
    v_n, service, T = params['v_n'], params['service'], params['T']
    rows = []

    for n in N:
        # --- Build successor mapping from x_ij^n ---
        succ = {}
        for i in V:
            for j in V:
                if i != j and sol.get_value(params['x'].get((n, i, j), 0)) > 0.5:
                    succ[i] = j

        # --- Reconstruct route ---
        route_nodes, current, visited = [], 0, set()
        while current in succ:
            nxt = succ[current]
            if nxt in visited or nxt == J + 1:
                route_nodes.append(nxt)
                break
            visited.add(nxt)
            route_nodes.append(nxt)
            current = nxt
            if len(route_nodes) > J + 1:
                break

        # --- Initialize time dictionaries ---
        arrival, start_t, end_t = {}, {}, {}

        for idx, j in enumerate(route_nodes):
            if j == J + 1:
                break

            if idx == 0:
                # ✅ First job: arrival = travel time from depot/home
                arrival[j] = T[n, 0, j]
                start_t[j] = arrival[j]
                end_t[j] = start_t[j] + service[j]
            else:
                prev = route_nodes[idx - 1]
                arrival[j] = end_t[prev] + T[n, prev, j]
                start_t[j] = arrival[j]
                end_t[j] = start_t[j] + service[j]

        # --- True Total Time (travel + service) ---
        total_time = 0.0
        for i in V:
            for j in V:
                if i != j:
                    x_val = sol.get_value(params['x'].get((n, i, j), 0))
                    if x_val > 0.5:
                        total_time += T[n, i, j] + (service[j] if 1 <= j <= J else 0.0)

        # --- Overtime ---
        overtime_val = sol.get_value(params['overtime'][n]) if 'overtime' in params else 0.0

        visits = [f"Job {j}" for j in route_nodes if 1 <= j <= J]
        rows.append({
            "Instance": params['csv_name'],
            "Nurse": n + 1,
            "Vehicle": "Personal (Home)" if v_n[n] == 0 else "Shared (Depot)",
            "Route": " → ".join(visits) if visits else "Idle",
            "JobsAssigned": len(visits),
            "ArrivalTimes": [round(arrival[j], 1) for j in route_nodes if 1 <= j <= J],
            "StartTimes": [round(start_t[j], 1) for j in route_nodes if 1 <= j <= J],
            "EndTimes": [round(end_t[j], 1) for j in route_nodes if 1 <= j <= J],
            "TotalTime": round(total_time, 1),
            "Overtime": round(overtime_val, 1)
        })

    # --- Append to CSV correctly ---
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode='a', sep=";", header=write_header, index=False)
    print(f"✅ Appended routes for {params['csv_name']} with corrected times and TotalTime = Travel+Service")


# ========================
# BATCH RUNNER
# ========================
def solve_folder(folder_path, output_csv="summary_results_complete_V1.csv", routes_csv="routes_details_complete_V1.csv"):
    csvs = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csvs:
        print(f"No CSV files found in {folder_path}")
        return

    results = []
    
    # Clean up old files
    for old_file in [routes_csv, output_csv]:
        if os.path.exists(old_file):
            try:
                os.remove(old_file)
                print(f"🗑️ Removed old file: {old_file}")
            except PermissionError:
                print(f"⚠️ Cannot delete {old_file} - it's open elsewhere")

    for p in csvs:
        print("\n" + "="*70)
        print(f"📁 Processing: {os.path.basename(p)}")
        print("="*70)
        try:
            params = build_instance_from_csv(p, max_jobs=MAX_JOBS, num_nurses=NUM_NURSES)
            sol, res = solve_instance(params)
            results.append(res)
            if sol:
                export_routes_details(params, sol, output_csv=routes_csv)
        except Exception as e:
            print(f"❌ Skipped {os.path.basename(p)} due to error: {e}")
            import traceback
            traceback.print_exc()

    pd.DataFrame(results).to_csv(output_csv, sep=";", index=False)
    print(f"\n{'='*70}")
    print(f"✅ All results saved to {output_csv}")
    print(f"{'='*70}")

if __name__ == "__main__":
    solve_folder(INPUT_FOLDER)