import sys
import os
from itertools import product
from math import fabs, sqrt, pi

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import dearpygui.dearpygui as dpg

from smo import SingleServerSMO
from distributions import ExponentialDistribution, RayleighDistribution


# ==============================================================================
# ШРИФТ (твой)
# ==============================================================================

runs_per_combination = 50

def setup_custom_font():
    possible_paths = ["/usr/share/fonts/truetype/DejaVuSans.ttf"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with dpg.font_registry():
                    with dpg.font(path, 30) as main_font:
                        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
                        dpg.add_font_range(0x0300, 0x03ff)
                        dpg.bind_font(main_font)
                return
            except Exception as e:
                print(f"Ошибка загрузки шрифта: {e}")
    print("DejaVuSans не найден. Используется стандартный шрифт.")


# ==============================================================================
# ПФЭ + РЕГРЕССИИ
# ==============================================================================

def run_pfe(ranges, task_count):
    names = list(ranges.keys())

    def to_real(x_row):
        res = []
        for i, x in enumerate(x_row):
            a, b = ranges[names[i]]
            x0 = (a + b) / 2
            dx = (b - a) / 2
            res.append(x0 + x * dx)
        return res

    def simulate(l1, l2, sigma0, sigma1):
        gen1 = ExponentialDistribution(1 / l1)
        gen2 = ExponentialDistribution(1 / l2)
        serv0 = RayleighDistribution(sqrt(2 / (pi)) / sigma0)
        serv1 = RayleighDistribution(sqrt(2 / (pi)) / sigma1)

        smo = SingleServerSMO(gen1, gen2, serv0, task_count, serv1)
        smo.run()

        return (
            smo.avg_waiting_time_priority(0),
            smo.avg_waiting_time_priority(1)
        )

    base = list(product([-1, 1], repeat=4))

    def extend(row):
        x1, x2, x3, x4 = row
        return [1, x1, x2, x3, x4, x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4, x1*x2*x3, x1*x2*x4, x1*x3*x4, x2*x3*x4, x1*x2*x3*x4]

    matrix = [extend(r) for r in base]

    y0, y1 = [], []

    for row in base:
        l1, l2, s0, s1 = to_real(row)
        
        # Run multiple times and average results
        results0 = []
        results1 = []
        for _ in range(runs_per_combination):
            r0, r1 = simulate(l1, l2, s0, s1)
            results0.append(r0)
            results1.append(r1)
        
        # Average the results
        avg0 = sum(results0) / len(results0)
        avg1 = sum(results1) / len(results1)
        y0.append(avg0)
        y1.append(avg1)

    def calc_regression(y, ranges):
        N = len(matrix)

        b = [
            sum(matrix[i][j] * y[i] for i in range(N)) / N
            for j in range(len(matrix[0]))
        ]

        # Calculate x0 and dx for each factor for transformation
        names = list(ranges.keys())
        x0 = []  # Center values
        dx = []  # Half-ranges
        for name in names:
            a, b_val = ranges[name]
            x0.append((a + b_val) / 2)
            dx.append((b_val - a) / 2)

        def y_lin(row):
            return b[0] + b[1]*row[1] + b[2]*row[2] + b[3]*row[3] + b[4]*row[4]

        def y_full(row):
            return sum(b[j] * row[j] for j in range(len(row)))

        table = []

        for i in range(N):
            real = to_real(base[i])
            yl = y_lin(matrix[i])
            yf = y_full(matrix[i])

            table.append({
                "x": base[i],
                "real": real,
                "y": y[i],
                "y_lin": yl,
                "y_full": yf,
                "d_lin": fabs(y[i] - yl),
                "d_full": fabs(y[i] - yf)
            })

        terms = ["1","x1","x2","x3","x4","x1x2","x1x3","x1x4","x2x3","x2x4","x3x4","x1x2x3","x1x2x4","x1x3x4","x2x3x4","x1x2x3x4"]
        # Map term index to which variables it contains
        term_vars = [
            [],        # 0: 1
            [0],       # 1: x1
            [1],       # 2: x2
            [2],       # 3: x3
            [3],       # 4: x4
            [0,1],     # 5: x1x2
            [0,2],     # 6: x1x3
            [0,3],     # 7: x1x4
            [1,2],     # 8: x2x3
            [1,3],     # 9: x2x4
            [2,3],     # 10: x3x4
            [0,1,2],   # 11: x1x2x3
            [0,1,3],   # 12: x1x2x4
            [0,2,3],   # 13: x1x3x4
            [1,2,3],   # 14: x2x3x4
            [0,1,2,3]  # 15: x1x2x3x4
        ]
        var_names = ["λ1", "λ2", "μ1", "μ2"]

        eq_full = "y = " + " + ".join(f"{b[i]:.4f}*{terms[i]}" for i in range(len(b)))
        eq_lin = f"y = {b[0]:.4f} + {b[1]:.4f}x1 + {b[2]:.4f}x2 + {b[3]:.4f}x3 + {b[4]:.4f}x4"

        # Transform to natural coefficients
        # For linear equation: y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4
        # Substituting xi = x0_i + dx_i * x_norm_i:
        # y = b0 + b1*(x0_0+dx_0*x1) + b2*(x0_1+dx_1*x2) + b3*(x0_2+dx_2*x3) + b4*(x0_3+dx_3*x4)
        # y = (b0 + b1*x0_0 + b2*x0_1 + b3*x0_2 + b4*x0_3) + (b1*dx_0)*x1 + (b2*dx_1)*x2 + (b3*dx_2)*x3 + (b4*dx_3)*x4
        
        a0 = b[0] + b[1]*x0[0] + b[2]*x0[1] + b[3]*x0[2] + b[4]*x0[3]
        a1 = b[1] * dx[0]
        a2 = b[2] * dx[1]
        a3 = b[3] * dx[2]
        a4 = b[4] * dx[3]

        eq_lin_natural = f"y = {a0:.4f} + {a1:.4f}*λ1 + {a2:.4f}*λ2 + {a3:.4f}*μ1 + {a4:.4f}*μ2"

        # Transform full (non-linear) equation to natural coefficients
        # Each term is a product of certain x_i, we need to expand (x0_i + dx_i*x_i_n) for each
        # This creates a sum of terms: coefficient * product of natural variables
        # We'll collect coefficients for: constant, linear, quadratic (products of 2), cubic (products of 3), quartic
        
        from collections import defaultdict
        
        # Coefficient for each combination of variables (tuple of indices)
        nat_coeffs = defaultdict(float)
        
        # Process each term in the normalized regression
        for j in range(len(b)):
            coeff = b[j]
            vars_in_term = term_vars[j]
            
            if not vars_in_term:
                # Constant term: just add
                nat_coeffs[()] += coeff
            else:
                # Expand (x0_i + dx_i * x_i_n) for each variable in the term
                # We need to expand all combinations
                # Start with just the constant term
                expansions = {(): 1.0}
                
                for var_idx in vars_in_term:
                    new_expansions = {}
                    for exp_key, exp_coeff in expansions.items():
                        # Add x0_i term (no new variable)
                        new_expansions[exp_key] = new_expansions.get(exp_key, 0) + exp_coeff * x0[var_idx]
                        # Add dx_i * x_i term (add variable to product)
                        new_key = exp_key + (var_idx,)
                        new_expansions[new_key] = new_expansions.get(new_key, 0) + exp_coeff * dx[var_idx]
                    expansions = new_expansions
                
                # Add all expansions multiplied by the term's coefficient
                for exp_key, exp_coeff in expansions.items():
                    nat_coeffs[exp_key] += coeff * exp_coeff
        
        # Build the natural coefficient equation string
        # Sort by number of variables (constant first, then linear, etc.)
        sorted_terms = sorted(nat_coeffs.items(), key=lambda x: (len(x[0]), x[0]))
        
        eq_full_parts = []
        for var_tuple, coeff in sorted_terms:
            if abs(coeff) < 1e-10:
                continue
            
            if not var_tuple:
                # Constant
                eq_full_parts.append(f"{coeff:.4f}")
            else:
                # Product of variables
                var_str = "*".join(var_names[i] for i in var_tuple)
                eq_full_parts.append(f"{coeff:.4f}*{var_str}")
        
        eq_full_natural = "y = " + " + ".join(eq_full_parts)

        return table, eq_lin, eq_full, eq_lin_natural, eq_full_natural

    table0, eq_lin0, eq_full0, eq_lin_nat0, eq_full_nat0 = calc_regression(y0, ranges)
    table1, eq_lin1, eq_full1, eq_lin_nat1, eq_full_nat1 = calc_regression(y1, ranges)

    return (table0, eq_lin0, eq_full0, eq_lin_nat0, eq_full_nat0), (table1, eq_lin1, eq_full1, eq_lin_nat1, eq_full_nat1)


def run_dfe(ranges, task_count):
    names = list(ranges.keys())

    def to_real(x_row):
        res = []
        for i, x in enumerate(x_row):
            a, b = ranges[names[i]]
            x0 = (a + b) / 2
            dx = (b - a) / 2
            res.append(x0 + x * dx)
        return res

    def simulate(l1, l2, sigma0, sigma1):
        gen1 = ExponentialDistribution(1 / l1)
        gen2 = ExponentialDistribution(1 / l2)
        serv0 = RayleighDistribution(sqrt(2 / (pi)) / sigma0)
        serv1 = RayleighDistribution(sqrt(2 / (pi)) / sigma1)

        smo = SingleServerSMO(gen1, gen2, serv0, task_count, serv1)
        smo.run()

        return (
            smo.avg_waiting_time_priority(0),
            smo.avg_waiting_time_priority(1)
        )

    # --- ДФЭ: 3 независимых фактора ---
    base = []
    for x1, x2, x3 in product([-1, 1], repeat=3):
        x4 = x1 * x2 * x3  # генератор
        base.append([x1, x2, x3, x4])

    def extend(row):
        x1, x2, x3, x4 = row
        return [
            1, x1, x2, x3, x4,
            x1*x2, x1*x3, x1*x4,
            x2*x3, x2*x4, x3*x4
        ]

    matrix = [extend(r) for r in base]

    y0, y1 = [], []

    for row in base:
        l1, l2, s0, s1 = to_real(row)

        results0, results1 = [], []

        for _ in range(runs_per_combination):
            r0, r1 = simulate(l1, l2, s0, s1)
            results0.append(r0)
            results1.append(r1)

        y0.append(sum(results0)/len(results0))
        y1.append(sum(results1)/len(results1))

    def calc_regression(y, ranges):
        N = len(matrix)

        b = [
            sum(matrix[i][j] * y[i] for i in range(N)) / N
            for j in range(len(matrix[0]))
        ]

        names = list(ranges.keys())
        x0 = []
        dx = []

        for name in names:
            a, b_val = ranges[name]
            x0.append((a + b_val) / 2)
            dx.append((b_val - a) / 2)

        def y_lin(row):
            return b[0] + b[1]*row[1] + b[2]*row[2] + b[3]*row[3] + b[4]*row[4]

        def y_nl(row):
            return sum(b[j]*row[j] for j in range(len(row)))

        table = []

        for i in range(N):
            yl = y_lin(matrix[i])
            yn = y_nl(matrix[i])

            table.append({
                "x": base[i],
                "real": to_real(base[i]),
                "y": y[i],
                "y_lin": yl,
                "y_full": yn,
                "d_lin": fabs(y[i]-yl),
                "d_full": fabs(y[i]-yn)
            })

        # --- уравнения ---
        eq_lin = f"y = {b[0]:.4f} + {b[1]:.4f}x1 + {b[2]:.4f}x2 + {b[3]:.4f}x3 + {b[4]:.4f}x4"

        terms = ["1","x1","x2","x3","x4","x1x2","x1x3","x1x4","x2x3","x2x4","x3x4"]
        eq_nl = "y = " + " + ".join(f"{b[i]:.4f}*{terms[i]}" for i in range(len(b)))

        # --- НАТУРАЛЬНАЯ (линейная) ---
        a0 = b[0] + b[1]*x0[0] + b[2]*x0[1] + b[3]*x0[2] + b[4]*x0[3]
        a1 = b[1]*dx[0]
        a2 = b[2]*dx[1]
        a3 = b[3]*dx[2]
        a4 = b[4]*dx[3]

        eq_lin_nat = f"y = {a0:.4f} + {a1:.4f}*λ1 + {a2:.4f}*λ2 + {a3:.4f}*μ1 + {a4:.4f}*μ2"

        # --- НАТУРАЛЬНАЯ (частично нелинейная) ---
        var_names = ["λ1","λ2","μ1","μ2"]

        from collections import defaultdict
        nat = defaultdict(float)

        term_vars = [
            [],
            [0],[1],[2],[3],
            [0,1],[0,2],[0,3],
            [1,2],[1,3],[2,3]
        ]

        for j in range(len(b)):
            coeff = b[j]
            vars_in_term = term_vars[j]

            if not vars_in_term:
                nat[()] += coeff
                continue

            expansions = {():1.0}

            for v in vars_in_term:
                new = {}
                for key, val in expansions.items():
                    new[key] = new.get(key,0)+val*x0[v]
                    new_key = key+(v,)
                    new[new_key] = new.get(new_key,0)+val*dx[v]
                expansions = new

            for key,val in expansions.items():
                nat[key]+=coeff*val

        parts = []
        for k,v in sorted(nat.items(), key=lambda x:(len(x[0]),x[0])):
            if abs(v)<1e-10:
                continue
            if not k:
                parts.append(f"{v:.4f}")
            else:
                name="*".join(var_names[i] for i in k)
                parts.append(f"{v:.4f}*{name}")

        eq_nl_nat = "y = " + " + ".join(parts)

        return table, eq_lin, eq_nl, eq_lin_nat, eq_nl_nat

    res0 = calc_regression(y0, ranges)
    res1 = calc_regression(y1, ranges)

    return res0, res1

# ==============================================================================
# GUI ЛОГИКА
# ==============================================================================

def run_pfe_gui():
    try:
        ranges = {
            "λ1": (dpg.get_value("l1_min"), dpg.get_value("l1_max")),
            "λ2": (dpg.get_value("l2_min"), dpg.get_value("l2_max")),
            "μ1": (dpg.get_value("s0_min"), dpg.get_value("s0_max")),
            "μ2": (dpg.get_value("s1_min"), dpg.get_value("s1_max")),
        }

        task_count = int(dpg.get_value("task_count"))

        dpg.set_value("result_text", "Моделирование...")

        res0, res1 = run_pfe(ranges, task_count)

        draw_table("table0", "table_group0", res0[0])
        draw_table("table1", "table_group1", res1[0])

        text = f"""
=== Приоритет 0 ===
Линейная (нормированная):
{res0[1]}

Линейная (натуральная):
{res0[3]}

Нелинейная (нормированная):
{res0[2]}

Нелинейная (натуральная):
{res0[4]}

=== Приоритет 1 ===
Линейная (нормированная):
{res1[1]}

Линейная (натуральная):
{res1[3]}

Нелинейная (нормированная):
{res1[2]}

Нелинейная (натуральная):
{res1[4]}
"""
        dpg.set_value("result_text", text)

    except Exception as e:
        dpg.set_value("result_text", f"Ошибка: {str(e)}")


def run_dfe_gui():
    try:
        ranges = {
            "λ1": (dpg.get_value("l1_min"), dpg.get_value("l1_max")),
            "λ2": (dpg.get_value("l2_min"), dpg.get_value("l2_max")),
            "μ1": (dpg.get_value("s0_min"), dpg.get_value("s0_max")),
            "μ2": (dpg.get_value("s1_min"), dpg.get_value("s1_max")),
        }

        task_count = int(dpg.get_value("task_count"))

        dpg.set_value("result_text", "Моделирование ДФЭ...")

        res0, res1 = run_dfe(ranges, task_count)

        draw_table("table_dfe0", "table_group_dfe0", res0[0])
        draw_table("table_dfe1", "table_group_dfe1", res1[0])

        text = f"""
=== ДФЭ Приоритет 0 ===
Линейная:
{res0[1]}

Частично нелинейная:
{res0[2]}

=== ДФЭ Приоритет 1 ===
Линейная:
{res1[1]}

Частично нелинейная:
{res1[2]}
"""

        dpg.set_value("result_text", text)

    except Exception as e:
        dpg.set_value("result_text", f"Ошибка: {str(e)}")

def run_all_gui():
    try:
        ranges = {
            "λ1": (dpg.get_value("l1_min"), dpg.get_value("l1_max")),
            "λ2": (dpg.get_value("l2_min"), dpg.get_value("l2_max")),
            "μ1": (dpg.get_value("s0_min"), dpg.get_value("s0_max")),
            "μ2": (dpg.get_value("s1_min"), dpg.get_value("s1_max")),
        }

        task_count = int(dpg.get_value("task_count"))

        dpg.set_value("result_text", "Моделирование...")

        pfe0, pfe1 = run_pfe(ranges, task_count)
        dfe0, dfe1 = run_dfe(ranges, task_count)

        # таблицы
        draw_table("table_pfe0", "table_group_pfe0", pfe0[0])
        draw_table("table_pfe1", "table_group_pfe1", pfe1[0])
        draw_table("table_dfe0", "table_group_dfe0", dfe0[0])
        draw_table("table_dfe1", "table_group_dfe1", dfe1[0])

        text = f"""
=========== ПФЭ ===========
--- Приоритет 0 ---
Линейная норм:
{pfe0[1]}
Линейная натур:
{pfe0[3]}
Нелинейная норм:
{pfe0[2]}
Нелинейная натур:
{pfe0[4]}

--- Приоритет 1 ---
Линейная норм:
{pfe1[1]}
Линейная натур:
{pfe1[3]}
Нелинейная норм:
{pfe1[2]}
Нелинейная натур:
{pfe1[4]}

=========== ДФЭ ===========
--- Приоритет 0 ---
Линейная норм:
{dfe0[1]}
Линейная натур:
{dfe0[3]}
Частично нелинейная норм:
{dfe0[2]}
Частично нелинейная натур:
{dfe0[4]}

--- Приоритет 1 ---
Линейная норм:
{dfe1[1]}
Линейная натур:
{dfe1[3]}
Частично нелинейная норм:
{dfe1[2]}
Частично нелинейная натур:
{dfe1[4]}
"""

        dpg.set_value("result_text", text)

    except Exception as e:
        dpg.set_value("result_text", f"Ошибка: {str(e)}")

def draw_table(tag, parent, data):
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    with dpg.table(header_row=True, tag=tag, parent=parent, row_background=True):

        headers = ["x1","x2","x3","x4","λ1","λ2","μ1","μ2","y","y_lin","y_full","Δ_lin","Δ_full"]
        for h in headers:
            dpg.add_table_column(label=h)

        for row in data:
            with dpg.table_row():
                x1, x2, x3, x4 = row["x"]
                l1, l2, s0, s1 = row["real"]

                dpg.add_text(str(x1))
                dpg.add_text(str(x2))
                dpg.add_text(str(x3))
                dpg.add_text(str(x4))

                dpg.add_text(f"{l1:.3f}")
                dpg.add_text(f"{l2:.3f}")
                dpg.add_text(f"{s0:.3f}")
                dpg.add_text(f"{s1:.3f}")

                dpg.add_text(f"{row['y']:.4f}")
                dpg.add_text(f"{row['y_lin']:.4f}")
                dpg.add_text(f"{row['y_full']:.4f}")
                dpg.add_text(f"{row['d_lin']:.4f}")
                dpg.add_text(f"{row['d_full']:.4f}")


# ==============================================================================
# UI
# ==============================================================================

def setup_page():
    with dpg.group(parent="main"):
        dpg.add_text("Дробный факторный эксперимент (ДФЭ)")
        dpg.add_separator()

        dpg.add_text("Диапазоны факторов:")

        dpg.add_input_float(label="λ1 min", tag="l1_min", default_value=0.3, width=300)
        dpg.add_input_float(label="λ1 max", tag="l1_max", default_value=0.8, width=300)

        dpg.add_input_float(label="λ2 min", tag="l2_min", default_value=0.4, width=300)
        dpg.add_input_float(label="λ2 max", tag="l2_max", default_value=0.9, width=300)

        dpg.add_input_float(label="μ1 (приоритет 0) min", tag="s0_min", default_value=1.7, width=300)
        dpg.add_input_float(label="μ1 (приоритет 0) max", tag="s0_max", default_value=2.1, width=300)

        dpg.add_input_float(label="μ2 (приоритет 1) min", tag="s1_min", default_value=1.5, width=300)
        dpg.add_input_float(label="μ2 (приоритет 1) max", tag="s1_max", default_value=2.5, width=300)

        dpg.add_input_int(label="Количество заявок", tag="task_count", default_value=1000)

        dpg.add_separator()

        dpg.add_button(label="Запустить ПФЭ + ДФЭ", callback=run_all_gui, width=500, height=100)

        dpg.add_separator()

        # dpg.add_child_window(height=300, width=1400, horizontal_scrollbar=True)
        with dpg.group():
            dpg.add_text("", tag="result_text", wrap=0)

        dpg.add_separator()
        dpg.add_text("Результаты (ПФЭ и ДФЭ)")

        with dpg.child_window(tag="tables_scroll", height=600, width=1800, horizontal_scrollbar=True):

            dpg.add_text("ПФЭ — Приоритет 0")
            dpg.add_child_window(tag="table_group_pfe0", height=300)

            dpg.add_separator()
            dpg.add_text("ПФЭ — Приоритет 1")
            dpg.add_child_window(tag="table_group_pfe1", height=300)

            dpg.add_separator()
            dpg.add_text("ДФЭ — Приоритет 0")
            dpg.add_child_window(tag="table_group_dfe0", height=300)

            dpg.add_separator()
            dpg.add_text("ДФЭ — Приоритет 1")
            dpg.add_child_window(tag="table_group_dfe1", height=300)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    dpg.create_context()

    setup_custom_font()

    with dpg.window(tag="MainWindow", width=1400, height=1200):
        with dpg.group(tag="main"):
            setup_page()

    dpg.create_viewport(title="ПФЭ СМО", width=1500, height=1300)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()