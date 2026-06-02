import sys
import os
from itertools import product
from math import fabs, sqrt, pi

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import dearpygui.dearpygui as dpg

from smo import SingleServerSMO
from distributions import ExponentialDistribution, RayleighDistribution

runs_per_combination = 30


# ==============================================================================
# ШРИФТ
# ==============================================================================

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
# СИМУЛЯЦИЯ
# ==============================================================================

def simulate(l1, l2, s0, s1, task_count):
    gen1 = ExponentialDistribution(1 / l1)
    gen2 = ExponentialDistribution(1 / l2)
    serv0 = RayleighDistribution(sqrt(2 / pi) / s0)
    serv1 = RayleighDistribution(sqrt(2 / pi) / s1)

    smo = SingleServerSMO(gen1, gen2, serv0, task_count, serv1)
    smo.run()

    return (
        smo.avg_waiting_time_priority(0),
        smo.avg_waiting_time_priority(1)
    )


# ==============================================================================
# ОЦКП
# ==============================================================================

def run_occp(ranges, task_count):
    names = list(ranges.keys())
    k = 4
    n = 2 ** k
    N = 2 ** k + 2 * k + 1

    # формулы из лекции
    alpha = sqrt(n/2 * (sqrt(N/n) - 1))
    a = sqrt(n / N)

    def to_real(x):
        res = []
        for i, xi in enumerate(x):
            a_, b_ = ranges[names[i]]
            x0 = (a_ + b_) / 2
            dx = (b_ - a_) / 2
            res.append(x0 + xi * dx)
        return res

    # план
    base = list(product([-1,1], repeat=4))

    stars = []
    for i in range(4):
        for s in [-alpha, alpha]:
            r = [0]*4
            r[i] = s
            stars.append(r)

    center = [[0]*4]

    plan = base + stars + center

    def extend(r):
        x1,x2,x3,x4 = r
        return [
            1,
            x1,x2,x3,x4,
            x1*x2,x1*x3,x1*x4,x2*x3,x2*x4,x3*x4,
            x1*x1 - a,
            x2*x2 - a,
            x3*x3 - a,
            x4*x4 - a
        ]

    matrix = [extend(r) for r in plan]

    y0, y1 = [], []

    for r in plan:
        l1,l2,s0,s1 = to_real(r)

        res0,res1 = [],[]

        for _ in range(runs_per_combination):
            a0,a1 = simulate(l1,l2,s0,s1,task_count)
            res0.append(a0)
            res1.append(a1)

        y0.append(sum(res0)/len(res0))
        y1.append(sum(res1)/len(res1))

    # ==============================================================================
    # РЕГРЕССИЯ
    # ==============================================================================

    def calc(y):
        Np = len(matrix)

        # коэффициенты регрессии
        b = [
            sum(matrix[i][j] * y[i] for i in range(Np)) / Np
            for j in range(len(matrix[0]))
        ]

        names = list(ranges.keys())

        # --- переход к натуральным переменным ---
        x0 = []
        dx = []
        for name in names:
            a_, b_ = ranges[name]
            x0.append((a_ + b_) / 2)
            dx.append((b_ - a_) / 2)

        # =========================
        # ЛИНЕЙНАЯ МОДЕЛЬ
        # =========================
        def y_lin(r):
            return (
                b[0]
                + b[1]*r[1]
                + b[2]*r[2]
                + b[3]*r[3]
                + b[4]*r[4]
            )

        # натуральная линейная
        a0 = b[0] + b[1]*x0[0] + b[2]*x0[1] + b[3]*x0[2] + b[4]*x0[3]
        a1 = b[1]*dx[0]
        a2 = b[2]*dx[1]
        a3 = b[3]*dx[2]
        a4 = b[4]*dx[3]

        eq_lin = f"y = {b[0]:.4f} + {b[1]:.4f}x1 + {b[2]:.4f}x2 + {b[3]:.4f}x3 + {b[4]:.4f}x4"
        eq_lin_nat = f"y = {a0:.4f} + {a1:.4f}*λ1 + {a2:.4f}*λ2 + {a3:.4f}*μ1 + {a4:.4f}*μ2"

        # =========================
        # НЕЛИНЕЙНАЯ МОДЕЛЬ (ОЦКП)
        # =========================
        b = [
            sum(matrix[i][j] * y[i] for i in range(Np)) / sum(matrix[i][j]**2  for i in range(Np))
            for j in range(len(matrix[0]))
        ]

        def y_nl(r):
            return sum(b[i] * r[i] for i in range(len(r)))

        eq_nl = "y = " + " + ".join(
            f"{b[i]:.4f}*{['1','x1','x2','x3','x4','x1x2','x1x3','x1x4','x2x3','x2x4','x3x4','(x1^2-a)','(x2^2-a)','(x3^2-a)','(x4^2-a)'][i]}"
            for i in range(len(b))
        )

        # =========================
        # НАТУРАЛЬНАЯ НЕЛИНЕЙНАЯ (как в ДФЭ)
        # =========================
        var_names = ["λ1", "λ2", "μ1", "μ2"]

        term_vars = [
            [],
            [0],[1],[2],[3],
            [0,1],[0,2],[0,3],
            [1,2],[1,3],[2,3],
            [0],[1],[2],[3]
        ]

        from collections import defaultdict
        nat = defaultdict(float)

        for j in range(len(b)):
            coeff = b[j]
            vars_in_term = term_vars[j]

            if not vars_in_term:
                nat[()] += coeff
                continue

            expansions = {(): 1.0}

            for v in vars_in_term:
                new = {}
                for key, val in expansions.items():
                    new[key] = new.get(key, 0) + val * x0[v]
                    new_key = key + (v,)
                    new[new_key] = new.get(new_key, 0) + val * dx[v]
                expansions = new

            for key, val in expansions.items():
                nat[key] += coeff * val

        parts = []
        for k, v in sorted(nat.items(), key=lambda x: (len(x[0]), x[0])):
            if abs(v) < 1e-10:
                continue
            if not k:
                parts.append(f"{v:.4f}")
            else:
                parts.append(f"{v:.4f}*" + "*".join(var_names[i] for i in k))

        eq_nl_nat = "y = " + " + ".join(parts)

        # =========================
        # ТАБЛИЦА ОШИБОК
        # =========================
        table = []

        for i in range(Np):
            yl = y_lin(matrix[i])
            yn = y_nl(matrix[i])

            table.append({
                "x": plan[i],
                "real": to_real(plan[i]),
                "y": y[i],
                "y_lin": yl,
                "y_nl": yn,
                "d_lin": fabs(y[i] - yl),
                "d_nl": fabs(y[i] - yn),
            })

        return table, eq_lin, eq_lin_nat, eq_nl, eq_nl_nat

    return calc(y0), calc(y1), alpha, a


# ==============================================================================
# GUI
# ==============================================================================

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

        res0, res1, alpha, a = run_occp(ranges, task_count)

        draw_table("table0","table_group0",res0[0])
        draw_table("table1","table_group1",res1[0])
        

        text = f"""
α = {alpha:.4f}
a = {a:.4f}

===== ПРИОРИТЕТ 0 =====

Линейная (норм):
{res0[1]}

Линейная (натур):
{res0[2]}

Нелинейная (норм):
{res0[3]}

Нелинейная (натур):
{res0[4]}

===== ПРИОРИТЕТ 1 =====

Линейная (норм):
{res1[1]}

Линейная (натур):
{res1[2]}

Нелинейная (норм):
{res1[3]}

Нелинейная (натур):
{res1[4]}
"""
        dpg.set_value("result_text", text)

    except Exception as e:
        dpg.set_value("result_text", str(e))


def draw_table(tag,parent,data):
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)

    with dpg.table(tag=tag,parent=parent,header_row=True,row_background=True):

        headers = ["x1","x2","x3","x4","λ1","λ2","μ1","μ2","y",
                   "y_lin","Δ_lin","y_nl","Δ_nl"]

        for h in headers:
            dpg.add_table_column(label=h)

        for row in data:
            with dpg.table_row():
                for v in row["x"]:
                    dpg.add_text(str(v))
                for v in row["real"]:
                    dpg.add_text(f"{v:.3f}")

                dpg.add_text(f"{row['y']:.4f}")
                dpg.add_text(f"{row['y_lin']:.4f}")
                dpg.add_text(f"{row['d_lin']:.4f}")
                dpg.add_text(f"{row['y_nl']:.4f}")
                dpg.add_text(f"{row['d_nl']:.4f}")


# ==============================================================================
# UI
# ==============================================================================

def setup_page():
    with dpg.group(parent="main"):

        dpg.add_text("ОЦКП")
        dpg.add_separator()

        dpg.add_input_float(label="λ1 min", tag="l1_min", default_value=0.3, width=300)
        dpg.add_input_float(label="λ1 max", tag="l1_max", default_value=0.8, width=300)

        dpg.add_input_float(label="λ2 min", tag="l2_min", default_value=0.4, width=300)
        dpg.add_input_float(label="λ2 max", tag="l2_max", default_value=0.9, width=300)

        dpg.add_input_float(label="μ1 min", tag="s0_min", default_value=1.7, width=300)
        dpg.add_input_float(label="μ1 max", tag="s0_max", default_value=2.1, width=300)

        dpg.add_input_float(label="μ2 min", tag="s1_min", default_value=1.5, width=300)
        dpg.add_input_float(label="μ2 max", tag="s1_max", default_value=2.5, width=300)

        dpg.add_input_int(label="Количество заявок", tag="task_count", default_value=1000)

        dpg.add_separator()

        dpg.add_button(label="Запустить ОЦКП", callback=run_all_gui, width=500, height=100)

        dpg.add_separator()

        dpg.add_text("", tag="result_text")

        dpg.add_separator()

        with dpg.child_window(height=600,width=1800,horizontal_scrollbar=True):
            dpg.add_text("Приоритет 0")
            dpg.add_child_window(tag="table_group0",height=300)

            dpg.add_separator()
            dpg.add_text("Приоритет 1")
            dpg.add_child_window(tag="table_group1",height=300)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    dpg.create_context()
    setup_custom_font()

    with dpg.window(tag="MainWindow", width=1400, height=1200):
        with dpg.group(tag="main"):
            setup_page()

    dpg.create_viewport(title="ОЦКП", width=1500, height=1300)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()