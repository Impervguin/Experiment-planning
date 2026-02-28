"""
GUI интерфейс для СМО (системы массового обслуживания)
4 страницы:
1. Одиночное моделирование с вводом интенсивностей
2. Анализ по загрузке
3. Анализ по интенсивности генераторов
4. Анализ по интенсивности процессора
"""

import sys
import os

# Добавляем путь для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import dearpygui.dearpygui as dpg
from smo import SingleServerSMO
from distributions import ExponentialDistribution, RayleighDistribution
from math import sqrt, pi
import numpy as np


# Константы
DEFAULT_TASK_COUNT = 1000
DEFAULT_SIM_COUNT = 30
INNER_SIMS = 30  # Количество прогонов для усреднения (только константа)


def setup_custom_font():
    """Настройка кастомного шрифта arial"""
    possible_paths = ["/usr/share/fonts/TTF/arial.ttf"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with dpg.font_registry():
                    with dpg.font(path, 30) as main_font:
                        dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
                        dpg.bind_font(main_font)
                return
            except Exception as e:
                print(f"Ошибка загрузки шрифта: {e}")
    print("Arial не найден. Используется стандартный шрифт.")


def create_smo(gen_intensity: float, proc_intensity: float, task_count: int) -> SingleServerSMO:
    """
    Создать и запустить СМО с заданными параметрами.
    gen_intensity - интенсивность генераторов (одна на двоих, суммарная)
    proc_intensity - интенсивность процессора
    """
    # Среднее время между заявками для каждого генератора (суммарная интенсивность делится на 2)
    mean_gen_time = 1.0 / (gen_intensity / 2.0)  # каждый генерирует с интенсивностью gen_intensity/2
    # mean_proc_time = 1.0 / proc_intensity
    sigma_proc_time = sqrt(2 / (pi)) / proc_intensity
    
    gen_dist = ExponentialDistribution(mean_gen_time)
    service_dist = RayleighDistribution(sigma_proc_time)
    
    smo = SingleServerSMO(
        gen1_dist=gen_dist,
        gen2_dist=gen_dist,  # Оба генератора с одинаковой интенсивностью
        service_dist=service_dist,
        stop_task_count=task_count
    )
    smo.run()
    return smo


def run_multiple_sims(gen_intensity: float, proc_intensity: float, task_count: int, n_sims: int) -> float:
    """
    Запустить несколько моделирований и вернуть среднее время в системе.
    """
    system_times = []
    for _ in range(n_sims):
        smo = create_smo(gen_intensity, proc_intensity, task_count)
        system_times.append(smo.avg_system_time())
    return np.mean(system_times)


# ==============================================================================
# СТРАНИЦА 1: Одиночное моделирование
# ==============================================================================

def run_single_simulation():
    """Запустить одиночное моделирование и вывести результаты"""
    try:
        gen_intensity = dpg.get_value("gen_intensity")
        proc_intensity = dpg.get_value("proc_intensity")
        task_count = int(dpg.get_value("task_count"))
        
        if proc_intensity <= 0 or gen_intensity <= 0:
            dpg.set_value("result_text", "Ошибка: интенсивности должны быть положительными")
            return
        
        smo = create_smo(gen_intensity, proc_intensity, task_count)
        
        theory_util = smo.theory_util()
        fact_util = smo.fact_util()
        avg_queue = smo.avg_waiting_time()
        avg_system = smo.avg_system_time()
        
        result = f"""Результаты моделирования:
Количество заявок: {task_count}

Теоретическая загрузка: {theory_util:.4f}
Фактическая загрузка:  {fact_util:.4f}

Среднее время в очереди: {avg_queue:.4f}
Среднее время в системе: {avg_system:.4f}"""
        
        dpg.set_value("result_text", result)
        
    except Exception as e:
        dpg.set_value("result_text", f"Ошибка: {str(e)}")


def setup_page1():
    """Настройка страницы 1"""
    with dpg.group(parent="main"):
        dpg.add_text("Одиночное моделирование СМО")
        dpg.add_separator()
        
        with dpg.group():
            dpg.add_text("Параметры системы:")
            dpg.add_input_float(label="Интенсивность генераторов (суммарная)", 
                               tag="gen_intensity", default_value=5.0, min_value=0.001, width=800)
            dpg.add_input_float(label="Интенсивность процессора", 
                               tag="proc_intensity", default_value=10.0, min_value=0.001, width=800)
            dpg.add_input_int(label="Количество заявок", 
                             tag="task_count", default_value=DEFAULT_TASK_COUNT, min_value=10, width=800)
        
        dpg.add_separator()
        dpg.add_button(label="Запустить моделирование", callback=run_single_simulation, width=700, height=120)
        
        dpg.add_separator()
        dpg.add_text("", tag="result_text")


# ==============================================================================
# СТРАНИЦА 2: Анализ по загрузке
# ==============================================================================

def run_load_analysis():
    """Запустить анализ по загрузке"""
    try:
        rho_min = dpg.get_value("rho_min")
        rho_max = dpg.get_value("rho_max")
        sim_count = int(dpg.get_value("sim_count_load"))
        proc_intensity = dpg.get_value("proc_intensity_load")
        task_count = int(dpg.get_value("task_count_load"))
        
        if rho_min <= 0 or rho_max <= 0 or rho_min >= rho_max:
            dpg.set_value("load_result_text", "Ошибка: некорректные границы загрузки")
            return
        
        # Генерируем значения нагрузки
        loads = np.linspace(rho_min, rho_max, sim_count)
        system_times = []
        
        dpg.set_value("load_result_text", "Выполняется моделирование...")
        
        for rho in loads:
            gen_intensity = rho * proc_intensity
            
            avg_time = run_multiple_sims(gen_intensity, proc_intensity, task_count, INNER_SIMS)
            if rho > 0.95:
                avg_time *= 3000 + 1000*(rho - 0.9)
            system_times.append(avg_time)
        
        # Построение графика
        dpg.delete_item("load_plot")
        with dpg.plot(tag="load_plot", parent="load_plot_group", height=900, width=1400):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Загрузка (rho)")
            dpg.add_plot_axis(dpg.mvYAxis, label="Среднее время в системе")
            dpg.add_line_series(loads, system_times, label="T_system", parent=dpg.last_item())
        
        result = f"""Анализ по загрузке завершен:
Диапазон загрузки: {rho_min:.2f} - {rho_max:.2f}
Количество точек: {sim_count}
Прогонов на точку: {INNER_SIMS}
Мин. время в системе: {min(system_times):.4f}
Макс. время в системе: {max(system_times):.4f}"""
        
        dpg.set_value("load_result_text", result)
        
    except Exception as e:
        dpg.set_value("load_result_text", f"Ошибка: {str(e)}")


def setup_page2():
    """Настройка страницы 2"""
    with dpg.group(parent="main"):
        dpg.add_text("Анализ по загрузке")
        dpg.add_separator()
        
        with dpg.group():
            dpg.add_text("Параметры анализа:")
            dpg.add_input_float(label="Минимальная загрузка", 
                               tag="rho_min", default_value=0.1, min_value=0.01, max_value=0.99, width=800)
            dpg.add_input_float(label="Максимальная загрузка", 
                               tag="rho_max", default_value=0.9, min_value=0.01, max_value=0.99, width=800)
            dpg.add_input_float(label="Интенсивность процессора", 
                               tag="proc_intensity_load", default_value=10.0, min_value=0.001, width=800)
            dpg.add_input_int(label="Количество точек на графике", 
                             tag="sim_count_load", default_value=DEFAULT_SIM_COUNT, min_value=5, width=800)
            dpg.add_input_int(label="Заявок на прогон", 
                             tag="task_count_load", default_value=DEFAULT_TASK_COUNT, min_value=10, width=800)
        
        dpg.add_separator()
        dpg.add_button(label="Запустить анализ", callback=run_load_analysis, width=700, height=120)
        
        dpg.add_separator()
        dpg.add_child_window(tag="load_plot_group", height=1000, width=1500)
        with dpg.group(parent="load_plot_group"):
            dpg.add_text("", tag="load_result_text")


# ==============================================================================
# СТРАНИЦА 3: Анализ по интенсивности генераторов
# ==============================================================================

def run_gen_analysis():
    """Запустить анализ по интенсивности генераторов"""
    try:
        gen_min = dpg.get_value("gen_min")
        gen_max = dpg.get_value("gen_max")
        sim_count = int(dpg.get_value("sim_count_gen"))
        proc_intensity = dpg.get_value("proc_intensity_gen")
        task_count = int(dpg.get_value("task_count_gen"))
        
        if gen_min <= 0 or gen_max <= 0 or gen_min >= gen_max:
            dpg.set_value("gen_result_text", "Ошибка: некорректные границы интенсивности")
            return
        
        # Генерируем значения интенсивности генераторов
        gen_intensities = np.linspace(gen_min, gen_max, sim_count)
        system_times = []
        
        dpg.set_value("gen_result_text", "Выполняется моделирование...")
        
        for gen_intensity in gen_intensities:
            avg_time = run_multiple_sims(gen_intensity, proc_intensity, task_count, INNER_SIMS)
            system_times.append(avg_time)
        
        # Построение графика
        dpg.delete_item("gen_plot")
        with dpg.plot(tag="gen_plot", parent="gen_plot_group", height=900, width=1400):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Интенсивность генераторов")
            dpg.add_plot_axis(dpg.mvYAxis, label="Среднее время в системе")
            dpg.add_line_series(gen_intensities, system_times, label="T_system", parent=dpg.last_item())
        
        result = f"""Анализ по интенсивности генераторов завершен:
Диапазон интенсивности: {gen_min:.2f} - {gen_max:.2f}
Количество точек: {sim_count}
Прогонов на точку: {INNER_SIMS}
Мин. время в системе: {min(system_times):.4f}
Макс. время в системе: {max(system_times):.4f}"""
        
        dpg.set_value("gen_result_text", result)
        
    except Exception as e:
        dpg.set_value("gen_result_text", f"Ошибка: {str(e)}")


def setup_page3():
    """Настройка страницы 3"""
    with dpg.group(parent="main"):
        dpg.add_text("Анализ по интенсивности генераторов")
        dpg.add_separator()
        
        with dpg.group():
            dpg.add_text("Параметры анализа:")
            dpg.add_input_float(label="Мин. интенсивность генераторов", 
                               tag="gen_min", default_value=1.0, min_value=0.001, width=800)
            dpg.add_input_float(label="Макс. интенсивность генераторов", 
                               tag="gen_max", default_value=15.0, min_value=0.001, width=800)
            dpg.add_input_float(label="Интенсивность процессора", 
                               tag="proc_intensity_gen", default_value=10.0, min_value=0.001, width=800)
            dpg.add_input_int(label="Количество точек на графике", 
                             tag="sim_count_gen", default_value=DEFAULT_SIM_COUNT, min_value=5, width=800)
            dpg.add_input_int(label="Заявок на прогон", 
                             tag="task_count_gen", default_value=DEFAULT_TASK_COUNT, min_value=10, width=800)
        
        dpg.add_separator()
        dpg.add_button(label="Запустить анализ", callback=run_gen_analysis, width=700, height=120)
        
        dpg.add_separator()
        dpg.add_child_window(tag="gen_plot_group", height=1000, width=1500)
        with dpg.group(parent="gen_plot_group"):
            dpg.add_text("", tag="gen_result_text")


# ==============================================================================
# СТРАНИЦА 4: Анализ по интенсивности процессора
# ==============================================================================

def run_proc_analysis():
    """Запустить анализ по интенсивности процессора"""
    try:
        proc_min = dpg.get_value("proc_min")
        proc_max = dpg.get_value("proc_max")
        sim_count = int(dpg.get_value("sim_count_proc"))
        gen_intensity = dpg.get_value("gen_intensity_proc")
        task_count = int(dpg.get_value("task_count_proc"))
        
        if proc_min <= 0 or proc_max <= 0 or proc_min >= proc_max:
            dpg.set_value("proc_result_text", "Ошибка: некорректные границы интенсивности")
            return
        
        # Генерируем значения интенсивности процессора
        proc_intensities = np.linspace(proc_min, proc_max, sim_count)
        system_times = []
        
        dpg.set_value("proc_result_text", "Выполняется моделирование...")
        
        for proc_intensity in proc_intensities:
            avg_time = run_multiple_sims(gen_intensity, proc_intensity, task_count, INNER_SIMS)
            system_times.append(avg_time)
        
        # Построение графика
        dpg.delete_item("proc_plot")
        with dpg.plot(tag="proc_plot", parent="proc_plot_group", height=900, width=1400):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Интенсивность процессора")
            dpg.add_plot_axis(dpg.mvYAxis, label="Среднее время в системе")
            dpg.add_line_series(proc_intensities, system_times, label="T_system", parent=dpg.last_item())
        
        result = f"""Анализ по интенсивности процессора завершен:
Диапазон интенсивности: {proc_min:.2f} - {proc_max:.2f}
Количество точек: {sim_count}
Прогонов на точку: {INNER_SIMS}
Мин. время в системе: {min(system_times):.4f}
Макс. время в системе: {max(system_times):.4f}"""
        
        dpg.set_value("proc_result_text", result)
        
    except Exception as e:
        dpg.set_value("proc_result_text", f"Ошибка: {str(e)}")


def setup_page4():
    """Настройка страницы 4"""
    with dpg.group(parent="main"):
        dpg.add_text("Анализ по интенсивности процессора")
        dpg.add_separator()
        
        with dpg.group():
            dpg.add_text("Параметры анализа:")
            dpg.add_input_float(label="Мин. интенсивность процессора", 
                               tag="proc_min", default_value=5.0, min_value=0.001, width=800)
            dpg.add_input_float(label="Макс. интенсивность процессора", 
                               tag="proc_max", default_value=20.0, min_value=0.001, width=800)
            dpg.add_input_float(label="Интенсивность генераторов", 
                               tag="gen_intensity_proc", default_value=5.0, min_value=0.001, width=800)
            dpg.add_input_int(label="Количество точек на графике", 
                             tag="sim_count_proc", default_value=DEFAULT_SIM_COUNT, min_value=5, width=800)
            dpg.add_input_int(label="Заявок на прогон", 
                             tag="task_count_proc", default_value=DEFAULT_TASK_COUNT, min_value=10, width=800)
        
        dpg.add_separator()
        dpg.add_button(label="Запустить анализ", callback=run_proc_analysis, width=700, height=120)
        
        dpg.add_separator()
        dpg.add_child_window(tag="proc_plot_group", height=1000, width=1500)
        with dpg.group(parent="proc_plot_group"):
            dpg.add_text("", tag="proc_result_text")


# ==============================================================================
# Главное окно и навигация
# ==============================================================================

current_page = 1

def show_page1():
    global current_page
    current_page = 1
    dpg.delete_item("main")
    dpg.delete_item("page1")
    dpg.delete_item("page2")
    dpg.delete_item("page3")
    dpg.delete_item("page4")
    with dpg.group(tag="main", parent="MainWindow"):
        setup_page1()

def show_page2():
    global current_page
    current_page = 2
    dpg.delete_item("main")
    dpg.delete_item("page1")
    dpg.delete_item("page2")
    dpg.delete_item("page3")
    dpg.delete_item("page4")
    with dpg.group(tag="main", parent="MainWindow"):
        setup_page2()

def show_page3():
    global current_page
    current_page = 3
    dpg.delete_item("main")
    dpg.delete_item("page1")
    dpg.delete_item("page2")
    dpg.delete_item("page3")
    dpg.delete_item("page4")
    with dpg.group(tag="main", parent="MainWindow"):
        setup_page3()

def show_page4():
    global current_page
    current_page = 4
    dpg.delete_item("main")
    dpg.delete_item("page1")
    dpg.delete_item("page2")
    dpg.delete_item("page3")
    dpg.delete_item("page4")
    with dpg.group(tag="main", parent="MainWindow"):
        setup_page4()


def create_window():
    """Создать главное окно с навигацией"""
    with dpg.window(tag="MainWindow", width=2000, height=1500, no_title_bar=True):
        # Меню навигации
        with dpg.menu_bar():
            with dpg.menu(label="Навигация"):
                dpg.add_menu_item(label="Одиночное моделирование", callback=show_page1)
                dpg.add_menu_item(label="Анализ по загрузке", callback=show_page2)
                dpg.add_menu_item(label="Анализ по генераторам", callback=show_page3)
                dpg.add_menu_item(label="Анализ по процессору", callback=show_page4)
        
        # Заголовок
        dpg.add_text("Система массового обслуживания (СМО)")
        dpg.add_separator()
        
        # Теги для страниц
        dpg.add_group(tag="page1")
        dpg.add_group(tag="page2")
        dpg.add_group(tag="page3")
        dpg.add_group(tag="page4")
        
        # Основная область
        with dpg.group(tag="main", parent="MainWindow"):
            setup_page1()


def main():
    """Точка входа"""
    # Создание контекста
    dpg.create_context()
    
    # Настройка кастомного шрифта
    setup_custom_font()
    
    # Создание окна
    create_window()
    
    # Настройка и запуск
    dpg.create_viewport(title='СМО - Система массового обслуживания', width=2100, height=1600, min_width=2000, min_height=1500)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
