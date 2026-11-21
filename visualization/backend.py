"""Interactive launchers must undo batch benchmarks' Agg backend selection."""


def enable_gui_backend():
    import matplotlib.pyplot as plt
    try:
        plt.switch_backend('TkAgg')
    except (ImportError,RuntimeError) as error:
        raise RuntimeError('Cannot open Tk animation window. Run in a desktop session '
                           'with a valid DISPLAY, or use --no-show for batch execution.') from error
