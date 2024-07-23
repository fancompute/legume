"""
Various utilities used in the main code for printing

"""
import sys
import numpy as np
# Import rich if available
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich import print  # This is an upgraded version of standard print
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def load_bar(perc, precision=20):
    box = "\u2588"
    hor = "\u2500"
    down_right = "\u250C"
    down_left = "\u2510"
    up_right = "\u2514"
    up_left = "\u2518"
    pipe = "\u2502"

    filled_blocks = int(perc / 100 *
                        precision)  # Considering 100 characters width for 100%
    bar = box * filled_blocks + '-' * (precision - filled_blocks)

    # Creating a box around the loading bar
    box_top = down_right + hor * precision + down_left
    box_bottom = up_right + hor * precision + up_left
    bar_with_box = pipe + bar + pipe

    return bar_with_box


def verbose_print(text, verbose, flush=False, end='\n'):
    """Print if verbose_ex==True
            """
    if verbose == True:
        if flush == False:
            print(text, end=end)
        else:
            sys.stdout.write("\r" + text)
            sys.stdout.flush()


def update_prog(ik, num_k, verbose, text):
    verbose_print(f"{text} {load_bar((ik+1)/num_k*100,precision=30)}" +
                  f" {ik+1} of {num_k}",
                  verbose,
                  flush=True)


def print_GME_report_rich(gme):
    t_tot = gme.total_time
    t_guided, t_guided_perc = gme.t_guided, gme.t_guided / t_tot * 100
    t_eps_inv, t_eps_inv_perc = gme.t_eps_inv, gme.t_eps_inv / t_tot * 100
    t_diag, t_diag_perc = gme.t_eig - gme.t_symmetry, (
        gme.t_eig - gme.t_symmetry) / t_tot * 100
    t_creat_mat, t_creat_mat_perc = gme.t_creat_mat, gme.t_creat_mat / t_tot * 100
    t_symmetry, t_symmetry_perc = gme.t_symmetry, gme.t_symmetry / t_tot * 100

    verbose_print("", gme.verbose, flush=True)
    if gme.verbose:
        table = Table(title="")
        table.add_column(
            f"Steps in GuidedModeExp: {np.shape(gme.gvec)[1]} plane waves" +
            f" and {len(gme.gmode_inds)} guided modes",
            justify="Left",
            style="cyan",
            no_wrap=True)
        table.add_column("Time (s)", style="magenta")
        table.add_column("% vs total T", justify="right", style="green")

        table.add_row(
            f"Guided modes computation with gmode_compute='[b]{gme.gmode_compute.lower()}[/b]'",
            f"{t_guided:.3f}",
            f"{load_bar(t_guided_perc):<23}{t_guided_perc:>4.0f}%")
        table.add_row(
            "Inverse matrix of Fourier-space permittivity", f"{t_eps_inv:.3f}",
            f"{load_bar(t_eps_inv_perc):<23}{t_eps_inv_perc:>4.0f}%")
        table.add_row(
            f"Matrix diagionalization using the '[b]{gme.eig_solver.lower()}[/b]' solver",
            f"{t_diag:.3f}",
            f"{load_bar(t_diag_perc):<23}{t_diag_perc:>4.0f}%")
        table.add_row(
            "Creating GME matrix",
            f"{t_creat_mat:.3f}",
            f"{load_bar(t_creat_mat_perc):<23}{t_creat_mat_perc:>4.0f}%",
            end_section=not gme.kz_symmetry)
        if gme.kz_symmetry:
            if gme.use_sparse == True:
                str_mat_used = "sparse"
            elif gme.use_sparse == False:
                str_mat_used = "dense"
            table.add_row(
                f"Creating change of basis matrix using [b]{str_mat_used}[/b] matrices",
                f"{t_symmetry:.3f}",
                f"{load_bar(t_symmetry_perc):<23}{t_symmetry_perc:>4.0f}%",
                end_section=True)
        table.add_row(
            f"Total time for real part of frequencies for {gme.kpoints.shape[1]} k-points",
            f"[u]{t_tot:>3.3f}[/u]",
            f"{load_bar(100):<23}{100:>4.0f}%",
            style="bold",
            end_section=True)

        console = Console()
        console.print(table)


def print_GME_report(gme):
    t_tot = gme.total_time
    t_guided, t_guided_perc = gme.t_guided, gme.t_guided / t_tot * 100
    t_eps_inv, t_eps_inv_perc = gme.t_eps_inv, gme.t_eps_inv / t_tot * 100
    t_diag, t_diag_perc = gme.t_eig - gme.t_symmetry, (
        gme.t_eig - gme.t_symmetry) / t_tot * 100
    t_creat_mat, t_creat_mat_perc = gme.t_creat_mat, gme.t_creat_mat / t_tot * 100
    t_symmetry, t_symmetry_perc = gme.t_symmetry, gme.t_symmetry / t_tot * 100
    verbose_print("", gme.verbose, flush=True)
    verbose_print(
        f"{t_tot:.3f}s total time for real part of frequencies in GuidedModeExp, of which",
        gme.verbose)
    verbose_print(
        f"  {t_guided:.3f}s ({t_guided_perc:.0f}%) for guided modes computation using"
        f" the gmode_compute='{gme.gmode_compute.lower()}' method",
        gme.verbose)
    verbose_print(
        f"  {t_eps_inv:.3f}s ({t_eps_inv_perc:.0f}%) for inverse matrix of Fourier-space "
        f"permittivity", gme.verbose)
    verbose_print(
        f"  {t_diag:.3f}s ({t_diag_perc:.0f}%) for matrix diagionalization using "
        f"the '{gme.eig_solver.lower()}' solver", gme.verbose)
    verbose_print(
        f"  {t_creat_mat:.3f}s ({t_creat_mat_perc:.0f}%) for creating GME matrix",
        gme.verbose)

    if gme.kz_symmetry:
        if gme.use_sparse == True:
            str_mat_used = "sparse"
        elif gme.use_sparse == False:
            str_mat_used = "dense"
        verbose_print(
            f"  {t_symmetry:.3f}s ({t_symmetry_perc:.0f}%) for creating change of basis matrix and multiply it"
            + f" using {str_mat_used} matrices", gme.verbose)


def print_GME_im_report_rich(gme):
    verbose_print("", gme.verbose, flush=True)
    if gme.verbose:
        table = Table(title="")
        table.add_column(
            f"Steps in GuidedModeExp: {np.shape(gme.gvec)[1]} plane waves" +
            f" and {len(gme.gmode_inds)} guided modes",
            justify="Left",
            style="cyan",
            no_wrap=True)
        table.add_column("Time (s)", style="magenta")
        table.add_row(
            f"Total time for imaginary part of frequencies for {len(gme.freqs.flatten())} eigenmodes",
            f"[u]{(gme.t_imag):.3f}[/u]",
            style="bold")
        console = Console()
        console.print(table)


def print_GME_im_report(gme):
    verbose_print("", gme.verbose, flush=True)
    verbose_print(
        f"{(gme.t_imag):.3f}s  total time for imaginary part"
        " of frequencies", gme.verbose)


def print_ESE_report_rich(exc):
    verbose_print("", exc.verbose_ex, flush=True)
    if exc.verbose_ex:
        table = Table(title="")
        table.add_column(
            f"Steps in ExcitonSchroedEq: {np.shape(exc.gvec)[1]} plane waves",
            justify="Left",
            style="cyan",
            no_wrap=True)
        table.add_column("Time (s)", style="magenta")
        table.add_column("% vs total T", justify="right", style="green")
        table.add_row(f"Diagonalization of the Hamiltonian",
                      f"{exc.t_eig:.4f}",
                      f"{exc.t_eig/exc.total_time*100:.0f}%",
                      end_section=True)
        table.add_row(
            f"Total time for excitonic energies for {exc.kpoints.shape[1]} k-points",
            f"[u]{exc.total_time:.4f}[/u]",
            "100%",
            style="bold")
        console = Console()
        console.print(table)


def print_ESE_report(exc):
    verbose_print("", exc.verbose_ex, flush=True)
    verbose_print(
        f"{exc.total_time:.4f}s total time for excitonic energies, of which",
        exc.verbose_ex)
    verbose_print(f"  {exc.t_eig:.4f}s for diagonalization of the Hamiltonian",
                  exc.verbose_ex)


def print_HP_report_rich(pol):
    verbose_print("", pol.verbose, flush=True)
    if pol.verbose:
        table = Table(title="")
        table.add_column(
            f"Steps in HopfieldPol: {pol.gme.numeig} photonic modes, {np.shape(pol.exc_list)[0]}"
            +
            f" active layers with {pol.exc_list[0].numeig_ex} excitonic modes",
            justify="Left",
            style="cyan",
            no_wrap=True)
        table.add_column("Time (s)", style="magenta")
        table.add_row(
            f"Total time for matrix calculation" +
            f" and diagonalization for {pol.kpoints.shape[1]} k-points",
            f"[u]{pol.total_time:.4f}[/u]",
            style="bold")
        console = Console()
        console.print(table)


def print_HP_report(pol):
    verbose_print("", pol.verbose, flush=True)
    verbose_print(
        f"{pol.total_time:.3f}s total time for matrix calculation" +
        " and diagonalization.", pol.verbose)
