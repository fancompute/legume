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
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def verbose_print(text, verbose, flush=False, end='\n'):
    """Print if verbose_ex==True
            """
    if verbose == True:
        if flush == False:
            print(text, end=end)
        else:
            sys.stdout.write("\r" + text)
            sys.stdout.flush()


def print_GME_report(gme):
    if RICH_AVAILABLE:
        verbose_print("", gme.verbose, flush=True)
        if gme.verbose:
            table = Table(title="")
            table.add_column("Process",
                             justify="Left",
                             style="cyan",
                             no_wrap=True)
            table.add_column("Time (s)", style="magenta")
            table.add_column("% vs total T", justify="right", style="green")

            table.add_row(
                f"Guided modes computation using the gmode_compute='{gme.gmode_compute.lower()}' method",
                f"{gme.t_guided:.3f}",
                f"{gme.t_guided/gme.total_time*100:.0f}%")
            table.add_row("Inverse matrix of Fourier-space permittivity",
                          f"{gme.t_eps_inv:.3f}",
                          f"{gme.t_eps_inv/gme.total_time*100:.0f}%")
            table.add_row(
                f"Matrix diagionalization using the '{gme.eig_solver.lower()}' solver",
                f"{(gme.t_eig-gme.t_symmetry):.3f}",
                f"{(gme.t_eig-gme.t_symmetry)/gme.total_time*100:.0f}%")
            table.add_row("For creating GME matrix",
                          f"{gme.t_creat_mat:.3f}",
                          f"{gme.t_creat_mat/gme.total_time*100:.0f}%",
                          end_section=not gme.kz_symmetry)
            if gme.kz_symmetry:
                if gme.use_sparse == True:
                    str_mat_used = "sparse"
                elif gme.use_sparse == False:
                    str_mat_used = "dense"
                table.add_row(
                    f"For creating change of basis matrix and multiply it using {str_mat_used} matrices",
                    f"{gme.t_symmetry:.3f}",
                    f"{gme.t_symmetry/gme.total_time*100:.0f}%",
                    end_section=True)
            table.add_row("Total time for real part of frequencies",
                          f"{gme.total_time:.3f}",
                          f"{100}%",
                          style="bold",
                          end_section=True)

            console = Console()
            console.print(table)

    else:
        verbose_print("", gme.verbose, flush=True)
        verbose_print(
            f"{gme.total_time:.3f}s total time for real part of frequencies, of which",
            gme.verbose)
        verbose_print(
            f"  {gme.t_guided:.3f}s ({gme.t_guided/gme.total_time*100:.0f}%) for guided modes computation using"
            f" the gmode_compute='{gme.gmode_compute.lower()}' method",
            gme.verbose)
        verbose_print(
            f"  {gme.t_eps_inv:.3f}s ({gme.t_eps_inv/gme.total_time*100:.0f}%) for inverse matrix of Fourier-space "
            f"permittivity", gme.verbose)
        verbose_print(
            f"  {(gme.t_eig-gme.t_symmetry):.3f}s ({(gme.t_eig-gme.t_symmetry)/gme.total_time*100:.0f}%) for matrix diagionalization using "
            f"the '{gme.eig_solver.lower()}' solver", gme.verbose)
        verbose_print(
            f"  {gme.t_creat_mat:.3f}s ({gme.t_creat_mat/gme.total_time*100:.0f}%) for creating GME matrix",
            gme.verbose)

        if gme.kz_symmetry:
            if gme.use_sparse == True:
                str_mat_used = "sparse"
            elif gme.use_sparse == False:
                str_mat_used = "dense"
            verbose_print(
                f"  {gme.t_symmetry:.3f}s ({gme.t_symmetry/gme.total_time*100:.0f}%) for creating change of basis matrix and multiply it"
                + f" using {str_mat_used} matrices", gme.verbose)


def print_GME_im_report(gme):
    if RICH_AVAILABLE:
        verbose_print("", gme.verbose, flush=True)
        if gme.verbose:
            table = Table(title="")
            table.add_column("Process",
                             justify="Left",
                             style="cyan",
                             no_wrap=True)
            table.add_column("Time (s)", style="magenta")
            table.add_row(f"Total time for imaginary part of frequencies",
                          f"{(gme.t_imag):.3f}",
                          style="bold")
            console = Console()
            console.print(table)

    else:
        verbose_print("", gme.verbose, flush=True)
        verbose_print(
            f"{(gme.t_imag):.3f}s  total time for imaginary part"
            " of frequencies", gme.verbose)


def print_EXC_report(exc):
    if RICH_AVAILABLE:
        verbose_print("", exc.verbose_ex, flush=True)
        if exc.verbose_ex:
            table = Table(title="")
            table.add_column("Process",
                             justify="Left",
                             style="cyan",
                             no_wrap=True)
            table.add_column("Time (s)", style="magenta")
            table.add_column("% vs total T", justify="right", style="green")
            table.add_row(f"Diagonalization of the Hamiltonian",
                          f"{exc.t_eig:.4f}",
                          f"{exc.t_eig/exc.total_time*100:.0f}%",
                          end_section=True)
            table.add_row(f"Total time for excitonic energies",
                          f"{exc.total_time:.4f}",
                          "100%",
                          style="bold")
            console = Console()
            console.print(table)

    else:
        verbose_print("", exc.verbose_ex, flush=True)
        verbose_print(
            f"{exc.total_time:.4f}s total time for excitonic energies, of which",
            exc.verbose_ex)
        verbose_print(
            f"  {exc.t_eig:.4f}s for diagonalization of the Hamiltonian",
            exc.verbose_ex)


def print_HOP_report(pol):
    if RICH_AVAILABLE:
        verbose_print("", pol.verbose, flush=True)
        if pol.verbose:
            table = Table(title="")
            table.add_column("Process",
                             justify="Left",
                             style="cyan",
                             no_wrap=True)
            table.add_column("Time (s)", style="magenta")
            table.add_row(f"Total time for Hopfield matrix calculation" +
                          " and diagonalization.",
                          f"{pol.total_time:.4f}",
                          style="bold")
            console = Console()
            console.print(table)

    else:
        verbose_print("", pol.verbose, flush=True)
        verbose_print(
            f"{total_time:.3f}s total time for Hopfield matrix calculation" +
            " and diagonalization.", pol.verbose)
