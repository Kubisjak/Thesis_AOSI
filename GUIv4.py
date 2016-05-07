# Imports --------------------------------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk
import time
import numpy as np
import matplotlib
from scipy.stats import norm

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # NavigationToolbar2TkAgg

# from matplotlib.figure import Figure
plt.rcParams["toolbar"] = "None"  # Do not display toolbar when calling plot
# ----------------------------------------------------------------------------------------------------------------------


class AsianOptionApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Asian Option Pricing Model")

        GUI = tk.Frame(self, borderwidth=1, padx=10, pady=10)
        GUI.pack()
        GUI.grid_columnconfigure(0, weight=1)
        GUI.grid_columnconfigure(1, weight=1)

        LeftFrame(GUI)
        RightFrame(GUI)
        OutputFrame(GUI)


class LeftFrame:

    def __init__(self, master):
        global frame_rows

        self.frame = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=500)
        self.frame.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        self.frame.rowconfigure(1, weight=1)
        self.frame.rowconfigure(5, weight=1)

        # Parameter Frame Label
        self.frame_position_row = 0
        self.parameter_frame_label = tk.Label(self.frame, text="Parameters: ", justify=tk.LEFT)
        self.parameter_frame_label.grid(row=self.frame_position_row, column=0, columnspan=1, sticky=tk.W)
        self.frame_position_row += 1

        # Parameter frame
        self.parameter_frame = tk.Frame(self.frame, relief=tk.GROOVE, borderwidth=1, padx=10, pady=10)
        self.parameter_frame.grid(row=self.frame_position_row, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        self.parameter_frame_position_row = 0

        S0 = tk.DoubleVar(master)  # Create IntVariable called
        label_S0 = tk.Label(self.parameter_frame, text="S0: ")
        label_S0.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_S0 = tk.Entry(self.parameter_frame, textvariable=S0)
        entry_S0.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        K = tk.DoubleVar(master)  # Create IntVariable
        label_K = tk.Label(self.parameter_frame, text="K: ")
        label_K.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_K = tk.Entry(self.parameter_frame, textvariable=K)
        entry_K.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        sigmaS = tk.DoubleVar(master)  # Create IntVariable
        label_sigmaS = tk.Label(self.parameter_frame, text="SigmaS: ")
        label_sigmaS.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_sigmaS = tk.Entry(self.parameter_frame, textvariable=sigmaS)
        entry_sigmaS.grid(row=self.parameter_frame_position_row, column=1,  sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        nsteps = tk.IntVar(master)  # Create IntVariable
        label_nsteps = tk.Label(self.parameter_frame, text="nsteps: ")
        label_nsteps.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_nsteps = tk.Entry(self.parameter_frame, textvariable=nsteps)
        entry_nsteps.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.parameter_frame_position_row += 1

        nsims = tk.IntVar(master)  # Create IntVariable
        label_nsims = tk.Label(self.parameter_frame, text="nsims: ")
        label_nsims.grid(row=self.parameter_frame_position_row, column=0, sticky=tk.E)
        entry_nsims = tk.Entry(self.parameter_frame, textvariable=nsims)
        entry_nsims.grid(row=self.parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)

        # Drop Down List
        self.frame_position_row += 1
        label_option_list_processes = tk.Label(self.frame, text="Stochastic interest process: ", justify=tk.LEFT)
        label_option_list_processes.grid(row=self.frame_position_row, column=0, columnspan=1, sticky=tk.W)

        self.frame_position_row += 1
        interest_stochastic_process_variable = tk.StringVar(master)
        interest_stochastic_process_variable.set("Ornstein-Uhlenbeck")  # default value

        self.frame_position_row += 1
        option_list_processes = tk.OptionMenu(self.frame, interest_stochastic_process_variable,
                                              "Ornstein-Uhlenbeck", "Vasicek", "Hull-White")
        option_list_processes.configure(anchor="w")
        option_list_processes.grid(row=self.frame_position_row, sticky=tk.W + tk.E)

        # Stochastic interest process parameter frame - SIP
        self.frame_position_row += 1
        self.SIP_parameter_frame = tk.Frame(self.frame, relief=tk.GROOVE, borderwidth=1, padx=10, pady=10)
        self.SIP_parameter_frame.grid(row=self.frame_position_row, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        self.SIP_parameter_frame_position_row = 0
        R0 = tk.DoubleVar(master)  # Create IntVariable
        label_R0 = tk.Label(self.SIP_parameter_frame, text="R0: ")
        label_R0.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_R0 = tk.Entry(self.SIP_parameter_frame, textvariable=R0)
        entry_R0.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        sigmaR = tk.DoubleVar(master)  # Create IntVariable
        label_sigmaR = tk.Label(self.SIP_parameter_frame, text="SigmaR: ")
        label_sigmaR.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_sigmaR = tk.Entry(self.SIP_parameter_frame, textvariable=sigmaR)
        entry_sigmaR.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        gamma = tk.DoubleVar(master)  # Create IntVariable
        label_gamma = tk.Label(self.SIP_parameter_frame, text="Gamma: ")
        label_gamma.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_gamma = tk.Entry(self.SIP_parameter_frame, textvariable=gamma)
        entry_gamma.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        rho = tk.DoubleVar(master)  # Create IntVariable
        label_rho = tk.Label(self.SIP_parameter_frame, text="Rho: ")
        label_rho.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_rho = tk.Entry(self.SIP_parameter_frame, textvariable=rho)
        entry_rho.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        alpha = tk.DoubleVar(master)  # Create IntVariable
        label_alpha = tk.Label(self.SIP_parameter_frame, text="Alpha: ")
        label_alpha.grid(row=self.SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_alpha = tk.Entry(self.SIP_parameter_frame, textvariable=alpha)
        entry_alpha.grid(row=self.SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        self.SIP_parameter_frame_position_row += 1

        # Total rows used
        # total_rows_left = self.parameter_frame_position_row + self.frame_left_position_row

        frame_rows = self.frame_position_row

class RightFrame:

    def __init__(self, master):
        global frame_rows
        self.frame = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=500)
        self.frame.grid(row=0, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

        # Right Frame ----------------------------------------------------------------------------------------------
        # Animation canvas frame
        canvas_rowspan = frame_rows

        canvas_frame = tk.Frame(self.frame)
        canvas_frame.grid(row=0, rowspan=canvas_rowspan, column=0, columnspan=2)
        f = plt.figure(figsize=(2, 2), dpi=150)
        canvas = FigureCanvasTkAgg(f, master=canvas_frame)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        # canvas.tkcanvas.pack(side=tk.TOP, fill=tk.BOTH)

        button_quit = ttk.Button(self.frame, text="Quit", command=lambda: self._quit(master))
        button_quit.grid(row=canvas_rowspan + 1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        button_calculate = ttk.Button(self.frame, text="Calculate", command=lambda: self._quit(master))
        button_calculate.grid(row=canvas_rowspan + 1, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

    def _quit(self, master):
        master.quit()
        master.destroy()


class OutputFrame:
    def __init__(self, master):
        self.frame_bottom_position_row = 0

        frame_bottom = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=100)
        frame_bottom.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)

        label_bottom_frame = tk.Label(frame_bottom, text="Output: ", justify=tk.LEFT)
        label_bottom_frame.grid(row=self.frame_bottom_position_row, column=0, columnspan=1, sticky=tk.W)


app = AsianOptionApp()
app.mainloop()
