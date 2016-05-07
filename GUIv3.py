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
LARGE_FONT = ("Verdana", 12)


class AsianOptionPricingApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        GUI = tk.Frame(self, borderwidth=1, padx=10, pady=10)
        GUI.pack()

        GUI.grid_columnconfigure(0, weight=1)
        GUI.grid_columnconfigure(1, weight=1)

        # Configure grid ---------------------------------------------------------------------------------------------

        GUI.grid_columnconfigure(0, weight=1)
        GUI.grid_columnconfigure(1, weight=1)


class frame_left(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # Create two frames that will host on the left side the parameters frame and on the right side the canvas frame

        frame_left = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=500)
        frame_left.grid(row=0, column=0, sticky=tk.W + tk.E + tk.N + tk.S)
        frame_left.rowconfigure(1, weight=1)  # Used to stretch the frame placed on row 1 to the size of frame_left

        frame_right = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=500)
        frame_right.grid(row=0, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

        frame_left_position_row = 0
        parameter_frame_position_row = 0

        # Left frame --------------------------------------------------------------------------------------------------
        # Label
        label_parameter_frame = tk.Label(frame_left, text="Parameters: ",
                                         justify=tk.LEFT)
        label_parameter_frame.grid(row=frame_left_position_row, column=0, columnspan=1, sticky=tk.W)
        frame_left_position_row += 1

        # Parameter frame
        parameter_frame = tk.Frame(frame_left, relief=tk.GROOVE, borderwidth=1, padx=10, pady=10)
        parameter_frame.grid(row=frame_left_position_row, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        S0 = tk.DoubleVar(master)  # Create IntVariable called
        label_S0 = tk.Label(parameter_frame, text="S0: ")
        label_S0.grid(row=parameter_frame_position_row, column=0, sticky=tk.E)
        entry_S0 = tk.Entry(parameter_frame, textvariable=S0)
        entry_S0.grid(row=parameter_frame_position_row, column=1,
                      sticky=tk.W + tk.E + tk.S + tk.N)
        parameter_frame_position_row += 1

        K = tk.DoubleVar(master)  # Create IntVariable
        label_K = tk.Label(parameter_frame, text="K: ")
        label_K.grid(row=parameter_frame_position_row, column=0, sticky=tk.E)
        entry_K = tk.Entry(parameter_frame, textvariable=K)
        entry_K.grid(row=parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        parameter_frame_position_row += 1

        sigmaS = tk.DoubleVar(master)  # Create IntVariable
        label_sigmaS = tk.Label(parameter_frame, text="SigmaS: ")
        label_sigmaS.grid(row=parameter_frame_position_row, column=0, sticky=tk.E)
        entry_sigmaS = tk.Entry(parameter_frame, textvariable=sigmaS)
        entry_sigmaS.grid(row=parameter_frame_position_row, column=1,
                          sticky=tk.W + tk.E + tk.S + tk.N)
        parameter_frame_position_row += 1

        nsteps = tk.IntVar(master)  # Create IntVariable
        label_nsteps = tk.Label(parameter_frame, text="nsteps: ")
        label_nsteps.grid(row=parameter_frame_position_row, column=0, sticky=tk.E)
        entry_nsteps = tk.Entry(parameter_frame, textvariable=nsteps)
        entry_nsteps.grid(row=parameter_frame_position_row, column=1,
                          sticky=tk.W + tk.E + tk.S + tk.N)
        parameter_frame_position_row += 1

        nsims = tk.IntVar(master)  # Create IntVariable
        label_nsims = tk.Label(parameter_frame, text="nsims: ")
        label_nsims.grid(row=parameter_frame_position_row, column=0, sticky=tk.E)
        entry_nsims = tk.Entry(parameter_frame, textvariable=nsims)
        entry_nsims.grid(row=parameter_frame_position_row, column=1,
                         sticky=tk.W + tk.E + tk.S + tk.N)

        # Drop Down List
        frame_left_position_row += 1
        label_option_list_processes = tk.Label(frame_left, text="Stochastic interest process: ",
                                               justify=tk.LEFT)
        label_option_list_processes.grid(row=frame_left_position_row, column=0, columnspan=1,
                                         sticky=tk.W)

        frame_left_position_row += 1
        interest_stochastic_process_variable = tk.StringVar(GUI)
        interest_stochastic_process_variable.set("Ornstein-Uhlenbeck")  # default value

        frame_left_position_row += 1
        option_list_processes = tk.OptionMenu(frame_left, interest_stochastic_process_variable,
                                              "Ornstein-Uhlenbeck", "Vasicek", "Hull-White")
        option_list_processes.configure(anchor="w")
        option_list_processes.grid(row=frame_left_position_row, sticky=tk.W + tk.E)

        # Stochastic interest process parameter frame - SIP
        frame_left_position_row += 1
        SIP_parameter_frame = tk.Frame(frame_left, relief=tk.GROOVE, borderwidth=1, padx=10, pady=10)
        SIP_parameter_frame.grid(row=frame_left_position_row, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        SIP_parameter_frame_position_row = 0
        R0 = tk.DoubleVar(GUI)  # Create IntVariable
        label_R0 = tk.Label(SIP_parameter_frame, text="R0: ")
        label_R0.grid(row=SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_R0 = tk.Entry(SIP_parameter_frame, textvariable=R0)
        entry_R0.grid(row=SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        SIP_parameter_frame_position_row += 1

        sigmaR = tk.DoubleVar(GUI)  # Create IntVariable
        label_sigmaR = tk.Label(SIP_parameter_frame, text="SigmaR: ")
        label_sigmaR.grid(row=SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_sigmaR = tk.Entry(SIP_parameter_frame, textvariable=sigmaR)
        entry_sigmaR.grid(row=SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        SIP_parameter_frame_position_row += 1

        gamma = tk.DoubleVar(GUI)  # Create IntVariable
        label_gamma = tk.Label(SIP_parameter_frame, text="Gamma: ")
        label_gamma.grid(row=SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_gamma = tk.Entry(SIP_parameter_frame, textvariable=gamma)
        entry_gamma.grid(row=SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        SIP_parameter_frame_position_row += 1

        rho = tk.DoubleVar(GUI)  # Create IntVariable
        label_rho = tk.Label(SIP_parameter_frame, text="Rho: ")
        label_rho.grid(row=SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_rho = tk.Entry(SIP_parameter_frame, textvariable=rho)
        entry_rho.grid(row=SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        SIP_parameter_frame_position_row += 1

        alpha = tk.DoubleVar(GUI)  # Create IntVariable
        label_alpha = tk.Label(SIP_parameter_frame, text="Alpha: ")
        label_alpha.grid(row=SIP_parameter_frame_position_row, column=0, sticky=tk.E)
        entry_alpha = tk.Entry(SIP_parameter_frame, textvariable=alpha)
        entry_alpha.grid(row=SIP_parameter_frame_position_row, column=1, sticky=tk.W + tk.E + tk.S + tk.N)
        SIP_parameter_frame_position_row += 1

        # Total rows used
        total_rows_left = parameter_frame_position_row + frame_left_position_row

        # Right Frame -------------------------------------------------------------------------------------------------
        # Animation canvas frame
        canvas_rowspan = parameter_frame_position_row

        canvas_frame = tk.Frame(frame_right)
        canvas_frame.grid(row=0, rowspan=canvas_rowspan, column=0, columnspan=2)
        f = plt.figure(figsize=(2, 2), dpi=150)
        canvas = FigureCanvasTkAgg(f, master=canvas_frame)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
        # canvas.tkcanvas.pack(side=tk.TOP, fill=tk.BOTH)

        button_quit = ttk.Button(frame_right, text="Quit", command=self._quit)
        button_quit.grid(row=canvas_rowspan + 1, column=0, sticky=tk.W + tk.E + tk.N + tk.S)

        button_calculate = ttk.Button(frame_right, text="Calculate", command=self._quit)
        button_calculate.grid(row=canvas_rowspan + 1, column=1, sticky=tk.W + tk.E + tk.N + tk.S)

        # Output Frame ------------------------------------------------------------------------------------------------
        frame_bottom_position_row = 0

        frame_bottom = tk.Frame(GUI, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=100)
        frame_bottom.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)

        label_bottom_frame = tk.Label(frame_bottom, text="Output: ", justify=tk.LEFT)
        label_bottom_frame.grid(row=frame_bottom_position_row, column=0, columnspan=1, sticky=tk.W)

    def _quit(self):
        self.quit()
        self.destroy()


class OutputFrame:
    def __init__(self, master):
        self.frame_bottom_position_row = 0

        frame_bottom = tk.Frame(master, borderwidth=1, padx=10, pady=10, relief=tk.GROOVE, height=100)
        frame_bottom.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)

        label_bottom_frame = tk.Label(frame_bottom, text="Output: ", justify=tk.LEFT)
        label_bottom_frame.grid(row=self.frame_bottom_position_row, column=0, columnspan=1, sticky=tk.W)

app = AsianOptionPricingApp()
app.mainloop()
