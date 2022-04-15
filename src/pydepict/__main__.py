#!/usr/bin/env python3

"""
pydepict.__main__

A program for parsing and rendering chemical structures from SMILES strings.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""


import argparse
import traceback
from tkinter import Tk
from tkinter.ttk import Button, Entry, Frame, Label

from . import show

PADDING = 10

__all__ = ["Program", "main"]


class Program:
    """
    Class representing a program for accepting SMILES input,
    and displaying the corresponding diagram.

    The program is built using :module:`tkinter`.
    """

    def __init__(self):
        self.root = Tk()
        self.root.wm_title("pydepict")

        self.frame = Frame(self.root, padding=PADDING)
        self.smiles_input_label = Label(self.frame, text="SMILES")
        self.smiles_input = Entry(self.frame)
        self.smiles_input.bind("<Return>", lambda _: self._show_smiles())
        self.display_button = Button(
            self.frame, text="Display", command=self._show_smiles
        )
        self.error_message = Label(self.frame, foreground="red")

        self.frame.grid()
        self.smiles_input_label.grid(column=0, row=0, padx=PADDING, pady=PADDING)
        self.smiles_input.grid(column=1, row=0, padx=PADDING, pady=PADDING)
        self.display_button.grid(column=0, row=1, columnspan=2, pady=PADDING)
        self.error_message.grid(column=0, row=2, columnspan=2, pady=PADDING)

    def _show_smiles(self):
        self.error_message.config(text="")
        self.root.withdraw()
        try:
            show(self.smiles_input.get())
        except Exception as e:
            self.error_message.config(text=f"{e.__class__.__name__}: {str(e)}")
            traceback.print_exc()
        self.root.deiconify()

    def run(self):
        """
        Run the program.
        """
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", nargs="?", default=None)

    return parser.parse_args()


def main():
    """
    Runs the standalone program.

    If the calling script is run with no arguments, then a dialog is shown,
    allowing entry of SMILES strings multiple times.

    If a SMILES string is passed as the first and only argument,
    then only the renderer window for that SMILES string is shown.
    """
    args = parse_args()
    if args.smiles is None:
        Program().run()
    else:
        show(args.smiles)


if __name__ == "__main__":
    main()
