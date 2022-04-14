#!/usr/bin/env python3

"""
pydepict.__main__

A program for parsing and rendering chemical structures from SMILES strings.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""


from tkinter import Tk
from tkinter.ttk import Button, Entry, Frame, Label
import traceback

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


def main():
    """
    Shortcut function equivalent to::
        Program().run()
    """
    Program().run()


if __name__ == "__main__":
    main()
