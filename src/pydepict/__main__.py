#!/usr/bin/env python3

"""
pydepict.__main__

A program for parsing and rendering chemical structures from SMILES strings.

Copyright (c) 2022 William Lee and The University of Sheffield. See LICENSE for details
"""


import argparse
import traceback
from tkinter import Event, Tk
from tkinter.ttk import Button, Entry, Frame, Label

from . import Renderer, depict, parse, show

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

        # Instantiate widgets
        self.frame = Frame(self.root, padding=PADDING)
        self.smiles_input_label = Label(self.frame, text="SMILES")
        self.smiles_input = Entry(self.frame)
        self.smiles_input.bind("<Return>", lambda _: self._show_smiles())
        self.smiles_input.bind("<Control-KeyRelease-a>", self._select_all)
        self.display_button = Button(
            self.frame, text="Display", command=self._show_smiles
        )
        self.error_message = Label(self.frame, foreground="red")

        # Arrange and display widgets
        self.frame.grid()
        self.smiles_input_label.grid(column=0, row=0, padx=PADDING, pady=PADDING)
        self.smiles_input.grid(column=1, row=0, padx=PADDING, pady=PADDING)
        self.display_button.grid(column=0, row=1, columnspan=2, pady=PADDING)
        self.error_message.grid(column=0, row=2, columnspan=2, pady=PADDING)

        # Instantiate renderer
        self.renderer = Renderer()
        self.renderer.show(False)
        # Bind renderer close event to close the program
        self.renderer.on("close", self.root.destroy)

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    @staticmethod
    def _select_all(event: Event):
        event.widget.select_range(0, "end")
        event.widget.icursor("end")

    def _show_smiles(self):
        self.error_message.config(text="")
        smiles = self.smiles_input.get()
        try:
            graph, _ = parse(smiles)
            depict(graph)
            self.renderer.graph = graph
        except Exception as e:
            self.error_message.config(text=f"{e.__class__.__name__}: {str(e)}")
            traceback.print_exc()

    def close(self):
        """
        Close the program.
        """
        # Remove callback to prevent deadlock
        self.renderer.not_on("close", self.root.destroy)
        self.renderer.close()
        self.root.destroy()

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
