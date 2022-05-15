pydepict
========

A library and accompanying program for parsing SMILES (simplified molecular-input line-entry system) strings and generating 2D depictions of chemical structures from these strings.

It was originally developed as part of a 3rd year computer science dissertation project.

Prerequisites
-------------

pydepict requires **Python 3.7 or newer**, and uses the following packages:

- pygame
- NetworkX

Specific versions of these libraries can be found in ``requirements.txt``.

Setup
-----

The recommended method of installation is via ``pip``, which installs pydepict as a package:

.. code-block:: sh

    python3 -m pip install pydepict

Use ``py -3`` instead of ``python3`` if you are on Windows.

Using ``pip`` installs the latest release of pydepict. Alternatively, if you want the latest development version, then you can clone the repository from GitHub:

.. code-block:: sh

    git clone https://github.com/WilliamWFLee/pydepict
    cd pydepict

If you want to run the program straight from the cloned repository then you must install pydepict's package requirements using ``pip``:

.. code-block:: sh

    python3 -m pip install -r requirements.txt

Or to install the development version as a package:

.. code-block:: sh

    python3 -m pip install .

Usage
-----

If you have installed pydepict as a package, then you can run the program by simply running:

.. code-block:: sh

    pydepict

Or you can invoke the module as a program:

.. code-block:: sh

    python3 -m pydepict

If you want to run the program straight from the cloned repository, then you can execute the script ``src/main.py``:

.. code-block:: sh

    python3 src/main.py

Executing the program without any command-line arguments opens a GUI allowing SMILES string input. Passing a SMILES string as the first command-line argument only displays the renderer window showing only the corresponding diagram. For example:

.. code-block:: sh

    python3 src/main.py CCO

parses the SMILES string ``CCO`` and displays the corresponding diagram.

Minimal Code Example
--------------------

.. code-block:: py

    from pydepict import depict, parse, render

    graph, _ = parse(input("Enter your SMILES string: "))
    positions = depict(graph)
    render(graph, positions)

License
-------

This library is licensed under the MIT Licence. See ``LICENSE`` for license and copyright details. Code files will also include a header explicitly stating that the license and copyright applies.

NetworkX is licensed under the 3-clause BSD license, while pygame is licensed under the GNU Lesser General Public License (LGPL). See ``LICENSE.networkx`` and ``LICENSE.pygame`` respectively for details.
