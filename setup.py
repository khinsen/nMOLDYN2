#!/usr/bin/env python

from distutils.core import setup

class Dummy:
    pass
pkginfo = Dummy()
execfile('nMoldyn/__pkginfo__.py', pkginfo.__dict__)                               

setup (name = "nMOLDYN",
       version = pkginfo.__version__,
       description = "Analysis of Molecular Dynamics trajectories",
       long_description =
"""nMOLDYN is an interactive program for the analysis of Molecular
Dynamics simulations. It is especially designed for the computation
and decomposition of neutron scattering spectra. The structure and
dynamics of the simulated systems can be characterized in terms of
various space and time correlation functions. To analyze the dynamics
of complex systems, rigid-body motions of arbitrarily chosen molecular
subunits can be studied.
""",
       author = "T. Rog, K. Murzyn, K. Hinsen, G.R. Kneller",
       author_email = "kneller@cnrs-orleans.fr",
       url = "http://dirac.cnrs-orleans.fr/nMOLDYN/",
       license = "CeCILL",
       packages = ['nMoldyn'],
       scripts = ['xMoldyn', 'pMoldyn', 'dcd_to_nc', 'nc_to_dcd',
                  'dlpoly_to_nc', 'dlpoly3_to_nc'],
       )

