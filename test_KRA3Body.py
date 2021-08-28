from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
import Cluster_Expansion
from KRA3Body import KRA3bodyInteractions
import unittest

class testKRA3body(unittest.TestCase):
    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        a0 = 1.0
        self.crys = crystal.Crystal.FCC(a0, chemistry="A")
        self.chem = 0
        self.jnetFCC = self.crys.jumpnetwork(0, 1.01*a0/np.sqrt(2))
        self.superlatt = 8 * np.eye(3, dtype=int)
        self.superFCC = supercell.ClusterSupercell(self.crys, self.superlatt)
        # get the number of sites in the supercell - should be 8x8x8
        numSites = len(self.superFCC.mobilepos)
        self.vacsite = cluster.ClusterSite((0, 0), np.zeros(3, dtype=int))
        self.vacsiteInd = self.superFCC.index(np.zeros(3, dtype=int), (0, 0))[0]

        # arguments in order
        # sup, jnet, chem, combinedShellRange, nnRange, cutoff, NSpec, Nvac, vacSite
        TScombShellRange = 1 # upto 1nn combined shell
        TSnnRange = 4
        TScutoff = np.sqrt(2)*a0 # 4th nn cutoff
        self.KRAexpander = KRA3bodyInteractions(self.superFCC, self.jnetFCC, self.chem, TScombShellRange, TSnnRange, TScutoff,
                                                self.NSpec, self.Nvac, self.vacsite)

        self.mobOccs = np.zeros((self.NSpec, numSites), dtype=int)
        for site in range(1, numSites):
            spec = np.random.randint(0, self.NSpec - 1)
            self.mobOccs[spec][site] = 1
        self.mobOccs[-1, self.vacsiteInd] = 1
        self.mobCountList = [np.sum(self.mobOccs[i]) for i in range(self.NSpec)]
        print("Done setting up")