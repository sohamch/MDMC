from onsager import crystal, supercell, cluster
import numpy as np
import collections
import itertools
from KRA3Body import KRA3bodyInteractions
import unittest

class testKRA3bodyFCC(unittest.TestCase):
    def setUp(self):
        self.NSpec = 3
        self.Nvac = 1
        a0 = 1.0
        self.a0=a0
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

    def test_clusterGroups(self):
        for key, item in self.KRAexpander.TransGroupsNN.items():
            self.assertEqual(len(item), 4)
            self.assertEqual(len(item[1]), 4, msg="{}".format(len(item[1])))  # 1nn to both
            self.assertEqual(len(item[2]), 4)  # 2nn to either
            self.assertEqual(len(item[3]), 8)  # 3nn to either
            self.assertEqual(len(item[4]), 2)  # 4nn to either

            # Now let's check some explictly
            for clSite in item[1]:
                # this is where we have nn to both
                x02 = np.dot(self.crys.lattice, clSite.R - key[0].R)
                x12 = np.dot(self.crys.lattice, clSite.R - key[1].R)
                self.assertAlmostEqual(np.linalg.norm(x02), np.linalg.norm(x12))
                self.assertAlmostEqual(np.linalg.norm(x02), np.linalg.norm(self.crys.lattice[0]))

            for clSite in item[2]:
                x02 = np.linalg.norm(np.dot(self.crys.lattice, clSite.R - key[0].R))
                x12 = np.linalg.norm(np.dot(self.crys.lattice, clSite.R - key[1].R))
                self.assertFalse(np.abs(x02-x12) < 1e-8, msg="{} {}".format(x02, x12))

                nnDist = np.linalg.norm(self.crys.lattice[0])
                if np.math.isclose(x02, nnDist):
                    self.assertAlmostEqual(x12, self.a0)
                elif np.math.isclose(x12, nnDist):
                    self.assertAlmostEqual(x02, self.a0)

            for clSite in item[3]:
                x02 = np.linalg.norm(np.dot(self.crys.lattice, clSite.R - key[0].R))
                x12 = np.linalg.norm(np.dot(self.crys.lattice, clSite.R - key[1].R))
                self.assertFalse(np.abs(x02-x12) < 1e-8, msg="{} {}".format(x02, x12))

                nnDist = np.linalg.norm(self.crys.lattice[0])
                if np.math.isclose(x02, nnDist):
                    self.assertAlmostEqual(x12, np.sqrt(1.5)*self.a0)
                elif np.math.isclose(x12, nnDist):
                    self.assertAlmostEqual(x02, np.sqrt(1.5)*self.a0)

            for clSite in item[4]:
                x02 = np.linalg.norm(np.dot(self.crys.lattice, clSite.R - key[0].R))
                x12 = np.linalg.norm(np.dot(self.crys.lattice, clSite.R - key[1].R))
                self.assertFalse(np.abs(x02-x12) < 1e-8, msg="{} {}".format(x02, x12))

                nnDist = np.linalg.norm(self.crys.lattice[0])
                if np.math.isclose(x02, nnDist):
                    self.assertAlmostEqual(x12, np.sqrt(2)*self.a0)
                elif np.math.isclose(x12, nnDist):
                    self.assertAlmostEqual(x02, np.sqrt(2)*self.a0)

    def test_SpeciesAssignment(self):
        self.assertEqual(len(self.KRAexpander.clusterSpeciesJumps), 24)
        specs = collections.defaultdict(list)
        for (key, item) in self.KRAexpander.clusterSpeciesJumps.items():
            specs[(key[0], key[1])].append(key[2])

        self.assertEqual(len(specs), 12)
        specs.default_factory = None
        for key, item in specs.items():
            self.assertEqual(sorted(item), [0, 1], msg="{}".format(item))

    def test_Jit_data(self):
        for Ind, jump in self.KRAexpander.Index2Jump.items():
            self.assertEqual(self.KRAexpander.jump2Index[jump], Ind)

        # Generate the JIT arrays

        # Let's make energies based on jumping atoms.
        # Ni = 0, Re = 1

        EnReJumps = np.array([4, 3, 2, 1])
        EnNiJumps = EnReJumps*2

        Energies = []

        for jumpInd in range(len(self.KRAexpander.jump2Index)):
            jumpkey = self.KRAexpander.Index2Jump[jumpInd]
            jumpSpec = jumpkey[2]
            if jumpSpec == 0:
                Energies.append(EnNiJumps.copy())
            else:
                Energies.append(EnReJumps.copy())

        CounterSpec = 1
        TsInteractIndexDict, Index2TSinteractDict, numSitesTSInteracts, TSInteractSites, TSInteractSpecs, \
        jumpFinSites, jumpFinSpec, FinSiteFinSpecJumpInd, numJumpPointGroups, numTSInteractsInPtGroups, \
        JumpInteracts, Jump2KRAEng =\
        self.KRAexpander.makeTransJitData(CounterSpec, Energies)

        self.assertEqual(len(TsInteractIndexDict), 18*24)

        # Next, we want to test if the correct sites and specs have been stored
        # and if the correct energies have been assigned to the interactions

        for (jumpkey, TSptGrps) in self.KRAexpander.clusterSpeciesJumps.items():
            jumpInd = self.KRAexpander.jump2Index[jumpkey]
            FinSite = jumpkey[1]
            FinSpec = jumpkey[2]
            # Check that the correct initial and final states have been stored
            self.assertEqual(jumpFinSites[jumpInd], FinSite)
            self.assertEqual(jumpFinSpec[jumpInd], FinSpec)

            # Check that the correct number of point groups are stored
            NptGrp = len(TSptGrps)
            self.assertEqual(numJumpPointGroups[jumpInd], NptGrp)

            # Check that in for each point group, the correct interactions are stored.
            for TsPtGpInd, (type, TSinteractList) in zip(itertools.count(), TSptGrps.items()):
                self.assertEqual(numTSInteractsInPtGroups[jumpInd, TsPtGpInd], len(TSinteractList))
                specList = [self.NSpec - 1, FinSpec] + [CounterSpec]  # only Re occupancy is going to be checked
                for interactInd, TSClust in enumerate(TSinteractList):
                    interact = tuple([(self.KRAexpander.sup.index(site.R, site.ci)[0], spec)
                                      for site, spec in zip(TSClust.sites, specList)])
                    interactStored = Index2TSinteractDict[JumpInteracts[jumpInd, TsPtGpInd, interactInd]]

                    self.assertEqual(set(interact), set(interactStored))
                    self.assertEqual(Jump2KRAEng[jumpInd, TsPtGpInd, interactInd], Energies[jumpInd][TsPtGpInd])

