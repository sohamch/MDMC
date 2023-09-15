import sys

import scipy.linalg

sys.path.append("../")

from onsager import crystal, supercell, cluster
import numpy as np
import scipy.linalg as spla
import Cluster_Expansion
import pickle
import h5py
import unittest

from LBAM_DataSet import *

cp = "../../CrysDat_FCC/"
class Test_HEA_LBAM(unittest.TestCase):

    def setUp(self):
        self.DataPath = ("testData_HEA_MEAM_orthogonal.h5")
        self.CrysDatPath = (cp + "CrystData_ortho_5_cube.h5")
        self.a0 = 3.595

        self.remap = True

        self.jList, self.dxList, self.superCell, self.vacsite, self.vacsiteInd, self.siteMap_nonPrimitive_to_primitive =\
            Load_crys_Data(self.CrysDatPath, ReduceToPrimitve=True)

        self.state1List, self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects = \
            Load_Data(self.DataPath, self.siteMap_nonPrimitive_to_primitive)

        self.assertTrue(self.siteMap_nonPrimitive_to_primitive is not None)

        self.crys = self.superCell.crys
        print(self.crys)

        print("Sites in Dataset: ", self.state1List.shape[1])
        print("Sites in supercell: ", len(self.superCell.mobilepos))
        self.AllSpecs = np.unique(self.state1List[0])
        self.NSpec = self.AllSpecs.shape[0]
        self.vacSpec = self.state1List[0, 0]
        self.SpecExpand = 5
        print("All Species: {}".format(self.AllSpecs))
        print("Vacancy Species: {}. Vacancy site: {}".format(self.vacSpec, self.vacsiteInd))
        print("Expanding Species: {}".format(self.SpecExpand))

        self.ClustCut = 1.01
        self.MaxOrder = 2

        self.VclusExp = makeVClusExp(self.superCell, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=False)

        self.JitExpander, self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.NVclus = CreateJitCalculator(self.VclusExp, self.NSpec,
                                                                                                scratch=True,
                                                                                                save=True)

        # Now re-make the same calculator by loading from saved h5 file
        self.JitExpander_load, self.numVecsInteracts_load, self.VecsInteracts_load, self.VecGroupInteracts_load,\
        self.NVclus_load = CreateJitCalculator(self.VclusExp, self.NSpec, scratch=False)

        self.VclusExp_all = makeVClusExp(self.superCell, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=True)

        self.JitExpander_all, self.numVecsInteracts_all, self.VecsInteracts_all, self.VecGroupInteracts_all,\
        self.NVclus_all = CreateJitCalculator(self.VclusExp_all, self.NSpec, scratch=True, save=False)

    def test_crystal_data(self):

        # Load the original data
        with h5py.File(self.DataPath, "r") as fl:
            state1List = np.array(fl["InitStates"])
            dispList = np.array(fl["SpecDisps"])
            rateList = np.array(fl["rates"])
            AllJumpRates = np.array(fl["AllJumpRates_Init"])
            jmpSelects = np.array(fl["JumpSelects"])

        # check that rates and displacements haven't been changed
        # self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects
        self.assertTrue(np.array_equal(self.dispList, dispList))
        self.assertTrue(np.array_equal(self.rateList, rateList))
        self.assertTrue(np.array_equal(self.AllJumpRates, AllJumpRates))
        self.assertTrue(np.array_equal(self.jumpSelects, jmpSelects))

        # Load the original crystal data
        with h5py.File(self.CrysDatPath, "r") as fl:
            lattice = np.array(fl["Lattice_basis_vectors"])
            superlatt = np.array(fl["SuperLatt"])

            try:
                basis_sites = np.array(fl["basis_sites"])
                basis = [[b for b in basis_sites]]
            except KeyError:
                basis = [[np.array([0., 0., 0.])]]

            dxList = np.array(fl["dxList_1nn"])
            NNList = np.array(fl["NNsiteList_sitewise"])

        # Create the original supercell
        crys = crystal.Crystal(lattice=lattice, basis=basis, chemistry=["A"], noreduce=self.remap)
        superCell = supercell.ClusterSupercell(crys, superlatt)

        # Match the cartesian coordinates after the mapping
        superCell_prim = self.VclusExp.sup
        crys_prim = self.VclusExp.sup.crys


        # Check the vacancy nearest neighbors
        self.assertTrue(np.array_equal(self.dxList, dxList))
        print("Original vacancy neighbors: ", NNList[1:, 0])
        print("Simulation vacancy neighbors: ", self.jList)

        for jumpInd in range(dxList.shape[0]):
            dxJump = self.dxList[jumpInd]
            Rsite, ciSite = crys_prim.cart2pos(dxJump)
            self.assertTrue(ciSite == (0, 0))
            siteInd_primitive = superCell_prim.index(Rsite, ciSite)[0]
            self.assertEqual(self.jList[jumpInd], siteInd_primitive)

            Rsite_original, ciSite_original = superCell.crys.cart2pos(dxJump)
            siteInd_original = superCell.index(Rsite_original, ciSite_original)[0]
            self.assertEqual(NNList[1 + jumpInd, 0], siteInd_original)

        if not self.remap:
            print("Testing no remapping")
            self.assertTrue(np.array_equal(self.jList, NNList[1:, 0]))
            self.assertTrue(self.siteMap_nonPrimitive_to_primitive is None)

            for stateInd in range(state1List.shape[0]):
                state = state1List[stateInd]
                state_mapped = self.state1List[stateInd]

                self.assertTrue(np.array_equal(state, state_mapped),
                                 msg="state {} failed\nSite map:\n{}".format(stateInd, self.siteMap_nonPrimitive_to_primitive))

        # check the re-indexing of the data
        # We'll get the cartesian position from the original supercell
        # Then we'll check it matches up with the position from the primitive supercell
        else:
            print("Testing remapping")
            self.assertTrue(self.siteMap_nonPrimitive_to_primitive is not None)

            # check that the site positions are consistent
            for siteInd_primitive in range(self.siteMap_nonPrimitive_to_primitive.shape[0]):
                ci_primitive, R_primitive = superCell_prim.ciR(siteInd_primitive)
                xPrimitive = superCell_prim.crys.pos2cart(R_primitive, ci_primitive)

                siteInd_original = self.siteMap_nonPrimitive_to_primitive[siteInd_primitive]
                ci_original, R_original = superCell.ciR(siteInd_original)
                xOriginal = superCell.crys.pos2cart(R_original, ci_original)

                self.assertTrue(np.allclose(xOriginal, xPrimitive, rtol=0, atol=1e-8))

            for stateInd in range(state1List.shape[0]):
                state = state1List[stateInd]
                state_mapped = self.state1List[stateInd]

                self.assertTrue(state.shape[0] == state_mapped.shape[0] == self.VclusExp.Nsites)

                self.assertFalse(np.array_equal(state, state_mapped),
                                 msg="state {} failed\nSite map:\n{}".format(stateInd, self.siteMap_nonPrimitive_to_primitive))

                self.assertTrue(np.array_equal(state_mapped, state[self.siteMap_nonPrimitive_to_primitive]))

                # First check that the composition is preserved
                specs_1, counts_1 = np.unique(state, return_counts=True)
                specs_2, counts_2 = np.unique(state_mapped, return_counts=True)
                self.assertTrue(np.array_equal(specs_1, specs_2))
                self.assertTrue(np.array_equal(counts_1, counts_2))

                # Then check that the occupancies of the sites are consistent
                self.assertTrue(self.siteMap_nonPrimitive_to_primitive is not None)
                for siteInd_primitive in range(self.VclusExp.Nsites):
                    siteInd_original = self.siteMap_nonPrimitive_to_primitive[siteInd_primitive]
                    if siteInd_primitive == 0:  # check that the vacancy is at 0 in both cases
                        self.assertEqual(siteInd_original, 0)

                    self.assertEqual(state[siteInd_original], state_mapped[siteInd_primitive])

            print("Tested {} states. Last site checked: {}".format(stateInd + 1, siteInd_primitive))

    def test_CreateJitCalculator(self):
        # This is to check whether the Jit arrays have been properly stored
        with h5py.File("JitArrays.h5", "r") as fl:
            numSitesInteracts = np.array(fl["numSitesInteracts"])
            self.assertTrue(np.array_equal(self.JitExpander.numSitesInteracts, numSitesInteracts))
            self.assertTrue(np.array_equal(self.JitExpander_load.numSitesInteracts, numSitesInteracts))

            SupSitesInteracts = np.array(fl["SupSitesInteracts"])
            self.assertTrue(np.array_equal(self.JitExpander.SupSitesInteracts, SupSitesInteracts))
            self.assertTrue(np.array_equal(self.JitExpander_load.SupSitesInteracts, SupSitesInteracts))

            SpecOnInteractSites = np.array(fl["SpecOnInteractSites"])
            self.assertTrue(np.array_equal(self.JitExpander.SpecOnInteractSites, SpecOnInteractSites))
            self.assertTrue(np.array_equal(self.JitExpander_load.SpecOnInteractSites, SpecOnInteractSites))

            numInteractsSiteSpec = np.array(fl["numInteractsSiteSpec"])
            self.assertTrue(np.array_equal(self.JitExpander.numInteractsSiteSpec, numInteractsSiteSpec))
            self.assertTrue(np.array_equal(self.JitExpander_load.numInteractsSiteSpec, numInteractsSiteSpec))

            SiteSpecInterArray = np.array(fl["SiteSpecInterArray"])
            self.assertTrue(np.array_equal(self.JitExpander.SiteSpecInterArray, SiteSpecInterArray))
            self.assertTrue(np.array_equal(self.JitExpander_load.SiteSpecInterArray, SiteSpecInterArray))

            numVecsInteracts = np.array(fl["numVecsInteracts"])
            self.assertTrue(np.array_equal(self.numVecsInteracts, numVecsInteracts))
            self.assertTrue(np.array_equal(self.numVecsInteracts_load, numVecsInteracts))

            VecsInteracts = np.array(fl["VecsInteracts"])
            self.assertTrue(np.array_equal(self.VecsInteracts, VecsInteracts))
            self.assertTrue(np.array_equal(self.VecsInteracts_load, VecsInteracts))

            VecGroupInteracts = np.array(fl["VecGroupInteracts"])
            self.assertTrue(np.array_equal(self.VecGroupInteracts, VecGroupInteracts))
            self.assertTrue(np.array_equal(self.VecGroupInteracts_load, VecGroupInteracts))

            NVclus = np.array(fl["NVclus"])[0]
            self.assertTrue(len(self.VclusExp.vecClus), NVclus)
            self.assertTrue(len(self.VclusExp.vecClus), self.NVclus)
            self.assertTrue(len(self.VclusExp.vecClus), self.NVclus_load)

            vacSpec = np.array(fl["vacSpec"])[0]
            self.assertEqual(self.vacSpec, vacSpec)

            self.assertEqual(np.max(VecGroupInteracts), NVclus - 1)

    def test_Calculate_L(self):
        sampInd = np.random.randint(0, self.state1List.shape[0] - 10)
        print("Testing 10 samples from: {}".format(sampInd))
        stateList = self.state1List[sampInd: sampInd + 10]
        jumpSelects = self.jumpSelects[sampInd: sampInd + 10]
        dispList = self.dispList[sampInd: sampInd + 10]
        rateList = self.rateList[sampInd: sampInd + 10]
        etaBar = np.random.rand(self.NVclus)

        self.assertTrue(self.vacsiteInd == 0)
        # (state1List, SpecExpand, VacSpec, rateList, dispList, jumpSelects,
        #  jList, dxList, vacsiteInd, NVclus, JitExpander, etaBar, start, end,
        #  numVecsInteracts, VecGroupInteracts, VecsInteracts)

        L, Lsamps = Calculate_L(stateList, self.SpecExpand, self.vacSpec, rateList, dispList, jumpSelects,
                    self.jList, self.dxList * self.a0, self.vacsiteInd, self.NVclus, self.JitExpander, etaBar, 0, 10,
                    self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

        self.assertAlmostEqual(np.sum(Lsamps) / 10, L, places=8)
        Lattice_translations = []
        translationSet = set()
        for siteInd in range(self.VclusExp.Nsites):
            ciSite, Rsite = self.VclusExp.sup.ciR(siteInd)
            Rtup = tuple([r for r in Rsite])
            assert Rtup not in translationSet
            Lattice_translations.append(Rsite)
            translationSet.add(Rtup)

        assert len(Lattice_translations) == self.VclusExp.Nsites // len(self.VclusExp.crys.basis[self.VclusExp.chem])

        # Now go through each of the samples and verify
        if self.SpecExpand == self.vacSpec:
            print("Testing vacancy.")
        for samp in range(10):
            state = stateList[samp]
            jmp = jumpSelects[samp]
            # small check for jump indexing
            state2_explict = state.copy()
            jSite = self.jList[jmp]
            state2_explict[jSite] = state[self.vacsiteInd]
            state2_explict[self.vacsiteInd] = state[jSite]

            disp = dispList[samp, self.SpecExpand, :]
            if self.SpecExpand == self.vacSpec:
                self.assertTrue(np.allclose(disp, self.dxList[jmp]*self.a0, atol=1e-10))

            # get the lambda of the two states by going through the vector stars explictly
            lamb1_all = np.zeros((len(self.VclusExp.vecVec), 3))
            lamb2_all = np.zeros((len(self.VclusExp.vecVec), 3))

            lamb1_req = np.zeros((len(self.VclusExp.vecVec), 3))
            lamb2_req = np.zeros((len(self.VclusExp.vecVec), 3))

            for vcInd in range(lamb1_all.shape[0]):
                clustList = self.VclusExp.vecClus[vcInd]
                vecList = self.VclusExp.vecVec[vcInd]

                for clust, vec in zip(clustList, vecList):
                    clSites = clust.siteList
                    clSpecs = clust.specList
                    # print(clSpecs)
                    self.assertEqual(len(clSpecs), len(clSites))
                    for R in Lattice_translations:
                        siteListNew = []  # to store translated sites
                        for cSite in clSites:
                            ciSite, Rsite = cSite.ci, cSite.R
                            RsiteNew = Rsite + R
                            siteIndNew, _ = self.VclusExp.sup.index(RsiteNew, ciSite)
                            siteListNew.append(siteIndNew)

                        # check if the translated cluster is on
                        off1 = 0
                        off2 = 0
                        for st, sp in zip(siteListNew, clSpecs):
                            if state[st] != sp:
                                off1 += 1
                            if state2_explict[st] != sp:
                                off2 += 1

                        if off1 == 0:
                            lamb1_all[vcInd] += vec
                        if off2 == 0:
                            lamb2_all[vcInd] += vec

                        # Now get the vectors only if the interactions contain the vacancy neighboring sites.
                        if any([st in [self.vacsiteInd] + list(self.jList) for st in siteListNew]):
                            off1 = 0
                            off2 = 0
                            for st, sp in zip(siteListNew, clSpecs):
                                if state[st] != sp:
                                    off1 += 1
                                if state2_explict[st] != sp:
                                    off2 += 1

                            if off1 == 0:
                                lamb1_req[vcInd] += vec
                            if off2 == 0:
                                lamb2_req[vcInd] += vec

            del_lamb_all = lamb2_all - lamb1_all
            del_lamb_req = lamb2_req - lamb1_req
            self.assertFalse(np.allclose(del_lamb_req, 0.)) # some clusters must have changed after a jump
            self.assertTrue(np.allclose(del_lamb_req, del_lamb_all, rtol=1e-8))
            dy = np.dot(del_lamb_req.T, etaBar)
            dispMod = disp + dy

            Ls = rateList[samp] * np.linalg.norm(dispMod) ** 2 / 6.
            self.assertAlmostEqual(Ls, Lsamps[samp], places=8)
            print("checked Lsamp : {}".format(Ls))

    def test_Expand(self):
        sampInd = 10
        stateList = self.state1List[sampInd:sampInd + 1]
        AllJumpRates = self.AllJumpRates[sampInd:sampInd + 1]
        jumpSelects = self.jumpSelects[sampInd:sampInd + 1]
        dispList = self.dispList[sampInd:sampInd + 1]
        rateList = self.rateList[sampInd:sampInd + 1]
        SpecExpand = self.SpecExpand
        NVclus = len(self.VclusExp.vecClus)
        assert NVclus == self.NVclus

        state = stateList[0]

        state2_explict = state.copy()
        jmp = jumpSelects[0]
        jSite = self.jList[jmp]
        state2_explict[jSite] = state[self.vacsiteInd]
        state2_explict[self.vacsiteInd] = state[jSite]

        dxJmp = self.dxList[jmp]
        state2Trans = np.zeros_like(state)
        for siteInd in range(state.shape[0]):
            cisite, Rsite = self.superCell.ciR(siteInd)
            dxSite = self.crys.pos2cart(Rsite, cisite)
            dxSiteNew = dxSite - dxJmp
            RsiteNew, ciSiteNew = self.crys.cart2pos(dxSiteNew)
            siteIndNew, _ = self.superCell.index(RsiteNew, ciSiteNew)
            state2Trans[siteIndNew] = state2_explict[siteInd]

        state2 = state2Trans

        # Next we'll check the change in the basis vectors with and without considering only the required sites
        off_sc = GetOffSite(state, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts,
                                   self.JitExpander.SpecOnInteractSites)
        self.assertTrue(off_sc.shape == self.JitExpander.numSitesInteracts.shape == (self.JitExpander.numSitesInteracts.shape[0],),
                        msg="{} {} {}".format(off_sc.shape, self.JitExpander.numSitesInteracts.shape,
                                              (self.JitExpander.numSitesInteracts.shape[0],)))
        # check off site count
        for interactID in range(self.JitExpander.numSitesInteracts.shape[0]):
            offCount = 0
            for IntSiteInd in range(self.JitExpander.numSitesInteracts[interactID]):
                supSite = self.JitExpander.SupSitesInteracts[interactID, IntSiteInd]
                sp = self.JitExpander.SpecOnInteractSites[interactID, IntSiteInd]
                if state[supSite] != sp:
                    offCount += 1
            self.assertEqual(offCount, off_sc[interactID])

        off_sc_all = GetOffSite(state, self.JitExpander_all.numSitesInteracts, self.JitExpander_all.SupSitesInteracts,
                                   self.JitExpander_all.SpecOnInteractSites)

        off_sc2 = GetOffSite(state2, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts,
                                   self.JitExpander.SpecOnInteractSites)

        off_sc2_all = GetOffSite(state2, self.JitExpander_all.numSitesInteracts, self.JitExpander_all.SupSitesInteracts,
                                    self.JitExpander_all.SpecOnInteractSites)

        off_sc2_explicit = GetOffSite(state2_explict, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts,
                                    self.JitExpander.SpecOnInteractSites)

        off_sc2_explicit_all = GetOffSite(state2_explict, self.JitExpander_all.numSitesInteracts, self.JitExpander_all.SupSitesInteracts,
                                             self.JitExpander_all.SpecOnInteractSites)

        on1 = np.zeros(NVclus, dtype=int)
        on2 = np.zeros(NVclus, dtype=int)
        assert off_sc2_explicit_all.shape[0] == off_sc2_all.shape[0]
        for interactID in range(off_sc2_all.shape[0]):
            if off_sc2_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    on1[vecGroup] += 1

            if off_sc2_explicit_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    on2[vecGroup] += 1

        self.assertTrue(np.array_equal(on1, on2))

        # Now get the lambda vectors for each state
        lamb1 = np.zeros((NVclus, 3))
        lamb2 = np.zeros((NVclus, 3))
        lamb2_explicit = np.zeros((NVclus, 3))

        assert self.numVecsInteracts.shape[0] == self.JitExpander.numSitesInteracts.shape[0]

        for interactID in range(self.numVecsInteracts.shape[0]):
            if off_sc[interactID] == 0:
                numVecs = self.numVecsInteracts[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts[interactID, vecInd]
                    lamb1[vecGroup] += vec

            if off_sc2[interactID] == 0:
                numVecs = self.numVecsInteracts[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts[interactID, vecInd]
                    lamb2[vecGroup] += vec

            if off_sc2_explicit[interactID] == 0:
                numVecs = self.numVecsInteracts[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts[interactID, vecInd]
                    lamb2_explicit[vecGroup] += vec

        # Now get the lambda vectors for each state with all interactions
        assert self.NVclus_all == len(self.VclusExp_all.vecClus)
        lamb1_all = np.zeros((self.NVclus_all, 3))
        lamb2_all = np.zeros((self.NVclus_all, 3))
        lamb2_explicit_all = np.zeros((self.NVclus_all, 3))

        assert self.numVecsInteracts_all.shape[0] == self.JitExpander_all.numSitesInteracts.shape[0]
        for interactID in range(self.numVecsInteracts_all.shape[0]):
            if off_sc_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts_all[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    lamb1_all[vecGroup] += vec

            if off_sc2_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts_all[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    lamb2_all[vecGroup] += vec

            if off_sc2_explicit_all[interactID] == 0:
                numVecs = self.numVecsInteracts_all[interactID]
                for vecInd in range(numVecs):
                    vec = self.VecsInteracts_all[interactID, vecInd]
                    vecGroup = self.VecGroupInteracts_all[interactID, vecInd]
                    lamb2_explicit_all[vecGroup] += vec

        # Translational symmetry will be broken if not all interactions are included.
        # So lamb2_explicit and lamb2 won't be the same.
        # However, all necessary interactions to compute CHANGE in lambda must still be accounted for.
        self.assertTrue(np.allclose(lamb2_explicit_all, lamb2_all),
                        msg="\n{} \n\n {}".format(lamb2_explicit_all[:10], lamb2_all[:10]))
        del_lamb_all_interacts = lamb2_all - lamb1_all
        del_lamb = lamb2_explicit - lamb1
        self.assertTrue(np.allclose(del_lamb, del_lamb_all_interacts),
                        msg="\n{} \n {}".format(del_lamb[:5], del_lamb_all_interacts[:5]))

        lamb1_comp = self.JitExpander.getLambda(off_sc, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                             self.VecsInteracts)

        self.assertTrue(np.allclose(lamb1_comp, lamb1, rtol=0, atol=1e-8))

        lamb2_comp = self.JitExpander.getLambda(off_sc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                     self.VecsInteracts)

        self.assertTrue(np.allclose(lamb2_comp, lamb2, rtol=0, atol=1e-8))

        del_lamb_comp = self.JitExpander.DoSwapUpdate(state, 0, self.jList[jumpSelects[0]], NVclus, off_sc,
                                        self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

        self.assertTrue(np.array_equal(off_sc, off_sc2_explicit))

        self.JitExpander.revert(off_sc, state, 0, self.jList[jumpSelects[0]])

        # check the del_lamb function
        del_lamb_comp_2 = self.JitExpander.getDelLamb(state, off_sc, 0, self.jList[jumpSelects[0]], self.NVclus,
                   self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

        self.assertTrue(np.allclose(del_lamb_comp, del_lamb), msg="\n{} \n {}".format(del_lamb_comp[:5], del_lamb[:5]))
        self.assertTrue(np.allclose(del_lamb_comp_2, del_lamb), msg="\n{} \n {}".format(del_lamb_comp[:5], del_lamb[:5]))

        # Try with single jump dataset first
        # Expand(T, state1List, vacsiteInd, Nsamples, jSiteList, dxList, AllJumpRates,
        #        jSelectList, dispSelects, ratesEscape, SpecExpand, JitExpander, NVclus,
        #        numVecsInteracts, VecsInteracts, VecGroupInteracts, aj, rcond=1e-8)

        Wbar, Bbar, etaBar, offscTime, expandTime = Expand(1073, stateList, self.vacsiteInd, 1, self.jList,
                                                                 self.dxList, AllJumpRates, jumpSelects, dispList, rateList,
                                                                 SpecExpand, self.JitExpander, NVclus,
                                                                 self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts,
                                                                 aj=False)

        Wbar_comp = np.zeros((NVclus, NVclus))
        Bbar_comp = np.zeros(NVclus)

        for l1 in range(NVclus):
            dx = dispList[0, SpecExpand]
            Bbar_comp[l1] = rateList[0] * np.dot(dx, del_lamb[l1])
            for l2 in range(NVclus):
                Wbar_comp[l1, l2] = rateList[0] * np.dot(del_lamb[l1], del_lamb[l2])

        self.assertTrue(np.allclose(Wbar, Wbar_comp))
        self.assertTrue(np.allclose(Bbar, Bbar_comp), msg="{} \n {} \n {}".format(SpecExpand, Bbar_comp[:10], Bbar[:10]))

        # Then try with all jumps
        a0 = np.linalg.norm(dispList[0, self.NSpec - 1, :]) / np.linalg.norm(self.dxList[0])
        print(a0)
        Wbar, Bbar, etaBar, offscTime, expandTime = Expand(1073, stateList, self.vacsiteInd, 1, self.jList,
                                                                 self.dxList * a0, AllJumpRates, jumpSelects, dispList, rateList,
                                                                 SpecExpand, self.JitExpander, NVclus,
                                                                 self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts,
                                                                 aj=True)

        Wbar_comp = np.zeros((NVclus, NVclus))
        Bbar_comp = np.zeros(NVclus)

        # check that off site counts have been correctly reverted in previous operations
        self.assertTrue(np.array_equal(off_sc, GetOffSite(state, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts,
                                   self.JitExpander.SpecOnInteractSites)))

        if SpecExpand == self.vacSpec:
            print("Testing vacancy")

        for jmp in range(self.dxList.shape[0]):
            del_lamb = self.JitExpander.getDelLamb(state, off_sc, 0, self.jList[jmp], self.NVclus,
                                                    self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

            for l1 in range(NVclus):
                if SpecExpand != self.vacSpec:
                    dx = -self.dxList[jmp] if state[self.jList[jmp]] == self.SpecExpand else np.zeros(3)
                else:
                    dx = self.dxList[jmp]
                Bbar_comp[l1] += AllJumpRates[0, jmp] * np.dot(dx * a0, del_lamb[l1])
                for l2 in range(NVclus):
                    Wbar_comp[l1, l2] += AllJumpRates[0, jmp] * np.dot(del_lamb[l1], del_lamb[l2])

        self.assertTrue(np.allclose(Wbar, Wbar_comp))
        self.assertTrue(np.allclose(Bbar, Bbar_comp))

    def test_vectors_explicit(self):
        sampInd = np.random.randint(0, self.state1List.shape[0])
        print("Testing sample: {}".format(sampInd))
        stateList = self.state1List[sampInd: sampInd + 1]
        AllJumpRates = self.AllJumpRates[sampInd: sampInd + 1]
        jumpSelects = self.jumpSelects[sampInd: sampInd + 1]
        dispList = self.dispList[sampInd: sampInd + 1]
        rateList = self.rateList[sampInd: sampInd + 1]

        Lattice_translations = []
        translationSet = set()
        for siteInd in range(self.VclusExp.Nsites):
            ciSite, Rsite = self.VclusExp.sup.ciR(siteInd)
            Rtup = tuple([r for r in Rsite])
            assert Rtup not in translationSet
            Lattice_translations.append(Rsite)
            translationSet.add(Rtup)

        assert len(Lattice_translations) == self.VclusExp.Nsites // len(self.VclusExp.crys.basis[self.VclusExp.chem])

        state = stateList[0]
        jmp = jumpSelects[0]
        # small check for jump indexing
        state2_explict = state.copy()
        jSite = self.jList[jmp]
        state2_explict[jSite] = state[0]
        state2_explict[0] = state[jSite]

        # get the lambda of the two states by going through the vector stars explictly
        lamb1_all = np.zeros((len(self.VclusExp.vecVec), 3))
        lamb2_all = np.zeros((len(self.VclusExp.vecVec), 3))

        lamb1_req = np.zeros((len(self.VclusExp.vecVec), 3))
        lamb2_req = np.zeros((len(self.VclusExp.vecVec), 3))
        on1 = 0
        on2 = 0
        for vcInd in range(lamb1_all.shape[0]):
            clustList = self.VclusExp.vecClus[vcInd]
            vecList = self.VclusExp.vecVec[vcInd]

            for clust, vec in zip(clustList, vecList):
                clSites = clust.siteList
                clSpecs = clust.specList
                # print(clSpecs)
                self.assertEqual(len(clSpecs), len(clSites))
                for R in Lattice_translations:
                    siteListNew = [] # to store translated sites
                    for cSite in clSites:
                        ciSite, Rsite = cSite.ci, cSite.R
                        RsiteNew = Rsite + R
                        siteIndNew, _ = self.VclusExp.sup.index(RsiteNew, ciSite)
                        siteListNew.append(siteIndNew)

                    # check if the translated cluster is on
                    off1 = 0
                    off2 = 0
                    for st, sp in zip(siteListNew, clSpecs):
                        if state[st] != sp:
                            off1 += 1
                        if state2_explict[st] != sp:
                            off2 += 1

                    if off1 == 0:
                        on1 += 1
                        lamb1_all[vcInd] += vec
                    if off2 == 0:
                        on2 += 1
                        lamb2_all[vcInd] += vec

                    if any([st in [self.vacsiteInd] + list(self.jList) for st in siteListNew]):
                        off1 = 0
                        off2 = 0
                        for st, sp in zip(siteListNew, clSpecs):
                            if state[st] != sp:
                                off1 += 1
                            if state2_explict[st] != sp:
                                off2 += 1

                        if off1 == 0:
                            lamb1_req[vcInd] += vec
                        if off2 == 0:
                            lamb2_req[vcInd] += vec

        print("on cluster counts: {} {}".format(on1, on2))
        del_lamb_all = lamb2_all - lamb1_all
        del_lamb_req = lamb2_req - lamb1_req
        self.assertFalse(np.allclose(del_lamb_req, 0.))
        self.assertTrue(np.allclose(del_lamb_req, del_lamb_all, rtol=1e-8))
        print("min, max components in del_lamb: {} {}".format(np.min(del_lamb_all), np.max(del_lamb_all)))

        off_sc = GetOffSite(state, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts,
                                   self.JitExpander.SpecOnInteractSites)

        del_lamb_comp = self.JitExpander.getDelLamb(state, off_sc, self.vacsiteInd, self.jList[jmp], self.NVclus,
                   self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

        print("min, max components in del_lamb computed: {} {}".format(np.min(del_lamb_comp), np.max(del_lamb_comp)))

        self.assertTrue(np.allclose(del_lamb_comp, del_lamb_all, rtol=1e-8), msg="{} \n\n {}".format(del_lamb_comp.shape, del_lamb_all.shape))

        # Now let's rotate the jump and check the vector change symmetries as well
        dxJmp = self.dxList[jmp]
        # make some random eta bar
        etabar = np.random.rand(self.NVclus)
        y1 = np.dot(del_lamb_comp.T, etabar)
        print(y1)
        for g in tqdm(self.VclusExp.sup.crys.G, ncols=65, position=0, leave=True):
            stateG = np.zeros_like(state)
            for site in range(state.shape[0]):
                ciSite, Rsite = self.VclusExp.sup.ciR(site)
                Rnew, ciNew = self.VclusExp.sup.crys.g_pos(g, Rsite, ciSite)
                siteIndNew, _ = self.VclusExp.sup.index(Rnew, ciNew)
                stateG[siteIndNew] = state[site]

            dxJmpG = np.dot(g.cartrot, dxJmp)
            jmpG = None
            count = 0
            for jInd in range(self.dxList.shape[0]):
                if np.allclose(dxJmpG, self.dxList[jInd]):
                    count += 1
                    jmpG = jInd
            self.assertEqual(count, 1)
            self.assertEqual(state[self.jList[jmp]], stateG[self.jList[jmpG]])
            off_sc_G = GetOffSite(stateG, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts,
                                       self.JitExpander.SpecOnInteractSites)

            del_lamb_comp_G = self.JitExpander.getDelLamb(stateG, off_sc_G, 0, self.jList[jmpG], self.NVclus,
                                                  self.numVecsInteracts, self.VecGroupInteracts, self.VecsInteracts)

            del_lamb_rot = np.dot(g.cartrot, del_lamb_comp.T).T

            self.assertTrue(np.allclose(del_lamb_rot, del_lamb_comp_G, rtol=1e-8))
            yg = np.dot(del_lamb_comp_G.T, etabar)

            self.assertTrue(np.allclose(np.dot(g.cartrot, y1), yg, rtol=1e-8))
            # print(np.dot(g.cartrot, y1), yg)

    def test_symm(self):
        initState = self.state1List[0]
        # let's make some random etabar
        NVclus = len(self.VclusExp.vecClus)
        etabar = np.random.rand(NVclus)
        for g in tqdm(self.VclusExp.sup.crys.G, ncols=65, position=0, leave=True):
            stateG = np.zeros_like(initState)

            for site in range(initState.shape[0]):
                ciSite, Rsite = self.VclusExp.sup.ciR(site)
                Rnew, ciNew = self.VclusExp.sup.crys.g_pos(g, Rsite, ciSite)
                siteIndNew, _ = self.VclusExp.sup.index(Rnew, ciNew)
                stateG[siteIndNew] = initState[site]

            # Now get their off site counts and basis vectors
            offsc1 = GetOffSite(initState, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts, self.JitExpander.SpecOnInteractSites)
            offsc2 = GetOffSite(stateG, self.JitExpander.numSitesInteracts, self.JitExpander.SupSitesInteracts, self.JitExpander.SpecOnInteractSites)

            # Now get their basis vectors
            lamb1 = self.JitExpander.getLambda(offsc1, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                                 self.VecsInteracts)
            lamb2 = self.JitExpander.getLambda(offsc2, NVclus, self.numVecsInteracts, self.VecGroupInteracts,
                                                 self.VecsInteracts)

            # Rotate lamb1 by the group operation
            lamb1G = np.dot(g.cartrot, lamb1.T).T
            self.assertTrue(np.allclose(lamb1G, lamb2, rtol=1e-8))

            y1 = np.dot(lamb1.T, etabar)
            y2 = np.dot(lamb2.T, etabar)
            self.assertTrue(np.allclose(np.dot(g.cartrot, y1), y2, rtol=1e-8))


class Test_HEA_LBAM_vac(Test_HEA_LBAM):

    def setUp(self):
        self.DataPath = ("testData_HEA_MEAM_orthogonal.h5")
        self.CrysDatPath = (cp + "CrystData_ortho_5_cube.h5")
        self.a0 = 3.595

        self.remap = True

        self.jList, self.dxList, self.superCell, self.vacsite, self.vacsiteInd, self.siteMap_nonPrimitive_to_primitive = \
            Load_crys_Data(self.CrysDatPath, ReduceToPrimitve=True)

        self.state1List, self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects = \
            Load_Data(self.DataPath, self.siteMap_nonPrimitive_to_primitive)

        self.assertTrue(self.siteMap_nonPrimitive_to_primitive is not None)

        self.crys = self.superCell.crys
        print(self.crys)

        print("Sites in Dataset: ", self.state1List.shape[1])
        print("Sites in supercell: ", len(self.superCell.mobilepos))

        self.AllSpecs = np.unique(self.state1List[0])
        self.NSpec = self.AllSpecs.shape[0]
        self.vacSpec = self.state1List[0, 0]
        self.SpecExpand = 0
        print("All Species: {}".format(self.AllSpecs))
        print("Vacancy Species: {}".format(self.vacSpec))
        print("Expanding Species: {}".format(self.SpecExpand))

        
        print("Generating New cluster expansion with vacancy at {}, {}".format(self.vacsite.ci, self.vacsite.R))

        self.ClustCut = 1.01
        self.MaxOrder = 2

        self.VclusExp = makeVClusExp(self.superCell, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=False)

        self.JitExpander, self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.NVclus = CreateJitCalculator(self.VclusExp, self.NSpec,
                                                                                                scratch=True,
                                                                                                save=True)

        # Now re-make the same calculator by loading from saved h5 file
        self.JitExpander_load, self.numVecsInteracts_load, self.VecsInteracts_load, self.VecGroupInteracts_load, \
        self.NVclus_load = CreateJitCalculator(self.VclusExp, self.NSpec, scratch=False)

        self.VclusExp_all = makeVClusExp(self.superCell, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=True)

        self.JitExpander_all, self.numVecsInteracts_all, self.VecsInteracts_all, self.VecGroupInteracts_all,\
        self.NVclus_all = CreateJitCalculator(self.VclusExp_all, self.NSpec, save=False)

class Test_HEA_LBAM_SR2(Test_HEA_LBAM):

    def setUp(self):
        self.DataPath = ("testData_SR2.h5")
        self.CrysDatPath = (cp + "CrystData.h5")
        self.a0 = 1

        self.remap = False

        self.jList, self.dxList, self.superCell, self.vacsite, self.vacsiteInd, self.siteMap_nonPrimitive_to_primitive = \
            Load_crys_Data(self.CrysDatPath, ReduceToPrimitve=self.remap)

        self.assertTrue(self.siteMap_nonPrimitive_to_primitive is None)

        self.state1List, self.dispList, self.rateList, self.AllJumpRates, self.jumpSelects = \
            Load_Data(self.DataPath, self.siteMap_nonPrimitive_to_primitive)

        self.crys = self.superCell.crys
        self.AllSpecs = np.unique(self.state1List[0])
        self.NSpec = self.AllSpecs.shape[0]
        self.vacSpec = self.state1List[0, 0]
        print(self.vacSpec)
        self.SpecExpand = 1
        print("All Species: {}".format(self.AllSpecs))
        print("Vacancy Species: {}".format(self.vacSpec))
        print("Expanding Species: {}".format(self.SpecExpand))

        self.ClustCut = 1.01
        self.MaxOrder = 2

        self.VclusExp = makeVClusExp(self.superCell, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=False)

        self.JitExpander, self.numVecsInteracts, self.VecsInteracts, self.VecGroupInteracts, self.NVclus = CreateJitCalculator(self.VclusExp, self.NSpec,
                                                                                                scratch=True,
                                                                                                save=True)

        # Now re-make the same calculator by loading from saved h5 file
        self.JitExpander_load, self.numVecsInteracts_load, self.VecsInteracts_load, self.VecGroupInteracts_load,\
        self.NVclus_load = CreateJitCalculator(self.VclusExp, self.NSpec, scratch=False)

        self.VclusExp_all = makeVClusExp(self.superCell, self.jList, self.ClustCut, self.MaxOrder, self.NSpec, self.vacsite, self.vacSpec,
                                AllInteracts=True)

        self.JitExpander_all, self.numVecsInteracts_all, self.VecsInteracts_all, self.VecGroupInteracts_all,\
        self.NVclus_all = CreateJitCalculator(self.VclusExp_all, self.NSpec, scratch=True, save=False)

