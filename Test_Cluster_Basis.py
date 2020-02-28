import numpy as np
from onsager import crystal, cluster, supercell
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        #  Let's test a BCC lattice with four atoms and a vacancy in it.
        self.crys = crystal.BCC(0.3, chemistry="A")
        self.mobList = list(range(5))
        # First, let's build a supercell - keep it diagonal for now.
        supmat = np.array([8, 8, 8], dtype=int)
        self.supBCC = supercell.ClusterSupercell(self.crys, supmat)

        # Once the supercell is made, we define occupancies.
        self.mobOccs = np.zeros((5, len(self.supBCC.mobilepos)), dtype=int)
        # First, scatter species about
        speciesAtSites = {}
        for siteindex, site in enumerate(self.supBCC.mobilepos):
            # speciesAtSites[siteindex]
            (c, i), R = self.supBCC.ciR(siteindex)
            if (c, i) == (0, 0) and np.allclose(R, 0):
                speciesAtSites[siteindex] = 4
            else:
                speciesAtSites[siteindex] = np.random.randint(0, 4)

        # Now, we make the mobile species array
        for siteInd, species in speciesAtSites.items():
            self.mobOccs[species][siteInd] = 1

        # generate the cluster expansion - nearest neighbor 4 body clusters
        self.clusexp = cluster.makeclusters(self.crys, 0.27, 4)

        # generate the jumpnetwork - nearest neighbor jumps
        self.jnetwork = self.crys.jumpnetwork(0, 0.27)

        # Then, generate the TSClusters - not sure if I will need these though
        self.TSclusexp = cluster.makeTSclusters(self.crys, 0, self.jnetwork, self.clusexp)


if __name__ == '__main__':
    unittest.main()
