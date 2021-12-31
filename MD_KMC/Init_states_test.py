import numpy as np
import sys
args = list(sys.argv)
N_units = int(args[1])
NSpec = int(args[2])
Ntraj = int(args[3])

Nsites = N_units * N_units * N_units

SiteIndToSpec = np.zeros((Ntraj, Nsites), dtype=int)

for traj in range(Ntraj):
    SiteIndToSpec[traj, 0] = -1
    SiteIndToSpec[traj, 1:] = np.random.randint(1, NSpec+1, Nsites-1)[:]
    SiteIndToSpec[traj, :] = np.random.permutation(SiteIndToSpec[traj, :])
vacSiteInd = np.zeros(Ntraj, dtype=int)
np.save("SiteIndToSpec.npy", SiteIndToSpec)
np.save("vacSiteInd.npy", vacSiteInd)