import numpy as np

def write_input_files(Ntr, potPath=None):
    for traj in range(Ntr):
        with open("in.neb_{0}".format(traj), "w") as fl:
            fl.write("units \t metal\n")
            fl.write("atom_style \t atomic\n")
            fl.write("atom_modify \t map array\n")
            fl.write("boundary \t p p p\n")
            fl.write("atom_modify \t sort 0 0.0\n")
            fl.write("read_data \t initial_{0}.data\n".format(traj))
            fl.write("pair_style \t meam\n")
            if potPath is None:
                fl.write("pair_coeff \t * * pot/library.meam Co Ni Cr Fe Mn pot/params.meam Co Ni Cr Fe Mn\n")
            else:
                fl.write("pair_coeff \t * * "+ potPath + "pot/library.meam Co Ni Cr Fe Mn " +
                         potPath + "pot/params.meam Co Ni Cr Fe Mn\n")
            fl.write("fix \t 1 all neb 1.0\n")
            fl.write("timestep \t 0.01\n")
            fl.write("min_style \t quickmin\n")
            fl.write("neb \t 1e-5 1e-5 500 500 10 final final_{0}.data".format(traj))


def write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, TopLines):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("initial_{}.data".format(traj), "w") as fl:
            fl.writelines(TopLines[:12])
            counter = 1
            for idx in range(SiteIndToSpec.shape[1]):
                spec = SiteIndToSpec[traj, idx]
                if spec == 0:  # if the site is vacant
                    assert idx == vacSiteInd[traj], "{} {}".format(idx, SiteIndToSpec[traj, idx])
                    continue
                pos = SiteIndToPos[idx]
                fl.write("{} {} {} {} {}\n".format(counter, spec, pos[0], pos[1], pos[2]))
                counter += 1


def write_final_states(SiteIndToPos, vacSiteInd, siteIndToNgb, jInd, writeAll=False):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("final_{}.data".format(traj), "w") as fl:
            fl.write("{}\n".format(SiteIndToPos.shape[0] - 1))
            counter = 1
            for siteInd in range(len(SiteIndToPos)):
                if siteInd == vacSiteInd[traj]:
                    continue
                # the jumping atom will have vac site as the new position
                if siteInd == siteIndToNgb[vacSiteInd[traj], jInd]:
                    pos = SiteIndToPos[vacSiteInd[traj]]
                else:
                    pos = SiteIndToPos[siteInd]
                fl.write("{} {} {} {}\n".format(counter, pos[0], pos[1], pos[2]))
                counter += 1

        if writeAll:
            with open("final_{}.data".format(traj), "r") as fl:
                lines = fl.readlines()
            with open("final_{}_{}.data".format(traj, jInd), "w") as fl:
                fl.writelines(lines)


# @jit(nopython=True)
def getJumpSelects(rates):
    Ntr = rates.shape[0]
    timeStep = 1. / np.sum(rates, axis=1)
    ratesProb = rates * timeStep.reshape(Ntr, 1)
    ratesProbSum = np.cumsum(ratesProb, axis=1)
    rn = np.random.rand(Ntr)
    jumpID = np.zeros(Ntr, dtype=int)
    for tr in range(Ntr):
        jSelect = np.searchsorted(ratesProbSum[tr, :], rn[tr])
        jumpID[tr] = jSelect
    # jumpID, rateProbs, ratesCum, rndNums, time_step
    return jumpID, ratesProb, ratesProbSum, rn, timeStep


# @jit(nopython=True)
def updateStates(SiteIndToNgb, Nspec, SiteIndToSpec, vacSiteInd, jumpID, dxList):
    Ntraj = jumpID.shape[0]
    jumpAtomSelectArray = np.zeros(Ntraj, dtype=int)
    X = np.zeros((Ntraj, Nspec, 3), dtype=float)
    for tr in range(Ntraj):
        assert SiteIndToSpec[tr, vacSiteInd[tr]] == 0
        jumpSiteSelect = SiteIndToNgb[vacSiteInd[tr], jumpID[tr]]
        jumpAtomSelect = SiteIndToSpec[tr, jumpSiteSelect]
        jumpAtomSelectArray[tr] = jumpAtomSelect
        SiteIndToSpec[tr, vacSiteInd[tr]] = jumpAtomSelect
        SiteIndToSpec[tr, jumpSiteSelect] = 0  # The next vacancy site
        vacSiteInd[tr] = jumpSiteSelect
        X[tr, 0, :] = dxList[jumpID[tr]]
        X[tr, jumpAtomSelect, :] = -dxList[jumpID[tr]]

    return jumpAtomSelectArray, X
