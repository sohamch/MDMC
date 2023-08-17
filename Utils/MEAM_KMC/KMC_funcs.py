import numpy as np

def write_input_files(Ntr, potPath=None, ts = 0.001, etol=0.0, ftol=0.01, k=1.0, perp=1.0, threshold=1.0,
                      NImages=11):
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
                fl.write("pair_coeff \t * * "+ potPath + "/library.meam Co Ni Cr Fe Mn " +
                         potPath + "/params.meam Co Ni Cr Fe Mn\n")

            fl.write("fix \t 1 all neb {} parallel ideal perp {}\n".format(k, perp))
            fl.write("min_style \t fire\n")
            fl.write("timestep \t {}\n".format(ts))
            fl.write("min_modify \t norm max abcfire yes tmax 5 dmax 0.01\n\n")

            # define the variables for the images
            s = "variable \t i universe"
            for im in range(NImages):
                s += " {}".format(im+1)
            fl.write(s + "\n")

            # set up dry NEB run to get images
            # 1. set all forces to zero
            fl.write("fix \t force_fix all setforce 0 0 0\n")
            # 2. Do one NEB iteration
            fl.write("neb \t {0} {1} 1 0 1 final final_{2}.data\n".format(etol, ftol, traj))
            # 3. Unfix the setforce
            fl.write("unfix \t force_fix\n")

            # 4. Invoke compute displace/atom to start atomic coordinates
            fl.write("variable \t Drel equal {}\n".format(threshold))
            fl.write("compute \t dsp all displace/atom\n\n")

            # Run full NEB
            fl.write("neb \t {0} {1} 10000 0 10 final final_{2}.data\n".format(etol, ftol, traj))

            # Dump atomic displacements.
            fl.write("dump \t disp_dmp all custom 1 Image_disps/disps_{}_$i.dump id type c_dsp[4]\n".format(traj))
            fl.write("dump_modify \t disp_dmp append no thresh c_dsp[4] > ${Drel}\n")
            fl.write("run 0")


def write_init_states(SiteIndToSpec, SiteIndToPos, vacSiteInd, TopLines):
    Ntr = vacSiteInd.shape[0]
    for traj in range(Ntr):
        with open("initial_{}.data".format(traj), "w") as fl:
            fl.writelines(TopLines)
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
            fl.write("{}\n".format(1))
            pos = SiteIndToPos[vacSiteInd[traj]]
            ngbInd = siteIndToNgb[vacSiteInd[traj], jInd]
            if ngbInd > vacSiteInd[traj]:
                LammpsAtomInd = ngbInd
            else:
                LammpsAtomInd = ngbInd + 1
            fl.write("{} {} {} {}\n".format(LammpsAtomInd, pos[0], pos[1], pos[2]))

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
