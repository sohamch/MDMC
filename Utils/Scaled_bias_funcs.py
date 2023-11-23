import numpy as np
from tqdm import tqdm

def ScaledBiasBasis(States, dxList, NNvacList, JumpProbs, vacSpec):
    # Initialize basis array
    NSpec = np.unique(States[0]).shape[0]
    # We'll store the basis vectors for each species separately
    ndim = dxList.shape[1]
    Basis_specs_states = np.zeros((NSpec, States.shape[0], ndim))

    for stateInd in tqdm(range(States.shape[0]), ncols=65, position=0, leave=True):
        state = States[stateInd]
        # First calculate for initial state
        for jmp in range(dxList.shape[0]):
            gamma_jump = JumpProbs[stateInd, jmp]
            # First, the vacancy
            Basis_specs_states[0][stateInd] += gamma_jump * (dxList[jmp])
            sp = state[NNvacList[jmp]]
            assert sp != vacSpec
            Basis_specs_states[sp][stateInd] += gamma_jump * (-dxList[jmp])

        assert np.allclose(-Basis_specs_states[0][stateInd],
                           sum([Basis_specs_states[sp][stateInd] for sp in range(1, 6)]))

        return Basis_specs_states

def ScaledResBias(States, yStates, yStateExits, dxList, NNvacList, JumpProbs, vacSpec, specExpand):
    # Now we do the scaled bias basis correction procedure to improve the transport coefficients.

    # Initialize basis array
    NSpec = np.unique(States[0]).shape[0]
    print(NSpec)
    # We'll store the basis vectors for each species separately
    ndim = dxList.shape[1]
    Basis_specs_states = np.zeros((NSpec, States.shape[0], ndim))

    z = JumpProbs.shape[1]
    assert z == dxList.shape[0] == NNvacList.shape[0]

    for stateInd in tqdm(range(States.shape[0]), ncols=65, position=0, leave=True):
        state1 = States[stateInd]

        # First calculate for initial state
        for jmp in range(z):
            dxJmp = dxList[jmp]
            gamma_jump = JumpProbs[stateInd, jmp]

            sp = state1[NNvacList[jmp]]
            assert sp != vacSpec

            # First, we handle the vacancy case
            if specExpand == vacSpec:
                # For the vacancy, accumulate scaled residual bias
                Basis_specs_states[0, stateInd] += \
                    gamma_jump * (dxJmp + yStateExits[stateInd * z + jmp] - yStates[stateInd])

                # For the other specs, accumulate just scaled bias
                Basis_specs_states[sp, stateInd] += gamma_jump * (-dxJmp)

            # Then the atomic species
            else:
                if sp == specExpand:
                    Basis_specs_states[sp, stateInd] += \
                        gamma_jump * (-dxJmp + yStateExits[stateInd * z + jmp] - yStates[stateInd])

                else:
                    # Accumulate residual bias with zero actual displacement
                    Basis_specs_states[specExpand, stateInd] += \
                        gamma_jump * (yStateExits[stateInd * z + jmp] - yStates[stateInd])

                    # For the other specs, accumulate just scaled bias
                    Basis_specs_states[sp, stateInd] += gamma_jump * (-dxJmp)

    return Basis_specs_states

def Expand_scaled_bias(Basis_specs_init_states, Basis_specs_fin_states, DispsSpecExpand, escapeRates, N_train):

    assert Basis_specs_init_states.shape == Basis_specs_fin_states.shape
    n = Basis_specs_init_states.shape[0]
    W_bar = np.zeros((n, n))
    b_bar = np.zeros(n)
    # Carry out expansion over the training set
    for stateInd in tqdm(range(N_train), ncols=65, position=0, leave=True):
        rateJump = escapeRates[stateInd]
        for sp1 in range(n):
            basis_sp1_init = Basis_specs_init_states[sp1][stateInd, :]
            basis_sp1_fin = Basis_specs_fin_states[sp1][stateInd, :]
            del_basis_sp1 = basis_sp1_fin - basis_sp1_init

            b_bar[sp1] += rateJump * np.dot(DispsSpecExpand[stateInd], del_basis_sp1)
            W_bar[sp1, sp1] += rateJump * np.dot(del_basis_sp1, del_basis_sp1)

            for sp2 in range(sp1):
                basis_sp2_init = Basis_specs_init_states[sp2][stateInd, :]
                basis_sp2_fin = Basis_specs_fin_states[sp2][stateInd, :]
                del_basis_sp2 = basis_sp2_fin - basis_sp2_init

                W_bar[sp1, sp2] += rateJump * np.dot(del_basis_sp1, del_basis_sp2)
                W_bar[sp2, sp1] = W_bar[sp1, sp2]

    return W_bar, b_bar