import numpy as np
import copy


def shock_gen(nBanks, externalAssets, lossRate):

    postExternalAssets = np.zeros([nBanks, nBanks])
    for i in range(0, nBanks):
        b = copy.copy(externalAssets)
        c = copy.copy(externalAssets)
        b[i] = c[i]*(1-lossRate)
        postExternalAssets[i]=b
    return postExternalAssets
