import numpy as np
import pandas as pd

#from keras.utils.np_utils import to_categorical

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical,num_classes

padshape = 16

target_lab = "label"

pdg_lab = "inTpPdgId"

headLab = ["run","evt","lumi","PU","bSX","bSY","bSZ","bSdZ"]

hitCoord = ["X","Y","Z","Phi","R"] #5

hitDet = ["DetSeq","IsBarrel","Layer","Ladder","Side","Disk","Panel","Module","IsFlipped","Ax1","Ax2"] #12

hitClust = ["ClustX","ClustY","ClustSize","ClustSizeX","ClustSizeY","PixelZero",
            "AvgCharge","OverFlowX","OverFlowY","Skew","IsBig","IsBad","IsEdge"] #13

hitPixel = ["Pix" + str(el) for el in range(1, padshape*padshape + 1)]

hitCharge = ["SumADC"]

hitLabs = hitCoord + hitDet + hitClust + hitPixel

inHitLabs = [ "in" + str(i) for i in hitLabs]
outHitLabs = [ "out" + str(i) for i in hitLabs]

inPixels = [ "in" + str(i) for i in hitPixel]
outPixels = [ "out" + str(i) for i in hitPixel]


particle = ["PId","TId","Px","Py","Pz","Pt","MT","ET","MSqr","PdgId",
                "Charge","NTrackerHits","NTrackerLayers","Phi","Eta","Rapidity",
                "VX","VY","VZ","DXY","DZ"]

hitFeatures = hitCoord + hitDet + hitClust  # 5 + 12 + 13 + 1 = 31

inParticle = [ "inTp" + str(i) for i in particle]
outParticle = [ "outTp" + str(i) for i in particle]

inHitFeature  = [ "in" + str(i) for i in hitFeatures]
outHitFeature = [ "out" + str(i) for i in hitFeatures]

particleLabs = ["label"] + inParticle +  ["BunchCrossing"] + outParticle

featureLabs = inHitFeature + outHitFeature

dataLab = headLab + inHitLabs + outHitLabs  + particleLabs

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

layer_ids = [0, 1, 2, 3, 14, 15, 16, 29, 30, 31]

particle_ids = [-1.,11.,13.,15.,22.,111.,211.,311.,321.,2212.,2112.,3122.,223.]

main_pdgs = [11.,13.,211.,321.,2212.]

allLayerPixels = []

for i in range(10):
    thisPixels = [ h + "_in_" + str(i) for h in hitPixel]
    allLayerPixels = allLayerPixels + thisPixels
for i in range(10):
    thisPixels = [ h + "_out_" + str(i) for h in hitPixel]
    allLayerPixels = allLayerPixels + thisPixels

def balance_data_by_pdg(dataSet, pdgIds):
    """ Balancing datasets by particles. """

    data_pos  = dataSet[dataSet[target_lab] == 1.0]
    data_neg  = dataSet[dataSet[target_lab] == -1.0]
    data_pdgs = []
    minimum = 1E8
    totpdg  = 0

    for p in pdgIds:
        data_excl  = data_pos[data_pos[pdg_lab].abs() != p]
        data_pdg = data_pos[data_pos[pdg_lab].abs() == p]
        data_pdgs.append(data_pdg)
        minimum=min(data_pdg.shape[0]*2,minimum)
        totpdg = totpdg + data_pdg.shape[0]
        totpdg = totpdg + data_pdg.shape[0]
        assert minimum > 0, "%.1f pdg id has zero entries. Returning." % p

    data_excl = data_excl.sample(frac=1.0)
    data_excl = data_excl.sample(totpdg/2)

    data_neg = data_neg.sample(frac=1.0)
    data_neg = data_neg.sample(totpdg)

    for d in data_pdgs:
        if d.shape[0] > minimum:
            d = d.sample(minimum)

    data_tot = pd.concat(data_pdgs + [data_excl,data_neg])
    data_tot = data_tot.sample(frac=1.0)

    return data_tot # allow method chaining

class Dataset:
    """ Load the dataset from txt files. """

    def __init__(self, fnames,balance=False,pdgIds=main_pdgs):
        self.data = pd.DataFrame(data=[], columns=dataLab)
        for i,f in enumerate(fnames):
            print("Loading file " + str(i+1) + "/" + str(len(fnames)) + " : " + f)
            df = 0
            if not f.lower().endswith("h5"):
                continue

            df = pd.read_hdf(f, mode='r')
            if balance:
                df = balance_data_by_pdg(df,pdgIds)

            df.columns = dataLab  # change wrong columns names
            df.sample(frac=1.0)
            self.data = self.data.append(df)

    def from_dataframe(self,data):
        """ Constructor method to initialize the classe from a DataFrame """
        self.data = data

    def recolumn(self):
        self.data.columns = dataLab


    def save(self, fname):
        # np.save(fname, self.data.as_matrix())
        self.data.to_hdf(fname, 'data', mode='w',append=False,complib="bzip2",complevel=9)

    # TODO: pick doublets from same event.
    def balance_data(self, max_ratio=0.5, verbose=True):
        """ Balance the data. """
        data_neg = self.data[self.data[target_lab] == -1.0]
        data_pos = self.data[self.data[target_lab] != -1.0]

        n_pos = data_pos.shape[0]
        n_neg = data_neg.shape[0]

        if n_pos==0:
            print("Number of negatives: " + str(n_neg))
            print("Number of positive: " + str(n_pos))
            print("Returning")
            return self
        if verbose:
            print("Number of negatives: " + str(n_neg))
            print("Number of positive: " + str(n_pos))
            print("Ratio: " + str(n_neg / n_pos))

        if n_pos > n_neg:
            return self

        data_neg = data_neg.sample(n_pos)
        balanced_data = pd.concat([data_neg, data_pos])
        balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
        self.data = balanced_data
        return self  # allow method chaining
