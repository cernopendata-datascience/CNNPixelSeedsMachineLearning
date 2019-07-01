# CNNPixelSeeds
Analyzer for CMS Open Data Pixel Seeds ML Applications http://opendata.cern.ch/

Data producer for developing machine learning algorithms to select and filter pixel doublet seeds for tracking applications at CMS experiments.

### Setup

The first step is the creation of a ``CMSSW_10_2_5`` release workarea

``` bash
cmsrel CMSSW_10_2_5
cd CMSSW_10_2_5/src/
git clone git@github.com:cms-legacydata-analyses/CNNPixelSeedsProducerTool.git .
scram b -j 2
cmsenv
```

### Dumping doublets in  ` txt ` files

Once the compilation is completed you are ready to produce the pixel doublets seeds datasets:
 
``` bash
cmsrel CMSSW_10_3_5
cd CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/
cmsRun step3_ML_trackingOnly.py
```

This configuration will run the full CMS track reconstruction on simulated (![equation](http://latex.codecogs.com/gif.latex?t\bar{t})) events and will produce in `CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/doublets/` directory (automatically created) a set of text (`txt`) files containing the doublets produced in each of the pixel seed based iterative tracking steps (red boxed in the picture below). For further details about track reconstruction at CMS and iterative tracking see [1],[2],[3]

![iterativeTracking](https://raw.githubusercontent.com/AdrianoDee/CNNFiltering/open/iterativeTracking.png)

Analogously 

``` bash
cmsrel CMSSW_10_3_5
cd CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/
cmsRun step3_ML_pixelOnly.py
```

will produce the doublets seeds used aas starting blocks for pixel-only tracks reconstruction. The text files generated are named with the following rules:

`_l_r_e_step_dnn_doublets.txt`

with

* l = lumisection number
* r = run number
* e = event number
* step = iterative tracking step name (e.g `pixelTracksHitDoublets` for pixel tracks step)

Both the configuration files (`step3_ML_pixelOnly.py` and `step3_ML_trackingOnly.py`) may receive in input few parameters:

| Name       | Type                 | Default | Description                                                                                    |
|------------|----------------------|---------|------------------------------------------------------------------------------------------------|
| pileUp     | int                  | 50      | Average number of simultaneous collisions per event  (for this use case should be kept to 50). |
| skipEvent  | int                  | 0       | Number of events to be skipped.                                                                |
| numEvents  | int                  | 100     | Total number of events to be processed (after skipping).                                       |
| numFile    | int                  | 0       | The index, in the list provided, of the file to be processed.                                  |
| openDataVM | bool (True or False) | True    | Flag to signal if you are working on an Open Data WM or somewhere else.                        
|

Any of these inputs should be parsed as follows:

`cmsRun step3_ML_trackingOnly.py inputName=VALUE`



### Conversion to ` HDF ` files

In order to convert the txt datasets to hdf table formats simply run (in `CMSSW_10_3_5/src/CNNFiltering/CNNAnalyze/test/`)

` python toHdf.py`

this will automatically read the content of `doublets` directory and produce two hdf files:

* in `doublets/original/` the plain hdf converted file;
* in `doublets/bal_data/` a new balanced hdf table where the yields of fake and true seeds have been forced to be equal, by sampling the more populated of the two classes;

### The dataset

The dataset created above consists of a collection of pixel doublet seeds that would be used by CMS track reconstruction workflow. Each doublet is characterised by a list of features:

<table>
  <tr>
   <td><strong>Event Info</strong>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>run
   </td>
   <td>Run number
   </td>
  </tr>
  <tr>
   <td>evt
   </td>
   <td>Event number
   </td>
  </tr>
  <tr>
   <td>lumi
   </td>
   <td>Lumisection number
   </td>
  </tr>
  <tr>
   <td>PU
   </td>
   <td>Number of primary vertices in the event
   </td>
  </tr>
  <tr>
   <td>bSX, bSY, bSZ, bSdZ
   </td>
   <td>Beam spot coordinates (x,y,z) and \sigma_{z}
   </td>
  </tr>
  <tr>
   <td><strong>Features</strong>
   </td>
   <td><em>(“in” or “out” prefix to indicate the inner or the outer hit of the doublet, e.g. inDetSeq, outX . . .)</em>
   </td>
  </tr>
  <tr>
   <td>DetSeq
   </td>
   <td>Sequential number for the inner hit and outer hit layer. For the silicon pixel detectors these numbers may be {0,1,2,3} for the four pixel barrel layers {14,15,16} for the three negative encap and {29,30,31} for the three positive endcap layers. 
   </td>
  </tr>
  <tr>
   <td>X, Y, Z, R 
   </td>
   <td>Doublet inner [outer] hit spatial coordinates. 
   </td>
  </tr>
  <tr>
   <td>Phi 
   </td>
   <td>Doublet inner [outer] hit azimuthal angle \phi.
   </td>
  </tr>
  <tr>
   <td>R 
   </td>
   <td>Doublet inner [outer] hit radial (r=\sqrt{x^2 + y^2}) direction.
   </td>
  </tr>
  <tr>
   <td>IsBarrel
   </td>
   <td>Flag for inner [outer] hit being on a barrel layer
   </td>
  </tr>
  <tr>
   <td>Layer, Ladder, Side, Disk, Panel, Module
   </td>
   <td>Inner [outer] hit detector specifics. For the barrel detector hit two numbers are meaningful: the layer number indicates on which cylindrical layer the hit lies; the ladder number  
   </td>
  </tr>
  <tr>
   <td>IsFlipped 
   </td>
   <td>Flag indicating if the module is flipped with respect to the standard outward orientation.
   </td>
  </tr>
  <tr>
   <td>Ax1
   </td>
   <td>Length of the vector connecting the the origin to the local module coordinate reference system origin (0,0,0) for the inner [outer] hit.
   </td>
  </tr>
  <tr>
   <td>Ax2
   </td>
   <td>Length of the vector connecting the the origin to the point (0,0,1) in the local module coordinate reference  system for the inner [outer] hit.
   </td>
  </tr>
  <tr>
   <td>ClustX, ClustY
   </td>
   <td>Pixel cluster local, i.e. in the local module layer system of reference, coordinates for the inner [outer] hit. 
   </td>
  </tr>
  <tr>
   <td>OverFlowX, OverFlowY, 
   </td>
   <td>Flags indicating if the the pixel cluster for the inner [outer] hit spans over the pad size (16) along the X or Y local detector module axes.
   </td>
  </tr>
  <tr>
   <td> ClustSize, ClustSizeX, ClustSizeY
   </td>
   <td>Inner [outer] pixel cluster absolute size, i.e. number of pixel composing it, and sizes along X and Y local detector module axes.
   </td>
  </tr>
  <tr>
   <td>SumADC
   </td>
   <td>Sum of the A.D.C. levels of all the pixels composing the cluster.
   </td>
  </tr>
  <tr>
   <td>IsBig 
   </td>
   <td>Flag indicating that the inner [outer] hits spans two (or more) ROCs modules.
   </td>
  </tr>
  <tr>
   <td>IsBad 
   </td>
   <td>Flag indicating that at least one pixel composing the inner [outer] hit is marked as malfunctioning.
   </td>
  </tr>
  <tr>
   <td>IsEdge 
   </td>
   <td>Flag indicating that the inner [outer] hit is on the edge of a ROC module.
   </td>
  </tr>
  <tr>
   <td>PixelZero
   </td>
   <td>Highest equivalent released charge (in A.D.C. levels) for a single pixel belonging to the inner [outer] hit pixel cluster.
   </td>
  </tr>
  <tr>
   <td>AvgCharge 
   </td>
   <td>Average charge released on each pixel forming the inner [outer] pixel cluster.
   </td>
  </tr>
  <tr>
   <td>Skew 
   </td>
   <td>Ratio between the inner [outer] pixel cluster Y size and X size.
   </td>
  </tr>
  <tr>
   <td><strong>Pixel Pads</strong>
   </td>
   <td><em>(“in” or “out” prefix to indicate the inner or the outer hit of the doublet, e.g. inDetSeq, outX . . .)</em>
   </td>
  </tr>
  <tr>
   <td>PixX
<p>
<em>with X = 0,...,255</em>
   </td>
   <td>Inner [outer] hit pixels A.D.C. levels with X ranging from 0 to 255 for a 16x16 pad). The X index spans from top left pad corner to bottom right: e.g. the last bottom row will span from inPix240 to inPix255.
   </td>
  </tr>
  <tr>
   <td><strong>Labels</strong>
   </td>
   <td><em>(if the hit is not matched to any tracking particle all these labels are set to -1.0. “in” or “out” prefix to indicate the inner or the outer hit of the doublet, e.g. inDetSeq, outX . . .)</em>
   </td>
  </tr>
  <tr>
   <td>PId 
   </td>
   <td>Flag set to 1.0 (-1.0) if the inner [outer] hit is (not) matched 
   </td>
  </tr>
  <tr>
   <td>TId 
   </td>
   <td>Inner [outer] hit matched tracking particle key number in the event collection of tracking particles.
   </td>
  </tr>
  <tr>
   <td>Px,Py,Pz,Pt
   </td>
   <td>Inner [outer] hit matched tracking particle momentum components (p_x, p_y, p_z) and transverse momentum (p_T).
   </td>
  </tr>
  <tr>
   <td>MT
   </td>
   <td>Inner [outer] hit matched tracking particle transverse mass.
   </td>
  </tr>
  <tr>
   <td>ET
   </td>
   <td>Inner [outer] hit matched tracking particle transverse energy.
   </td>
  </tr>
  <tr>
   <td>MSqr
   </td>
   <td>Inner [outer] hit matched tracking particle mass squared.
   </td>
  </tr>
  <tr>
   <td>PdgId
   </td>
   <td>Inner [outer] hit matched tracking particle PDG id, i.e. the index indicating which kind of particle it is. 
   </td>
  </tr>
  <tr>
   <td>Charge
   </td>
   <td>Inner [outer] hit matched tracking particle charge.
   </td>
  </tr>
  <tr>
   <td>NTrackerHits
   </td>
   <td>Inner [outer] hit matched tracking particle number of tracker hits.
   </td>
  </tr>
  <tr>
   <td>NTrackerLayers
   </td>
   <td>The number of tracker layers crossed by the inner [outer] hit matched tracking particle. 
   </td>
  </tr>
  <tr>
   <td>Phi
<p>
Eta
<p>
Rapidity
   </td>
   <td>Inner [outer] hit matched tracking particle phi, eta and y.
   </td>
  </tr>
  <tr>
   <td>VX, VY, VZ
   </td>
   <td>Inner [outer] hit matched tracking particle vertex global coordinates.
   </td>
  </tr>
  <tr>
   <td>DXY
   </td>
   <td>Inner [outer] hit matched tracking particle vertex transverse impact parameter.
   </td>
  </tr>
  <tr>
   <td>DZ
   </td>
   <td>Inner [outer] hit matched tracking particle vertex longitudinal impact parameter.
   </td>
  </tr>
  <tr>
   <td>BunchCrossing
   </td>
   <td>Event bunch crossing number.
   </td>
  </tr>
</table>



The notebook in `CNNPixelSeedsProducerTool/notebooks/cnn_filtering.ipynb` is a good starting point to explore and understand the datset features.


[1] https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideIterativeTracking

[2] https://cds.cern.ch/record/2308020

[3] https://cds.cern.ch/record/2293435
