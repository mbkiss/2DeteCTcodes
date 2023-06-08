# This file contains a function for reconstructing sinogram data for the 2DeteCT dataset.

import astra
import imageio
import warnings
from typing import Any

import glob # Used for browsing through the directories.
import os # Used for creating directories.
import shutil # Later used for copying files.
import time # Used for keeping processing time.
import NesterovGradient

import numpy as np
from scipy.interpolate import interp1d

# Data directories.
# We enter here some intrinsic details of the dataset needed for our reconstruction scripts.
# Set the variable "base_data_dir_str" to the path where the dataset is stored on your own workstation.
base_data_dir_str = '/export/scratch2/mbk/test1/'

# Set the variable "save_dir_str" to the path where you would like to store the
# reconstructions you compute.
save_dir_str = '/export/scratch2/mbk/2DeteCT_recons/' 

# User defined settings.

# Select the ID(s) of the slice(s) you want to reconstruct.
import random
slice_id = range(1,6370+1) #random.sample(range(1, 6370+1), 50)
# Adjust this to your liking.

# Select which modes you want to reconstruct.
modes = (1,2,3) #These are all modes available.

# Define whether you have a GPU for computations available and if you like specify which one to use.
use_GPU = True
#gpuIndex = 0 # Set the index of the GPU card used.
#astra.set_gpu_index(gpuIndex)

# Pre-processing parameters.
binning = 1 # Manual selection of detector pixel binning after acqusisition.
excludeLastPro = True # Exclude last projection angle which is often the same as the first one.
angSubSamp = 1 # Define a sub-sampling factor in angular direction.
# (all reference reconstructions are computed with full angular resolution).
maxAng = 360 # Maximal angle in degrees - for reconstructions with limited angle (standard: 360).

# Correction profiles.
# The detector is slightly shifted with respect to the ASTRA geometry specified.
# Furthermore, the detector has been changed shortly before 20220531 (between slices 2830 and 2831).
# The full correction profiles can be found below.
corr_profiles = dict()
corr_profiles['20220407_RvL'] = {'det_tan': 24.19, 'src_ort': -5.67, 'axs_tan': -0.5244, 'det_roll': -0.015}
corr_profiles['20220531_RvL'] = {'det_tan': 24.4203, 'src_ort': -6.2281, 'axs_tan': -0.5010, 'det_roll': -0.262}
# This array contains the simplified horizontal correction shift for both geometries.
corr = np.array([1.00, 0.0]) # Found the optimal shifts to be
# [2.75, 1.00] for (2048,2048). subsampling yields 
# [1.00, 0.00] for (1024,1024). 

# File names in dataset structure.
sino_name = 'sinogram.tif'
dark_name = 'dark.tif'
flat_name = ('flat1.tif', 'flat2.tif')
slcs_name ="slice{:05}"

# Reference information.
sino_dims = (3601,1912) # Dimensions of the full sinograms.
detPix = 0.0748 # Physical size of one detector pixel in mm.
# Early OOD scans: 5521 - 5870 
# Late OOD scans: 5871 - 6370

# Reconstruction parameters.
recSz = (2048,2048) # Used reconsttuction area to create as little model-inherent artifacts within the FOV.
outSz = (1024,1024) # Output size before downscaling corresponding to the FOV.
maxIter = 100 # Specify the maximal iteration number.

# Keep track of the processing time per reconstruction job.
t = time.time();
print('Starting reconstruction job...', flush=True)


for i_slc in slice_id:

    for i_mode in modes:

        # Load and pre-process data.

        # Get the current path for respective slice and mode within the dataset structure.
        current_path = base_data_dir_str + slcs_name.format(i_slc) + '/mode{}/'.format(i_mode)

        # load flat-field and dark-fields.
        # There are two flat-field images (taken before and after the acquisition of ten slices),
        # we simply average them.
        dark = imageio.imread(glob.glob(current_path + dark_name)[0]) 
        flat1 = imageio.imread(glob.glob(current_path + flat_name[0])[0])
        flat2 = imageio.imread(glob.glob(current_path + flat_name[1])[0])
        flat = np.mean(np.array([ flat1, flat2 ]), axis=0 )

        # Read in the sinogram.
        sinogram = imageio.imread(glob.glob(current_path + sino_name)[0])
        sinogram =  np.ascontiguousarray(sinogram)
        
        # Change data type of the giles from uint16 to float32
        sinogram = sinogram.astype('float32')
        dark = dark.astype('float32')
        flat = flat.astype('float32')
        
        # Down-sample the sinogram as well as the dark and flat field
        # for i in np.arange(sino_dims[0]):
        sinogram = (sinogram[:,0::2]+sinogram[:,1::2])
        dark = (dark[0,0::2]+dark[0,1::2])
        flat = (flat[0,0::2]+flat[0,1::2])
            
        print('Shape of down-sampled sinogram:', sinogram.shape)
        print('Shape of down-sampled dark field:', dark.shape)
        print(dark[0],dark[-1],dark[-2])
        print('Shape of down-sampled flat field:', flat.shape)
        print(flat[0],flat[-1],flat[-2])

        # Subtract the dark field, devide by the flat field,
        # and take the negative log to linearize the data according to the Beer-Lambert law.
        data = sinogram - dark
        data = data/(flat-dark)

        # Exclude last projection if desired.
        if excludeLastPro:
            data = data[0:-1,:]

        # Create detector shift via linear grid interpolation.
        if i_slc in range(1,2830+1) or i_slc in range(5521,5870+1):
            detShift = corr[0] * detPix
        else:
            detShift = corr[1] * detPix

        detGrid = np.arange(0,956) * detPix
        detGridShifted = detGrid + detShift
        detShiftCorr = interp1d(detGrid, data, kind='linear', bounds_error=False, fill_value='extrapolate')
        data = detShiftCorr(detGridShifted)

        # Clip the data on the lower end to 1e-6 to avoid division by zero in next step.
        data = data.clip(1e-6, None)
        print("Values have been clipped from [", np.min(data), ",", np.max(data),"] to [1e-6,None]")

        # Take negative log.
        data = np.log(data)
        data = np.negative(data)
        data = np.ascontiguousarray(data)

        # Create array that stores the used projection angles.
        angles = np.linspace(0,2*np.pi, 3601) # 3601 = full width of sinograms.

        # Apply exclusion of last projection if desired.
        if excludeLastPro:
            angles = angles[0:-1]
            print('Excluded last projection.')

        # Apply angular subsampling.
        data = data[0::angSubSamp,:]
        angles = angles[0::angSubSamp]
        angInd = np.where(angles<=(maxAng/180*np.pi))
        angles = angles[angInd]
        data = data[:(angInd[-1][-1]+1),:]

        print('Data shape:', data.shape)
        print('Length angles:', len(angles))

        print('Loading and pre-processing done', flush=True)


        print('Computing reconstruction for slice', i_slc, '...', flush=True)

        # Create ASTRA objects for reconstruction.
        detSubSamp = 2
        binning = 1
        detPixSz = detSubSamp * binning * detPix
        SOD = 431.019989 
        SDD = 529.000488

        # Scale factor calculation.
        # ASTRA assumes that the voxel size is 1mm.
        # For this to be true we need to calculate a scale factor for the geometry.
        # This can be done by first calculating the 'true voxel size' via the intersect theorem
        # and then scaling the geometry accordingly.

        # Down-sampled width of the detector.
        nPix = 956
        det_width = detPixSz * nPix

        # Physical width of the field of view in the measurement plane via intersect theorem.
        FOV_width = det_width * SOD/SDD
        print('Physical width of FOV (in mm):', FOV_width)

        # True voxel size with a given number of voxels to be used.
        nVox = 1024
        voxSz = FOV_width / nVox
        print('True voxel size (in mm) for', nVox, 'voxels to be used:', voxSz)

        # Scaling the geometry accordingly.
        scaleFactor = 1./voxSz
        print('Self-calculated scale factor:', scaleFactor)
        SDD = SDD * scaleFactor
        SOD = SOD * scaleFactor
        detPixSz = detPixSz * scaleFactor

        # Create ASTRA objects.
        projGeo = astra.create_proj_geom('fanflat', detPixSz, 956, angles, SOD, SDD - SOD)
        volGeo = astra.create_vol_geom(recSz[0], recSz[1])
        recID = astra.data2d.create('-vol', volGeo)
        sinoID = astra.data2d.create('-sino', projGeo, data)
        projID   = astra.create_projector('cuda', projGeo, volGeo)
        A = astra.OpTomo(projID)

        # Create an ASTRA configuration using a registered plugin.
        # This configuration dictionary setups an algorithm,
        # a projection and a volume geometry and returns
        # an ASTRA algorithm, which can be run on its own.

        astra.plugin.register(NesterovGradient.AcceleratedGradientPlugin)
        proj_id = astra.create_projector('cuda', projGeo, volGeo)
        cfg_agd = astra.astra_dict('AGD-PLUGIN')
        cfg_agd['ReconstructionDataId'] = recID
        cfg_agd['ProjectionDataId'] = sinoID
        cfg_agd['ProjectorId'] = proj_id
        cfg_agd['option'] = {}
        cfg_agd['option']['MinConstraint'] = 0

        # Create and run algorithm.
        algID = astra.algorithm.create(cfg_agd)
        iterations = maxIter
        astra.algorithm.run(algID, iterations)

        # Receive reconstruction.
        rec = astra.data2d.get(recID)
        rec = np.maximum(rec,0)

        # Cut the reconstruction to the desired area of (1024,1024).
        rec_cut = rec[511:1535,511:1535]
        print('Shape of cut out reconstruction area:', rec_cut.shape)

        # Save reconstruction.
        imageio.imwrite(str(save_dir_str + 'slice' + str(i_slc).zfill(5) + '/' 'mode' + str(i_mode)+'/reconstruction.tif'),(rec_cut.astype(np.float32)).reshape(outSz))
        
        
        # Clean up.
        astra.algorithm.delete(algID)
        astra.data2d.delete(recID)
        astra.data2d.delete(sinoID)
        astra.projector.delete(proj_id)

print(np.round_(time.time() - t, 3), 'sec elapsed for reconstructing', len(slice_id), 'slices.')