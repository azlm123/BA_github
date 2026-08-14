import runpy
import os
import sys
import numpy as np

import pandas as pd


THIS_DIR = os.path.dirname(__file__)
REPO_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_DIR)
# load extraction functions from parent directory module file
this_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(this_dir)
mod = runpy.run_path(os.path.join(repo_dir, 'dd_hdg_element_extraction.py'))
extract = mod['extract_subelements_data']
reconstruct = mod['reconstruct_from_subelements']
import fem_data
# parameters
nelx = 8
nely = 6
nselx = 2
nsely = 3

# create single-sample node fields: values = y*100 + x
Y = np.arange(nely+1)[:, None]
X = np.arange(nelx+1)[None, :]
U = (Y * 100 + X).astype(float)
F = (Y * 100 + X + 0.5).astype(float)

print('Full U:')
print(U)
print('\nFull F:')
print(F)

# extract
F_sub, U_sub = extract(U, F, nselx, nsely, nely, nelx)
print('\nF_sub shape:', F_sub.shape)
print('U_sub shape:', U_sub.shape)

# show first patch
print('\nFirst patch F_sub[0,0]:')
print(F_sub[0,0,:,:,0])
print('\nPatch (iy=1,ix=0) F_sub[1,0]:')
print(F_sub[1,0,:,:,0])

# reconstruct
F_rec, U_rec = reconstruct(F_sub, U_sub)
print('\nReconstructed F:')
print(F_rec)

print('\nReconstruction equal:', np.allclose(F_rec, F) and np.allclose(U_rec, U))

U, F = fem_data.generate_poisson_data(10, 0.1, nely, nelx)
pd.DataFrame(U[:,:,0]).to_csv('U_sample.csv', index=False)
pd.DataFrame(F[:,:,0]).to_csv('F_sample.csv', index=False)
pd.DataFrame(U[:,:,1]).to_csv('U_sample_1.csv', index=False)
pd.DataFrame(F[:,:,1]).to_csv('F_sample_1.csv', index=False)