import pandas as pd
from pathlib import Path

# read in labels per annotation
anndf = pd.read_csv(Path('resources') / 'annotation_df.csv')
measurementcols = ['volume', 'surface_area', 'diameter']
labelcols = ['sublety', 'internalstructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
noduledf  = anndf.groupby('nodule_id')[measurementcols+labelcols].agg(['mean', 'min', 'max'])
# noduledf  = anndf.groupby('nodule_id')[measurementcols+labelcols].agg(['mean', 'min', 'max'])
noduledf['filename'] = [str(x) + 'a1.npy' for x in noduledf.index.tolist()]

binarylabelcols = [x+'_binary' for x in labelcols]
noduledf[binarylabelcols] = (noduledf[labelcols] > 3).astype(int)
noduledf.to_csv(Path('data') / 'nodulelabels.csv', index=True, index_label='nodule_id')