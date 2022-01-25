import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os

# Pre-processing of raw data
path = r"C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Processed Features\\Slide_1\\Non-compensated Labeled Data" # use your path
path2 = r"C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Processed Features\\Slide_2\\Non-compensated Labeled Data" # use your path
path3 = r"C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Processed Features\\Slide_3\\Clustered Data" # use your path

def preprocess(path, path2, path3):
  """
  Function retrieves mean intensity and cell type data from all biopsy and cystectomy files.
  """
  
  all_files = glob.glob(path + "/*.csv")
  li = []
  for filename in all_files:
      df = pd.read_csv(filename, index_col=None, header=0)
      df["ROI"] = df["ROI"][0] + "_1"
      df["CYSvTUR"] = "CYS"
      li.append(df)
  slide1Data = pd.concat(li, axis=0, ignore_index=True)

  all_files2 = glob.glob(path2 + "/*.csv")
  li2 = []
  for filename in all_files2:
      df = pd.read_csv(filename, index_col=None, header=0)
      df["ROI"] = df["ROI"][0] + "_2"
      df["CYSvTUR"] = "CYS"
      li2.append(df)
  slide2Data = pd.concat(li2, axis=0, ignore_index=True)

  all_files3 = glob.glob(path3 + "/*.csv")
  li3 = []
  for filename in all_files3:
      df = pd.read_csv(filename, index_col=None, header=0)
      df["ROI"] = filename.split("\\")[19][33:].split(".")[0] + "_3"
      df["CYSvTUR"] = "TUR"
      li3.append(df)
  slide3Data = pd.concat(li3, axis=0, ignore_index=True)

  data = pd.concat([slide1Data, slide2Data, slide3Data], axis=0, ignore_index=True)

  rawData = data.replace('HLA-DR+ cell ', 'HLA-DR+ cell')
  rawData = rawData.drop(["Cell Label", "Cluster Label", "190BCKG_190BCKG.ome.tiff", "191Ir_191Ir-DNA1.ome.tiff", 
                          "193Ir_193Ir-DNA2.ome.tiff"], 1)

  # Remove cells that are unknown
  rawData = rawData[rawData['Cell Type'] != "Unknown"]
  rawData = rawData[rawData['Cell Type'] != "Immune cell"]
  rawData = rawData[rawData['Cell Type'] != "Proliferating Cell"]

  # Normalize data between 0 and 1
  normRows = rawData.columns[0:-3]
  for i in normRows:
      max_value = rawData[i].max()
      min_value = rawData[i].min()
      rawData[i] = (rawData[i] - min_value) / (max_value - min_value)
      rawData

  rawData["Patient"] = ""

  # ROI 1
  rawData['Patient'][rawData['ROI'] == 'ROI005_ROI_005_1']= "32"
  rawData['Patient'][rawData['ROI'] == 'ROI006_ROI_006_1']= "32"
  rawData['Patient'][rawData['ROI'] == 'ROI007_ROI_007_1']= "60"
  rawData['Patient'][rawData['ROI'] == 'ROI008_ROI_008_1']= "60"
  rawData['Patient'][rawData['ROI'] == 'ROI009_ROI_009_1']= "8"
  rawData['Patient'][rawData['ROI'] == 'ROI010_ROI_010_1']= "8"
  rawData['Patient'][rawData['ROI'] == 'ROI011_ROI_011_1']= "56"
  rawData['Patient'][rawData['ROI'] == 'ROI031_ROI_012 - split_1']= "56"
  rawData['Patient'][rawData['ROI'] == 'ROI013_ROI_013_1']= "33"
  rawData['Patient'][rawData['ROI'] == 'ROI014_ROI_014_1']= "33"
  rawData['Patient'][rawData['ROI'] == 'ROI015_ROI_015_1']= "6"
  rawData['Patient'][rawData['ROI'] == 'ROI016_ROI_016_1']= "6"
  rawData['Patient'][rawData['ROI'] == 'ROI017_ROI_017_1']= "59"
  rawData['Patient'][rawData['ROI'] == 'ROI018_ROI_018_1']= "59"
  rawData['Patient'][rawData['ROI'] == 'ROI021_ROI_021_1']= "23"
  rawData['Patient'][rawData['ROI'] == 'ROI022_ROI_022_1']= "23"
  rawData['Patient'][rawData['ROI'] == 'ROI023_ROI_023_1']= "54"
  rawData['Patient'][rawData['ROI'] == 'ROI024_ROI_024_1']= "54"
  rawData['Patient'][rawData['ROI'] == 'ROI027_ROI_027_1']= "64"
  rawData['Patient'][rawData['ROI'] == 'ROI028_ROI_028_1']= "64"


  # ROI 2
  rawData['Patient'][rawData['ROI'] == 'ROI005_ROI_005_2']= "22"
  rawData['Patient'][rawData['ROI'] == 'ROI006_ROI_006_2']= "22"
  rawData['Patient'][rawData['ROI'] == 'ROI009_ROI_009_2']= "45"
  rawData['Patient'][rawData['ROI'] == 'ROI010_ROI_010_2']= "45"
  rawData['Patient'][rawData['ROI'] == 'ROI011_ROI_011_2']= "61"
  rawData['Patient'][rawData['ROI'] == 'ROI012_ROI_012_2']= "61"
  rawData['Patient'][rawData['ROI'] == 'ROI013_ROI_013_2']= "19"
  rawData['Patient'][rawData['ROI'] == 'ROI014_ROI_014_2']= "19"
  rawData['Patient'][rawData['ROI'] == 'ROI015_ROI_015_2']= "55"
  rawData['Patient'][rawData['ROI'] == 'ROI016_ROI_016_2']= "55"

  # ROI 3
  rawData['Patient'][rawData['ROI'] == 'ROI005_ROI_005_3']= "64"
  rawData['Patient'][rawData['ROI'] == 'ROI006_ROI_006_3']= "64"
  rawData['Patient'][rawData['ROI'] == 'ROI007_ROI_007_3']= "61"
  rawData['Patient'][rawData['ROI'] == 'ROI008_ROI_008_3']= "61"
  rawData['Patient'][rawData['ROI'] == 'ROI009_ROI_009_3']= "60"
  rawData['Patient'][rawData['ROI'] == 'ROI010_ROI_010_3']= "60"
  rawData['Patient'][rawData['ROI'] == 'ROI011_ROI_011_3']= "59"
  rawData['Patient'][rawData['ROI'] == 'ROI012_ROI_012_3']= "59"
  rawData['Patient'][rawData['ROI'] == 'ROI013_ROI_013_3']= "56"
  rawData['Patient'][rawData['ROI'] == 'ROI014_ROI_014_3']= "56"
  rawData['Patient'][rawData['ROI'] == 'ROI015_ROI_015_3']= "55"
  rawData['Patient'][rawData['ROI'] == 'ROI016_ROI_016_3']= "55"
  rawData['Patient'][rawData['ROI'] == 'ROI017_ROI_017_3']= "54"

  rawData['Patient'][rawData['ROI'] == 'ROI023_ROI_023_3']= "45"
  rawData['Patient'][rawData['ROI'] == 'ROI024_ROI_024_3']= "45"
  rawData['Patient'][rawData['ROI'] == 'ROI027_ROI_027_3']= "33"
  rawData['Patient'][rawData['ROI'] == 'ROI028_ROI_028_3']= "33"
  rawData['Patient'][rawData['ROI'] == 'ROI029_ROI_029_3']= "32"
  rawData['Patient'][rawData['ROI'] == 'ROI030_ROI_030_3']= "32"
  rawData['Patient'][rawData['ROI'] == 'ROI033_ROI_033_3']= "23"
  rawData['Patient'][rawData['ROI'] == 'ROI034_ROI_034_3']= "23"
  rawData['Patient'][rawData['ROI'] == 'ROI035_ROI_035_3']= "22"
  rawData['Patient'][rawData['ROI'] == 'ROI036_ROI_036_3']= "22"
  rawData['Patient'][rawData['ROI'] == 'ROI037_ROI_037_3']= "19"
  rawData['Patient'][rawData['ROI'] == 'ROI038_ROI_038_3']= "19"
  rawData['Patient'][rawData['ROI'] == 'ROI041_ROI_041_3']= "8"
  rawData['Patient'][rawData['ROI'] == 'ROI042_ROI_042_3']= "8"
  rawData['Patient'][rawData['ROI'] == 'ROI043_ROI_043_3']= "6"
  rawData['Patient'][rawData['ROI'] == 'ROI044_ROI_044_3']= "6"

  # Select only relevant cells
  relevCells = ["Tumour", "Proliferating Tumour", "Granzyme B Tumour", "Macrophage", "CD163 Macrophage", "Actin+", "Stroma", "CD4 T cell", "CD8 T cell", "T regulatory Cell"]
  rawData.loc[~rawData["Cell Type"].isin(relevCells), "Cell Type"] = "Other"
  rawData.loc[rawData["Cell Type"].isin(["Proliferating Tumour", "Granzyme B Tumour"]), "Cell Type"] = "Tumour"
  # # Combining macrophages and stromal cells
  rawData.loc[rawData["Cell Type"].isin(["CD163 Macrophage"]), "Cell Type"] = "Macrophage"
  rawData.loc[rawData["Cell Type"].isin(["Actin+"]), "Cell Type"] = "Stroma"
  # # Remove tumour cells
  # rawData = rawData[rawData['Cell Type'] != "Tumour"]

  # Encode cell types into integers
  cellTypes = list(np.unique(rawData["Cell Type"]))

  # Encode cell type strings to integer
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  rawData["Cell Type"] = le.fit_transform(rawData["Cell Type"])

  rawData = rawData[['141Pr_141Pr_alpha-actin.ome.tiff',
                     '142Nd_142Nd_CD66b.ome.tiff', '143Nd_143Nd-Vimentin.ome.tiff',
                     '144Nd_144Nd-CD14.ome.tiff', '147Sm_147Sm-CD163.ome.tiff', '148Nd_148Nd-PANCK.ome.tiff',
                     '149Sm_149Sm-CD11b.ome.tiff', '151Eu_151Eu-GATA3.ome.tiff',
                     '152Sm_152Sm-CD45.ome.tiff', '154Sm_154SmCD366TIM3.ome.tiff',
                     '155Gd_155Gd-FOXP3.ome.tiff',
                     '156Gd_156Gd-CD4.ome.tiff', '158Gd_158Gd-CD11c.ome.tiff',
                     '159Tb_159Tb-CD68.ome.tiff', '161Dy_161Dy-CD20.ome.tiff',
                     '162Dy_162Dy-CD8a.ome.tiff', 
                     '165Ho_165Ho-PD1.ome.tiff', 
                     '167Er_167Er-GRANZB.ome.tiff', '168Er_168Er-KI67.ome.tiff',
                     '169Tm_169Tm-DCLAMP.ome.tiff',
                     '170Er_170Er-CD3.ome.tiff', 
                     '173Yb_173Yb-CD45RO.ome.tiff', '174Yb_174Yb-HLA-DR.ome.tiff', 
                     "CYSvTUR", "Cell Type", "ROI", "Patient"]]

