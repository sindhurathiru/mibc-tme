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
  return rawData

 def removeOutlier(rawData):
  """
  Find and remove any outliers in data
  """
  outlierCheck = rawData[rawData.columns[:23]]
  from scipy import stats
  nonOutliers = (np.abs(stats.zscore(outlierCheck)) < 4).all(axis=1)
  xRemOutlier = rawData[nonOutliers]
  xRemOutlier = xRemOutlier.reset_index(drop=True)
  return xRemOutlier

def trainTestSplit(xRemOutlier):
  """
  Split data into training and testing sets.
  """
  from sklearn.model_selection import train_test_split
  from imblearn.over_sampling import SMOTE

  # Splitting data using stratified random sampling
  labelData = xRemOutlier["Cell Type"]
  trainData = xRemOutlier.drop(["ROI", "Cell Type", "Patient", "CYSvTUR"], 1)
  x_train, x_test, y_train, y_test = train_test_split(trainData, labelData, test_size=0.33, random_state=42, stratify=labelData)

  # Balance datasets
  x_train, y_train = SMOTE().fit_resample(x_train, y_train)

  return x_train, x_test, y_train, y_test

def deepAutoencoder(x_train, x_test, y_train, y_test):
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  
  from tensorflow.keras.callbacks import EarlyStopping

  adam = tf.keras.optimizers.Adam(learning_rate=0.00005)
  bottleneck_size = 2 # number of dimensions to view latent space in; same as "code size"
  input_data = Input(shape=(23,)) # number of columns

  encoded = Dense(18, activation='relu')(input_data)
  encoded = Dense(12, activation='relu')(encoded)
  encoded = Dense(6, activation='relu')(encoded)
  encoded = Dense(bottleneck_size, activation='linear')(encoded)
  encoder = Model(input_data, encoded)

  encoded_input = Input(shape=(bottleneck_size,))
  decoded = Dense(6, activation='relu')(encoded_input)
  decoded = Dense(12, activation='relu')(decoded)
  decoded = Dense(18, activation='relu')(decoded)
  decoded = Dense(23, activation='sigmoid')(decoded)
  decoder = Model(encoded_input, decoded)

  output_data = decoder(encoder(input_data))
  ae = Model(input_data, output_data)
  ae.compile(optimizer=adam, loss='binary_crossentropy') # using binary cross entropy since input values are in range [0,1]

  early_stopping = EarlyStopping(patience=10)
  history = ae.fit(x_train, x_train, epochs=15000, batch_size=256, validation_data=(x_test, x_test), callbacks=[early_stopping])

  encoded_data = encoder.predict(x_test)
  decoded_data = decoder.predict(encoded_data)
  latentSpaceData = encoder.predict(trainData)
  
  return history, encoded_data, decoded_data
  
def plot_metric(history, metric):
  """
  Plot loss
  """
  train_metrics = history.history[metric]
  val_metrics = history.history['val_'+metric]
  epochs = range(2, len(train_metrics) + 1)
  plt.plot(epochs, train_metrics[1:])
  plt.plot(epochs, val_metrics[1:])
  plt.title('Training and validation '+ metric)
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend(["train_"+metric, 'val_'+metric])
  plt.show()
    
def latentSpace(latentSpaceData, xRemOutlier):
  """
  Add latent space coordinates to dataset.
  """
  latentSpaceDf = pd.DataFrame(latentSpaceData)
  latentSpaceDf["Cell Type"] = xRemOutlier["Cell Type"]
  latentSpaceDf["CYSvTUR"] = xRemOutlier["CYSvTUR"]
  latentSpaceDf["Patient"] = xRemOutlier["Patient"]
  latentSpaceDf["ROI"] = xRemOutlier["ROI"]

  latentSpaceDf.columns = ['Dim 1', 'Dim 2', "Cell Type", "CYSvTUR", "Patient", "ROI"]
  ctypes = le.inverse_transform(latentSpaceDf["Cell Type"])
  latentSpaceDf["Cell Type"] = ctypes

  return latentSpaceDf
  
  
def main():
  path = r"C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Processed Features\\Slide_1\\Non-compensated Labeled Data" # use your path
  path2 = r"C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Processed Features\\Slide_2\\Non-compensated Labeled Data" # use your path
  path3 = r"C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Processed Features\\Slide_3\\Clustered Data" # use your path

  rawData = preprocess(path, path2, path3)
  xRemOutlier = removeOutlier(rawData)
  x_train, x_test, y_train, y_test = trainTestSplit(xRemOutlier)
  history, encoded_data, decoded_data, latentSpaceData = deepAutoencoder(x_train, x_test, y_train, y_test)
  plot_metric(history, 'loss')
  latentSpaceDf = latentSpace(latentSpaceData, xRemOutlier)
  latentSpaceDf.to_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\CYS vs TUR\\AE Plots\\Slide 118\\full_latent_space_data.csv")

