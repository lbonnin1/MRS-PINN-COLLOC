import csv
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from training import loader_test,type_mapping,train
import torch
import json
from model import Model
from config import dossier_test,dossier_train ,transform,batch,pourcentage_train,num_output,epochs,criterion_amplitude,criterion_fit,criterion_spectrum,criterion_type,criterion_Width,output_size
import torch
import json
import seaborn as sns
from metrics import gaussian_area,lorentzian_area,draw_curve, mse,mean_absolute_percentage_error,r_squared

model_state_dict = torch.load(r'')

# Initialize the model and load the state dictionary
model = Model(13)
model.load_state_dict(model_state_dict)




# List of metabolites and center values
metabolites = ["PCr + Cr 2","GSH + Glu + Gln","MIns 2","MIns 1","PCh + GPC","PCr + Cr 1","Asp","NAA 2","Gln","Glu","NAA 1","Lac","Lip"]
center_values = [3.92, 3.768, 3.615, 3.52, 3.21, 3.02, 2.8, 2.57, 2.43, 2.34, 2, 1.3, 0.9]
basisset='' #path to the basis set csv file
predictions = []
ppm_data=[]
val_true_areas = []
val_pred_areas = []
train_true_areas = []
train_pred_areas = []

# Set the model to evaluation mode
model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(loader_test):

        batch_data = []
        
        for sample_idx, input_spectre in enumerate(batch):
            input_spectre = input_spectre.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
            Amplitude, Width, reconstructed_spectrum, type = model(input_spectre)
            
         
            # Convert tensors to numpy arrays
            Amplitude = Amplitude.cpu().numpy()
            Width = Width.cpu().numpy()
            type = type.cpu().numpy()
            type_tensor_np = np.where(type >= 0.5, 1, 0)
    
            for j in range(Amplitude.shape[1]):  # Iterate over elements in the batch
                if type[batch_idx, j].item() == 1:  # Assuming '1' represents Gaussian type  
                    pred_area = gaussian_area(Amplitude[batch_idx, j], Width[batch_idx, j])
                else:  # Lorentzian
                   
                    pred_area = lorentzian_area(Amplitude[batch_idx, j])
                val_pred_areas.append(pred_area)
        
            metabolites_data = []
            for metabolite_idx, metabolite in enumerate(metabolites):
                try:
                    width = float(Width[0][metabolite_idx])
                    amplitude = float(Amplitude[0][metabolite_idx])
                    center = center_values[metabolite_idx]
                    type_value =float( type_tensor_np[0][metabolite_idx])
                    
                    # Create a dictionary for each metabolite
                    metabolite_data = {
                        "name": metabolite,
                        "Width": width,
                        "Amplitude": amplitude,
                        "Center": center,
                        "Type": type_value
                    }
                    
                    # Append the metabolite data to the list
                    metabolites_data.append(metabolite_data)
                except IndexError as e:
                    print(f"IndexError: {e}")
                    continue
     
        
            # Write the batch data to a unique JSON file for each batch
            json_file_path = f'parameters_batch_250{sample_idx}.json'
            with open(json_file_path, "w") as json_file:
                json.dump(metabolites_data, json_file, indent=4)

print("Data saved to JSON files")
#uploading the basisset csv file and retrieving the first column values which represents the ppm range values
with open(basisset, 'r') as file:
    reader = csv.reader(file,delimiter=';')
    basisset_data= next(reader)
    for row in reader:
        ppm_data.append(float(row[0])) 
    ppm_data=np.array(ppm_data)
fit = np.zeros_like(ppm_data)

for metab_data in metabolites_data:
    fit += draw_curve(ppm_data, metab_data['Amplitude'], metab_data['Center'], metab_data['Width'], metab_data['Type'])
spectre=np.array(input_spectre.reshape(1,-1)).flatten()
print(spectre)
mse_fit = mse(spectre, fit)
print("MSE for fit:", mse_fit)

# Calculate and print R-squared for fit
r_squared_fit = r_squared(spectre, fit)
print("R-squared for fit:", r_squared_fit)


# Calculate and print MAPE for fit
mape_fit = mean_absolute_percentage_error(spectre, fit)
print("MAPE for fit:", mape_fit)


            
# Plotting the results
plt.plot(ppm_data, fit, label='Fit')
"""plt.plot(ppm_data, fit_lcmodel, label='Fit LCModel')"""
plt.plot(ppm_data, input_spectre.squeeze().cpu().numpy(), label='Spectre')
plt.xlabel('PPM')
plt.ylabel('Intensity')
plt.xlim(4, 0)
plt.title(f'Metabolite Spectra for Batch {batch_idx}, Sample {sample_idx}')
plt.legend()
plt.show()

print("Data saved to JSON files")

