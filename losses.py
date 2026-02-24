import csv
import torch
import torch.autograd as autograd
import numpy as np
from model import Model 
from metrics import draw_curve


type_mapping = {
    0: "Lorentzienne",
    1: "Gaussienne",
}

def loss_pinn(x_collocation, A, fwhm, spectre, type_tensor):
    metabolites = ["PCr + Cr 2", "GSH + Glu + Gln", "MIns 2", "MIns 1", "PCh + GPC", "PCr + Cr 1", 
                   "Asp", "NAA 2", "Gln", "Glu", "NAA 1", "Lac", "Lip"]
    center_values = [3.92, 3.768, 3.615, 3.52, 3.21, 3.02, 2.8, 2.57, 2.43, 2.34, 2.0, 1.3, 0.9]
    
    # Detach tensors from gradient computation and convert to numpy arrays
    x_collocation_np = x_collocation.detach().numpy().reshape(-1)
    A_np = A.detach().numpy()
    fwhm_np = fwhm.detach().numpy()
    spectre_np = spectre.detach().numpy()
    type_tensor_np = type_tensor.detach().numpy()
    
    ppm_data = torch.linspace(0, 10, 8192)
    
    type_tensor_np = np.where(type_tensor_np >= 0.5, 1, 0)
    
    avg_loss_coloc = 0.0
    avg_loss_physics = 0.0
    max_loss_coloc = 0.0
    max_loss_physics = 0.0
    for idx in range(A_np.shape[0]):
        fit = torch.zeros_like(ppm_data)
        
        
        for metab_idx, metab_name in enumerate(metabolites):
            fit += draw_curve(ppm_data, A_np[idx, metab_idx], fwhm_np[idx, metab_idx], 
                              center_values[metab_idx], type_tensor_np[idx, metab_idx])
        
        # Calculate loss at collocation points
        fit_collocation = np.interp(x_collocation_np, ppm_data.detach().numpy(), fit.numpy())
         # Print shapes for debugging
        

        spectre_flat = spectre_np[idx].flatten()
        spectre_collocation = np.interp(x_collocation_np, ppm_data.detach().numpy(), spectre_flat)
        
        loss_coloc = np.sum((spectre_collocation - fit_collocation) ** 2) / len(x_collocation_np)
        loss_physics = np.sum((fit.numpy() - spectre_flat) ** 2) / len(ppm_data)
        max_loss_coloc = max(max_loss_coloc, loss_coloc)
        max_loss_physics = max(max_loss_physics, loss_physics)
       
    
        # Average loss over all batches
        avg_loss_coloc += loss_coloc / max_loss_coloc
        avg_loss_physics += loss_physics / max_loss_physics
    
    return np.array(avg_loss_coloc),np.array(avg_loss_physics)  