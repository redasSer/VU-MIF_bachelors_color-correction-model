import torch
import numpy as np
import colour
import pytorch_ssim


class CustomLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        input_np = input.detach().cpu()
        target_np = target.detach().cpu()

        input_np = input_np.numpy()
        target_np = target_np.numpy()

        deltaE = calculateDeltaE2(input_np, target_np)
        ssim = ssim_loss(input, target)

        ssim = transformed_ssim(ssim)
        # print(f"deltaE: {deltaE} SSIM: {ssim}")
        
        loss_value = deltaE / ssim 

        ctx.save_for_backward(input, target)
        
        return torch.tensor(loss_value, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        
        grad_input = grad_output * 2 * (input - target) / input.numel()
        
        return grad_input, torch.neg(grad_input) 

class MyLoss(torch.nn.Module):
    def forward(self, input, target):
        return CustomLossFunction.apply(input, target)
    

def calculateDeltaE2(np_img1, np_img2):
    image_A = np_img1.squeeze(0)
    image_B = np_img2.squeeze(0)

    image_A = np.transpose(image_A, (1, 2, 0))
    image_B = np.transpose(image_B, (1, 2, 0))

    image_A = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(image_A))
    image_B = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(image_B))

    delta_e = np.mean(colour.difference.delta_E_CIE2000(image_A, image_B))

    return delta_e

ssim_loss = pytorch_ssim.SSIM()

def transformed_ssim(ssim):
    if ssim < 0:
        abs = torch.abs(ssim)
        diff = 1 - abs
        return diff*diff*0.01
    elif ssim == 0: 
      return torch.tensor(0.01)
    else:
        return ssim