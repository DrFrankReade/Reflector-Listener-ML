# SPDX-License-Identifier: GPL-2.0-or-later
# Inference snapshot; see SOURCE_PROVENANCE.json.
import torch

def linear_fft_filter(x, kernel):
    n=x.shape[-1];m=kernel.shape[-1]
    size=1 << (n+m-2).bit_length()
    # FIR is symmetric, so convolution equals the frozen conv1d correlation.
    result=torch.fft.irfft(torch.fft.rfft(x,size)*torch.fft.rfft(kernel.reshape(-1),size),size)
    return result[:,m//2:m//2+n]
