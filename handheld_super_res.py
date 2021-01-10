'''
Pseudo-code for handheld_super_res

'''

import calcGST
import lucas_kanade


def compute_local_gradients(raw_input_burst):
    return gradients

def compute_alignment_vectors(raw_input_burst):
    return align_vec

def compute_kernels(gradients):
    return kernel_output

def compute_local_stats(raw_input_burst):
    return local_stats

def compute_robustness(align_vec, local_stats):
    return robustness

def compute_contributions(kernel_output, robustness):
    for channel in ["R", "G", "B"]:
        for frame in frames[channel]:
            omega_hat   
            weight_frames += frame.compute(array_exp(-0.5 * transpose(frame_offset_vec) * inv(omega) * frame_offset_vec)
    return contrib
    
def generate_super_res(contrib):
    # Load burst of 15 images
    
    #
    
    
    #
    return output