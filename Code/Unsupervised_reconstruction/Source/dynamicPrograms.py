# Code base written by Andre He et al.

import torch
from torch import Tensor
from models.probcache import ProbCache
from source.editModel import EditModel
from models.models import SourceInferenceData, TargetInferenceData
from typing import Union

BIG_NEG = -1e9



def compute_mutation_prob(model:EditModel, sources_:SourceInferenceData, targets_:TargetInferenceData, return_posteriors:bool = False) -> Union[Tensor, ProbCache]:
    """
    Arguments:
        - model (EditModel) : the model predicting an edition from a reconstruction to its cognate in a given modern language
        - sources_ (list[SourceInferenceData]) : the already computed data about ancestral samples to do the inferences in the edit model.
        - targets_ : the data about the related forms in the modern language of the model. 
        return_posteriors (bool): if False, the function returns log(p(y|x)). Else, the backward dynamic program runs to compute the target probs, stocked in a ProbCache object (so it is expected B to equal 1).
    Computes log(p(y|x)) for each reconstruction or the targets probabilities for each sample.
    """
    rawSources_lengths, max_rawSource_sequenceLength = sources_[1]-2, sources_[2]-2
    rawTargets_lengths, max_rawTarget_sequenceLength = targets_[1]-1, targets_[2]-1
    batch_shape = rawSources_lengths.size() # (C, B), with C the number of cognates pairs and B the number of samples linked to the same cognate pair
    max_source_sequenceLength = max_rawSource_sequenceLength + 2
    max_target_sequenceLength = max_rawTarget_sequenceLength + 2
    
    model.cache_probs(sources_, targets_)

    # apparently faster in numpy for indexing but 
    
    f_sub = torch.full((max_source_sequenceLength, max_target_sequenceLength, *batch_shape), fill_value=BIG_NEG)
    f_ins = torch.full((max_source_sequenceLength, max_target_sequenceLength, *batch_shape), fill_value=BIG_NEG)
    # start by subbing ( with (. Now in insertion mode
    f_ins[0, 0] = 0.0
    
    for i in range(max_source_sequenceLength):   # start at i=0, might insert after (
        for j in range(max_target_sequenceLength):
            
            if j > 0:
                f_ins[i, j] = torch.logaddexp(
                    f_ins[i, j-1] + model.ins(i, j-1),  # inserted last char
                    f_sub[i, j-1] + model.sub(i, j-1)   # subbed last char
                )
            
            if i > 0:
                f_sub[i, j] = torch.logaddexp(
                    f_ins[i-1, j] + model.end(i-1, j),  # just ended insertion
                    f_sub[i-1, j] + model.dlt(i-1, j)   # deleted last char
                )
    
    """
    We consider x and y as the raw strings which respectively represent the source and the target. We also consider x_b and y_b as the same strings, but with the boundaries. The prob that y has emerged from x ( p(y|x) )is the total prob that ( '('+x+')' ) has been transformed into ( '('+y ) so that the final remaining operation is the substitution of ')' ( x_b[len(x)+1] ) into ')' ( y_b[len(x)+1] ). Then, the prob is given by f_sub[len(x)+1, len(y)], which justifies the final indexes below. 
    """
    source_final_idx = rawSources_lengths + 1
    target_final_idx = rawTargets_lengths

    if not return_posteriors:
        model.clear_cache()
        total_pr = f_sub[source_final_idx, target_final_idx.unsqueeze(1),
                        torch.arange(batch_shape[0]).unsqueeze(1), torch.arange(batch_shape[1]).unsqueeze(0)]
        return total_pr
    
    assert(batch_shape[-1]==1), 'The backward dynamic program only computes targets probs for one sampled proposal per cognate pair.'

    # the lengths are not according to the boundaries presence    
    f_sub_post = torch.full((max_source_sequenceLength, max_target_sequenceLength, *batch_shape), fill_value=BIG_NEG)
    f_ins_post = torch.full((max_source_sequenceLength, max_target_sequenceLength, *batch_shape), fill_value=BIG_NEG)
    f_sub_post[source_final_idx, target_final_idx.unsqueeze(1),
               torch.arange(batch_shape[0]).unsqueeze(1), torch.arange(batch_shape[1]).unsqueeze(0)] = 0.0

    posterior_cache = ProbCache(sources_, targets_, (batch_shape[0],1))

    for i in range(max_source_sequenceLength-1, -1, -1):
        for j in range(max_target_sequenceLength-1, -1, -1):

            if j < max_target_sequenceLength-1:
                ins_post_pr = (f_ins[i, j] + model.ins(i, j) - f_ins[i, j+1]) + f_ins_post[i, j+1]
                f_ins_post[i, j] = torch.logaddexp(f_ins_post[i, j], ins_post_pr)
                posterior_cache.ins[i, j] = ins_post_pr 

                sub_post_pr = (f_sub[i, j] + model.sub(i, j) - f_ins[i, j+1]) + f_ins_post[i, j+1] 
                f_sub_post[i, j] = torch.logaddexp(f_sub_post[i, j], sub_post_pr)
                posterior_cache.sub[i, j] = sub_post_pr

            if i < max_source_sequenceLength-1:
                dlt_post_pr = (f_sub[i, j] + model.dlt(i, j) - f_sub[i+1, j]) + f_sub_post[i+1, j]
                f_sub_post[i, j] = torch.logaddexp(f_sub_post[i, j], dlt_post_pr)
                posterior_cache.dlt[i, j] = dlt_post_pr

                end_post_pr = (f_ins[i, j] + model.end(i, j) - f_sub[i+1, j]) + f_sub_post[i+1, j]
                f_ins_post[i, j] = torch.logaddexp(f_ins_post[i, j], end_post_pr)
                posterior_cache.end[i, j] = end_post_pr

    model.clear_cache()
    return posterior_cache

def compute_posteriors(model:EditModel, sources_:SourceInferenceData, targets_:TargetInferenceData)->ProbCache:
    targetsProbsCache = compute_mutation_prob(model, sources_, targets_, return_posteriors = True)
    return targetsProbsCache #type: ignore