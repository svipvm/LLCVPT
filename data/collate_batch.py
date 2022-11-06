# encoding: utf-8

import torch, numpy

def plain_collate_fn(batch):
    data_L = torch.cat([item['L'].unsqueeze(0) for item in batch], dim=0)
    data_H = torch.cat([item['H'].unsqueeze(0) for item in batch], dim=0)
    path_L = numpy.stack([item['L_path'] for item in batch], axis=0)
    path_H = numpy.stack([item['H_path'] for item in batch], axis=0)
    return data_L, data_H, path_L, path_H