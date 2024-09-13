# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2023.07.05 - Modified weight quantization
#              Meta Platforms, Inc. <zechunliu@meta.com>
#
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import ipdb
import torch
import torch.nn.functional as F
import torch.nn as nn


class AsymGroupedQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, group_size):
        ctx.save_for_backward(input, clip_val)
        assert len(input.shape) == 4
        bs, nh, seqlen, d = input.shape
        num_groups = d // group_size
        if num_groups * group_size != input.shape[-1]:
            raise ValueError("group_size should be a factor of the last dimension size")


        input_in_groups = input.view(bs, nh, seqlen, num_groups, group_size)

        #####
        # input_in_groups_cpy = input_in_groups.clone().detach()
        #####

        mx, mn = input_in_groups.max(dim=-1)[0], input_in_groups.min(dim=-1)[0]
        mx, mn = mx.unsqueeze(-1), mn.unsqueeze(-1)

        scale = (mx - mn) / (2 ** num_bits - 1)
        input_in_groups = (input_in_groups - mn) / scale
        input_in_groups = F.relu(input_in_groups)
        rounded_input_in_groups = input_in_groups.round_()
        dequantized_input_in_groups = rounded_input_in_groups * scale + mn
        dequantized_input = dequantized_input_in_groups.view(bs, nh, seqlen, -1)
        return dequantized_input
    
    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output

        # clip version
        # grad_input[input.ge(clip_val[1])] = 0
        # grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None

### group by channel
class AsymGroupedQuantizerByChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, group_size):
        ctx.save_for_backward(input, clip_val)
        assert len(input.shape) == 4
        bs, nh, num_instances, d = input.shape
        if num_instances % group_size != 0:
            new_num_instances = (num_instances // group_size + 1) * group_size
            delta = new_num_instances - num_instances
            # input = torch.cat([input,
            #                    input.min(dim=2, keepdim=True)[0].expand([bs, nh, delta, d])], 2)
            input = torch.cat([input,
                               torch.zeros([bs, nh, delta, d], dtype=input.dtype, device=input.device)], 2)
        else:
            new_num_instances = num_instances
        num_groups = new_num_instances // group_size
        input_in_groups = input.view(bs, nh, num_groups, group_size, d)
        mx, mn = input_in_groups.max(dim=-2)[0], input_in_groups.min(dim=-2)[0]
        mx, mn = mx.unsqueeze(-2), mn.unsqueeze(-2)
        scale = (mx - mn + 2e-6) / (2 ** num_bits - 1)
        input_in_groups = (input_in_groups - mn) / scale 
        # input_in_groups = torch.nan_to_num(input_in_groups, nan=0.0)
        input_in_groups = F.relu(input_in_groups)
        rounded_input_in_groups = input_in_groups.round_()
        dequantized_input_in_groups = rounded_input_in_groups * scale + mn
        dequantized_input = dequantized_input_in_groups.view(bs, nh, -1, d)
        if dequantized_input.isnan().any():
            import ipdb; ipdb.set_trace()
        if num_instances % group_size != 0:
            dequantized_input = dequantized_input[:, :, :num_instances, :]
        assert dequantized_input.shape == (bs, nh, num_instances, d)
        return dequantized_input


    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output
        return grad_input, None, None, None, None


@torch.no_grad()
def test_group_quantize():
    print('#' * 50 + 'Quant by Token' + '#' * 50)
    input = torch.randn((4, 32, 445, 128), dtype=torch.float16, device='cuda') * 100
    for num_bits, group_size in [ (2, 64), (4, 64), (8, 64), \
                                 (2, 128), (4, 128), (8, 128)]:
        output = AsymGroupedQuantizer.apply(input, None, num_bits, group_size)
        err = torch.mean(torch.abs(input - output)).item()
        print(num_bits, group_size, err)

    print('#' * 50 + 'Quant by Channel' + '#' * 50)
    input = torch.randn((4, 32, 333, 128), dtype=torch.float16, device='cuda') * 100
    for num_bits, group_size in [ (2, 64), (4, 64), (8, 64), \
                                 (2, 128), (4, 128), (8, 128)]:
        output = AsymGroupedQuantizerByChannel.apply(input, None, num_bits, group_size)
        err = torch.mean(torch.abs(input - output)).item()
        print(num_bits, group_size, err)


@torch.no_grad()
def process_input_by_channel(input, group_size):
    num_features = input.shape[-1]
    # input_flatten: [num_feats, bs * seqlen]
    input_flatten = input.view(-1, num_features).transpose(0, 1)
    num_instances = input_flatten.shape[-1]
    # Compute min, max by groups
    if num_instances % group_size != 0:
        # Padding
        new_num_instances = (num_instances // group_size + 1) * group_size
        delta = new_num_instances - num_instances
        input_flatten = torch.cat([input_flatten,
                                   torch.zeros([num_features, delta], dtype=input.dtype, device=input.device)], 1)
    input_groups = input_flatten.reshape(-1, group_size)
    mn, mx = torch.min(input_groups, 1)[0], torch.max(input_groups, 1)[0]
    return input_groups.view(num_features, -1, group_size), mn.view(num_features, -1), mx.view(num_features, -1)


@torch.no_grad()
def quant_by_channel_and_pack(input, group_size, num_bits, simulate=False):
    assert len(input.shape) == 3
    shape = input.shape
    ori_num_instances = shape[0] * shape[1]
    input_groups, mn, mx = process_input_by_channel(input, group_size)
    if simulate:
        mn, mx = mn.unsqueeze(-1), mx.unsqueeze(-1)
        scale = (mx - mn) / (2 ** num_bits - 1)
        input_groups = (input_groups - mn) / scale
        input_groups = F.relu(input_groups)
        rounded_input = input_groups.round_()
        return rounded_input, scale, mn
        # dequantized_input = rounded_input * scale + mn
        # dequantized_input = dequantized_input.view(input.shape[-1], -1)
        # if ori_num_instances != dequantized_input.shape[1]:
        #     dequantized_input = dequantized_input[:, 0:ori_num_instances]
        # dequantized_input = dequantized_input.transpose(0, 1).view(shape)
        # assert dequantized_input.shape == shape
        # return dequantized_input, scale, mn
    else:
        output, scale = dequant_cuda.pack_single_precision(input_groups, mn, mx, num_bits, False)
    assert len(scale.shape) >= 2 and len(mn.shape) >= 2
    if len(scale.shape) == 3:
        scale = scale.squeeze(-1)
    if len(mn.shape) == 3:
        mn = mn.squeeze(-1)
    return output, scale, mn


@torch.no_grad()
def dequantize_by_channel_and_unpack(data, group_size, shape, bits, scale, mn, simulate=False):
    num_feats = shape[-1]
    ori_num_instances = shape[0] * shape[1]
    if simulate:
        # import ipdb; ipdb.set_trace()
        data = data * scale + mn
    else:
        # Pad to group_size
        tot_num_instances = (ori_num_instances + (group_size - ori_num_instances % group_size) % group_size)

        # Unpack bitstream
        data = dequant_cuda.unpack_single_precision(data, bits, scale, mn, num_feats, tot_num_instances // group_size, group_size)
    dequantized_input = data.view(shape[-1], -1)
    if ori_num_instances != dequantized_input.shape[1]:
        dequantized_input = dequantized_input[:, 0:ori_num_instances]
    data = dequantized_input.transpose(0, 1).view(shape)
    return data



def test_channel_quantize():
    input = torch.randn((112, 334, 4096), dtype=torch.float16, device='cuda')
    shape = input.shape
    # for num_bits, group_size in [ (2, 64), (4, 64), (8, 64), \
    #                              (2, 128), (4, 128), (8, 128), \
    #                              (2, 256), (4, 256), (8, 256)]:
    for num_bits, group_size in [(2, 128), (4, 128)]:
        # fake_code, scale, mn = quant_by_channel_and_pack(input, group_size, num_bits, True)
        # output_fake = dequantize_by_channel_and_unpack(fake_code, group_size, shape, num_bits, scale, mn, True)
        # err = torch.mean(torch.abs(input - output_fake)).item()
        # print(num_bits, group_size, err)
        real_code, scale, mn = quant_by_channel_and_pack(input, group_size, num_bits, False)
        output_real = dequantize_by_channel_and_unpack(real_code, group_size, shape, num_bits, scale, mn, False)
        err = torch.mean(torch.abs(input - output_real)).item()
        print(num_bits, group_size, err)


if __name__ == '__main__':
    test_group_quantize()
    # test_channel_quantize()