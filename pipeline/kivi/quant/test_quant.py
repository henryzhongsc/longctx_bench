import os
import ipdb
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import time
import triton
import numpy as np
from new_pack import _pack_along_last_dim, _minmax_along_last_dim


# def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
# 	s_time = time.time()
# 	torch.cuda.synchronize()
# 	assert len(data.shape) == 4
# 	shape = data.shape
# 	B, nh, D, T = shape
# 	# ================== Get Scale & Zeros ===============
# 	assert T % group_size == 0
# 	num_groups = T // group_size
# 	new_shape = (B * nh * D, num_groups, group_size)
# 	scale_mn_shape = B, nh, D, num_groups
# 	# Quantize
# 	data = data.reshape(new_shape)
# 	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
# 	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
# 	BLOCK_SIZE_N = 128
# 	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], BLOCK_SIZE_N),)
# 	_minmax_along_last_dim[grid](data, mn, mx,
# 							 data.numel(), data.shape[0], num_groups, group_size,
# 							 BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=8) 
# 	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
# 	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
# 	scale = (mx - mn) / (2 ** bit - 1)
# 	data = data - mn.unsqueeze(-1)
# 	data.div_(scale.unsqueeze(-1))
# 	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
# 	data = data.view(-1, T)
# 	feat_per_int = 32 // bit
# 	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
# 	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
# 	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
# 	torch.cuda.synchronize()
# 	m_time = time.time()
# 	print('minmax used time:', m_time - s_time)
# 	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
# 								data.shape[1], feat_per_int, 
# 								BLOCK_SIZE_N=BLOCK_SIZE_N, 
# 								num_warps=8)
# 	torch.cuda.synchronize()
# 	e_time = time.time()
# 	print('pack used time:', e_time - m_time)
# 	import ipdb; ipdb.set_trace()
# 	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


def triton_quantize_and_pack_along_last_dim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.reshape(new_shape)
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], 32),)
	ipdb.set_trace()
	_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=32, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], 128), data.shape[1] // feat_per_int,)
	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=128, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


def triton_quantize_and_pack_along_second_lastdim(data: torch.Tensor, group_size: int, bit: int):
	assert len(data.shape) == 4
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	assert T % group_size == 0
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups
	# Quantize
	data = data.view(new_shape)
	import ipdb; ipdb.set_trace()
	mx = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	mn = torch.empty((B * nh * D, num_groups), device=data.device, dtype=data.dtype)
	grid = lambda meta: (triton.cdiv(data.shape[0]*data.shape[1], 32),)
	_minmax_along_last_dim[grid](data, mn, mx,
							 data.numel(), data.shape[0], num_groups, group_size,
							 BLOCK_SIZE_N=32, num_warps=8) 
	# mn = torch.min(data, dim=-1, keepdim=True)[0].squeeze(-1)
	# mx = torch.max(data, dim=-1, keepdim=True)[0].squeeze(-1)
	scale = (mx - mn) / (2 ** bit - 1)
	data = data - mn.unsqueeze(-1)
	data.div_(scale.unsqueeze(-1))
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32)
	data = data.view(-1, T)
	feat_per_int = 32 // bit
	packshape = (np.prod(shape[:-1]), shape[-1] // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	grid = lambda meta: (triton.cdiv(data.shape[0], 128), data.shape[1] // feat_per_int,)
	_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=128, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)


if __name__ == "__main__":
	nh, seqlen = 32, 8192
	v_data = torch.randn(1, nh, seqlen, 128).cuda()
	k_data = torch.randn(1, nh, 128, seqlen).cuda()
	nrepeat = 200
	st = time.time()
	torch.cuda.synchronize()
	for _ in range(nrepeat):
	    out = triton_quantize_and_pack_along_second_lastdim(k_data, 32, 2)
	torch.cuda.synchronize()
	print(f'32 layer kcache used time: {(time.time() - st) / nrepeat * 1000 * 32} ms')
	
	# st = time.time()
	# torch.cuda.synchronize()
	# for _ in range(nrepeat):
	#     out = triton_quantize_and_pack_along_last_dim(v_data, 32, 2)
	# print(f'32 layer vcache used time: {(time.time() - st) / nrepeat * 1000 * 32} ms')