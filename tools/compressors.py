"""
compressors.py  -  Unified Activation Compressor Library
========================================================

This module provides three activation compression methods extracted from
the MDI distributed inference project, wrapped behind a uniform API:

  1. TopKCompressor       - Top-K sparsification with PackBits mask encoding
  2. QuantizationCompressor - Uniform quantization (FP16 / INT8 / INT4 / INT2)
  3. LLMInt8Compressor    - LLM.int8()-style hybrid mixed-precision compression

All compressors share the same interface:

    compressed = compressor.compress(tensor, compression_params)
    restored   = compressor.decompress(compressed, device='cpu')
    nbytes     = compressor.get_compressed_size(compressed)
    name       = compressor.get_name()

Quick-start
-----------
    from compressors import TopKCompressor, QuantizationCompressor, LLMInt8Compressor

    # --- TopK: keep 50% of elements ---
    topk = TopKCompressor()
    data = topk.compress(tensor, 0.5)               # compression_params = k (float)
    out  = topk.decompress(data, device='cuda')

    # --- Quantization: k selects bit-width automatically ---
    #   0.5 -> FP16, 0.25 -> INT8, 0.125 -> INT4, 0.0625 -> INT2
    quant = QuantizationCompressor()
    data  = quant.compress(tensor, 0.25)            # compression_params = k (float)
    out   = quant.decompress(data)

    # --- LLM.int8(): separate outliers (FP16) from regular values (INT8) ---
    llm = LLMInt8Compressor()
    data = llm.compress(tensor, [0.01, 'fp16', 'int8'])
    #   compression_params = [k, outlier_precision, regular_precision]
    #   k = outlier ratio (e.g. 0.01 = top 1% as outliers)
    out  = llm.decompress(data)

Dependencies: torch, numpy
"""

import torch
import numpy as np


# ============================================================================
# Base class - defines the shared interface
# ============================================================================

class BaseCompressor:
    """
    Abstract base class for all compressors.

    Every subclass must implement:
        compress(tensor, compression_params)  -> dict
        decompress(compressed, device)        -> torch.Tensor
        get_compressed_size(compressed)       -> int   (bytes)
        get_name()                            -> str
    """

    def compress(self, tensor: torch.Tensor, compression_params) -> dict:
        raise NotImplementedError

    def decompress(self, compressed: dict, device: str = 'cpu') -> torch.Tensor:
        raise NotImplementedError

    def get_compressed_size(self, compressed: dict) -> int:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError


# ============================================================================
# TopK Compressor (PackBits)
# ============================================================================

class TopKCompressor(BaseCompressor):
    """
    Top-K sparsification compressor with bit-packed masks.

    How it works:
    -------------
    Compression:
      1. Flatten each sample to a 1-D vector of length N.
      2. Select the top-K elements by absolute value (K = N * k_ratio).
      3. Build a boolean mask of shape [batch, N] indicating selected positions.
      4. Pack the mask with numpy.packbits (8x reduction on the mask itself).
      5. Store the selected values in positional order plus the packed mask.

    Decompression:
      1. Unpack the mask with numpy.unpackbits.
      2. Allocate a zero tensor of the original shape.
      3. Fill values back into the True positions of the mask.
      4. Reshape to the original tensor shape.

    Compression ratio:
      approx  k + 1/8   (values + 1-bit mask)
      e.g. k=0.5 -> ~62.5% of original size

    Parameters
    ----------
    (no parameters - compression_params is passed per-call)
    """

    def compress(self, tensor, compression_params):
        """
        Compress an activation tensor by keeping only the top-K elements.

        Args:
            tensor: Input tensor, typically [batch, C, H, W] (FP32).
            compression_params: float, fraction of elements to keep
                                (0 < k <= 1).

        Returns:
            dict with keys: method, values, packed_mask, shape,
                            elements_per_sample, k.
        """
        k_ratio = compression_params  # float: fraction of elements to keep
        original_shape = tensor.shape
        batch_size = original_shape[0]

        # Flatten to [batch, N]
        reshaped = tensor.reshape(batch_size, -1)
        elements_per_sample = reshaped.shape[1]
        k = max(1, int(elements_per_sample * k_ratio))

        # Per-sample top-K by absolute value
        _, top_indices = torch.topk(reshaped.abs(), k, dim=1)

        # Boolean mask [batch, N]
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(1, top_indices, True)

        # Gather values in positional order (sorted indices)
        sorted_indices = torch.sort(top_indices, dim=1)[0]
        sorted_values = torch.gather(reshaped, 1, sorted_indices)

        # Pack mask: 8 bools -> 1 byte
        mask_np = mask.cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np, axis=1)

        return {
            'method': 'packbits_simple',
            'values': sorted_values.cpu().numpy(),
            'packed_mask': packed_mask,
            'shape': original_shape,
            'elements_per_sample': elements_per_sample,
            'k': k,
        }

    def decompress(self, compressed, device='cpu'):
        """
        Decompress a TopK-compressed payload back to a full tensor.

        Args:
            compressed: dict returned by compress().
            device: Target torch device.

        Returns:
            Decompressed tensor with zeros in pruned positions.
        """
        values = torch.from_numpy(compressed['values']).to(device)
        packed_mask = compressed['packed_mask']
        shape = compressed['shape']
        elements_per_sample = compressed['elements_per_sample']
        batch_size = shape[0]

        # Unpack mask
        mask_np = np.unpackbits(packed_mask, axis=1)[:, :elements_per_sample]
        mask = torch.from_numpy(mask_np).bool().to(device)

        # Scatter values into a zero tensor
        reshaped = torch.zeros(batch_size, elements_per_sample,
                               device=device, dtype=values.dtype)
        reshaped[mask] = values.flatten()
        return reshaped.reshape(shape)

    def get_compressed_size(self, compressed):
        """Return total compressed payload size in bytes."""
        values_bytes = compressed['values'].nbytes
        mask_bytes = compressed['packed_mask'].nbytes
        metadata_bytes = 16  # shape, elements_per_sample, k
        return values_bytes + mask_bytes + metadata_bytes

    def get_name(self):
        return "TopK"


# ============================================================================
# Quantization Compressor  (FP16 / INT8 / INT4 / INT2)
# ============================================================================
#
# Internally delegates to one of four sub-compressors based on k value:
#
#   k == 0.5    -> _QuanFP16  (FP32 -> FP16,  2x compression)
#   k == 0.25   -> _QuanInt8  (FP32 -> UINT8, 4x compression)
#   k == 0.125  -> _QuanInt4  (FP32 -> 4-bit, 8x compression)
#   k == 0.0625 -> _QuanInt2  (FP32 -> 2-bit, 16x compression)
#   other       -> _QuanFP16  (default fallback)
#
# All sub-quantizers assume non-negative activations (post-ReLU) and use
# unsigned quantization ranges for maximum precision.


class _QuanFP16:
    """
    FP16 quantizer: cast FP32 -> FP16 and back.

    Compression ratio: exactly 2x.
    Precision loss: minimal (half-precision rounding only).
    """

    def compress(self, tensor, compression_params):
        return {
            'method': 'quan_fp16',
            'values': tensor.half().cpu().numpy(),
            'shape': tensor.shape,
        }

    def decompress(self, data, device='cpu'):
        values = torch.from_numpy(data['values'])
        return values.float().to(device).reshape(data['shape'])

    def get_compressed_size(self, data):
        return data['values'].nbytes + 16


class _QuanInt8:
    """
    INT8 symmetric quantizer.
    
    Steps:
      1. Compute scale = abs_max(tensor) / 127.0
      2. Quantize: round(tensor / scale).clamp(-127, 127).int8
      3. Dequantize: tensor_int8.float() * scale
    """

    def compress(self, tensor, compression_params):
        shape = tensor.shape
        abs_max = tensor.abs().max()
        scale = float(abs_max) / 127.0 if abs_max > 0 else 1.0
        
        quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        
        return {
            'method': 'quan_int8',
            'values': quantized.cpu().numpy(),
            'scale': scale,
            'shape': shape,
        }

    def decompress(self, data, device='cpu'):
        values = torch.from_numpy(data['values']).to(device)
        return (values.float() * data['scale']).reshape(data['shape'])

    def get_compressed_size(self, data):
        return data['values'].nbytes + 16 + 4  # +4 for scale


class _QuanInt4:
    """
    4-bit symmetric quantizer with stochastic rounding.

    Steps:
      1. Compute scale = abs_max(tensor) / 7.0
      2. Stochastic rounding:
         x = tensor / scale
         floor_x = floor(x), prob = x - floor_x
         quantized = floor_x + Bernoulli(prob)
      3. Clamp to [-7, 7], shift by +7 to [0, 14] and bit-pack: two 4-bit values per uint8 byte.
      4. Dequantize: unpack, sub 7, multiply by scale.
    """

    def compress(self, tensor, compression_params):
        shape = tensor.shape
        abs_max = tensor.abs().max()
        scale = float(abs_max) / 7.0 if abs_max > 0 else 1.0

        # Stochastic rounding
        x = tensor / scale
        floor_x = x.floor()
        prob = x - floor_x
        quantized = floor_x + torch.bernoulli(prob).to(x.device)
        
        quantized = quantized.clamp(-7, 7).to(torch.int8)
        shifted = (quantized + 7).to(torch.uint8)

        # Bit-pack: 2 x 4-bit -> 1 x uint8
        flat = shifted.flatten()
        n = flat.numel()
        padding = (2 - n % 2) % 2
        if padding:
            flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)])
            
        pairs = flat.reshape(-1, 2).cpu().numpy()
        packed = ((pairs[:, 0] << 4) | (pairs[:, 1] & 0x0F)).astype(np.uint8)

        return {
            'method': 'quan_int4',
            'packed_values': packed,
            'scale': scale,
            'shape': shape,
            'n_elements': n,
            'padding': padding,
        }

    def decompress(self, data, device='cpu'):
        packed = torch.from_numpy(data['packed_values']).to(device)
        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        unpacked = torch.stack([high, low], dim=1).flatten()
        if data['padding']:
            unpacked = unpacked[:data['n_elements']]
            
        quantized = unpacked.to(torch.int8) - 7
        return (quantized.float() * data['scale']).reshape(data['shape'])

    def get_compressed_size(self, data):
        return data['packed_values'].nbytes + 16 + 4 + 4 + 4


class _QuanInt2:
    """
    2-bit symmetric quantizer.

    Steps:
      1. Compute scale = abs_max(tensor) / 1.0
      2. Quantize: clamp(-1, 1).int8
      3. Shift by +1, bit-pack four 2-bit values per uint8 byte.
    """

    def compress(self, tensor, compression_params):
        shape = tensor.shape
        abs_max = tensor.abs().max()
        scale = float(abs_max) if abs_max > 0 else 1.0

        x = tensor / scale
        quantized = x.round().clamp(-1, 1).to(torch.int8)
        shifted = (quantized + 1).to(torch.uint8)

        flat = shifted.flatten()
        n = flat.numel()
        padding = (4 - n % 4) % 4
        if padding:
            flat = torch.cat([flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)])
            
        quads = flat.reshape(-1, 4).cpu().numpy()
        packed = ((quads[:, 0] << 6) | (quads[:, 1] << 4) |
                  (quads[:, 2] << 2) | quads[:, 3]).astype(np.uint8)

        return {
            'method': 'quan_int2',
            'packed_values': packed,
            'scale': scale,
            'shape': shape,
            'n_elements': n,
            'padding': padding,
        }

    def decompress(self, data, device='cpu'):
        packed = torch.from_numpy(data['packed_values']).to(device)
        b0 = (packed >> 6) & 0x03
        b1 = (packed >> 4) & 0x03
        b2 = (packed >> 2) & 0x03
        b3 = packed & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=1).flatten()
        if data['padding']:
            unpacked = unpacked[:data['n_elements']]
            
        quantized = unpacked.to(torch.int8) - 1
        return (quantized.float() * data['scale']).reshape(data['shape'])

    def get_compressed_size(self, data):
        return data['packed_values'].nbytes + 16 + 4 + 4 + 4


class QuantizationCompressor(BaseCompressor):
    """
    Smart quantization compressor that selects bit-width from k value.

    How it works:
    -------------
    The compression_params (a float k) encodes the desired bit-width:

        k         Bit-Width   Method       Compression
        -------   ---------   ----------   -----------
        0.5       FP16        _QuanFP16    2x
        0.25      INT8        _QuanInt8    4x
        0.125     INT4        _QuanInt4    8x
        0.0625    INT2        _QuanInt2    16x
        other     FP16        (default)    2x

    Smaller k = more aggressive compression.

    Compression:
      Forward the call to the matching sub-quantizer.

    Decompression:
      Read the 'method' key in the payload and dispatch to the right
      sub-quantizer's decompress().

    Parameters
    ----------
    (none - bit-width selected at compress-time via compression_params)
    """

    _K_TO_METHOD = {
        0.5:    'fp16',
        0.25:   'int8',
        0.125:  'int4',
        0.0625: 'int2',
    }

    def __init__(self):
        self._fp16 = _QuanFP16()
        self._int8 = _QuanInt8()
        self._int4 = _QuanInt4()
        self._int2 = _QuanInt2()

    # ---- internal dispatch helpers ----

    def _select_compressor(self, k):
        """Pick the sub-quantizer matching k value (default: FP16)."""
        if abs(k - 0.0625) < 1e-6:
            return self._int2
        elif abs(k - 0.125) < 1e-6:
            return self._int4
        elif abs(k - 0.25) < 1e-6:
            return self._int8
        elif abs(k - 0.5) < 1e-6:
            return self._fp16
        else:
            return self._fp16  # fallback

    def _dispatch_decompress(self, data):
        """Pick the sub-quantizer matching the stored method tag."""
        method = data.get('method')
        if method == 'quan_int2':
            return self._int2
        elif method == 'quan_int4':
            return self._int4
        elif method == 'quan_int8':
            return self._int8
        else:
            return self._fp16

    # ---- public API ----

    def compress(self, tensor, compression_params):
        k = compression_params  # float: k value selecting bit-width
        return self._select_compressor(k).compress(tensor, k)

    def decompress(self, compressed, device='cpu'):
        return self._dispatch_decompress(compressed).decompress(compressed, device)

    def get_compressed_size(self, compressed):
        return self._dispatch_decompress(compressed).get_compressed_size(compressed)

    def get_name(self):
        return "Quantization"


# ============================================================================
# LLM.int8() Compressor  (Hybrid mixed-precision)
# ============================================================================

class LLMInt8Compressor(BaseCompressor):
    """
    LLM.int8()-style hybrid mixed-precision compressor.

    How it works:
    -------------
    Inspired by the LLM.int8() paper (Dettmers et al., 2022), this
    compressor separates activation values into two groups:

      * **Outliers** - the top ``outlier_ratio`` fraction of elements by
        absolute value.  These are stored at higher precision (FP16 or INT8)
        to preserve the large-magnitude information that dominates inference
        quality.

      * **Regular values** - everything else.  Quantized row-wise to a
        lower precision (INT8 / INT4 / INT2) using AbsMax scaling.

    Compression pipeline:
      1. Compute a dynamic threshold = quantile(|tensor|, 1 - outlier_ratio).
      2. Build a 1-bit bitmask: 1 = outlier, 0 = regular.
      3. Extract outlier values and quantize to ``outlier_precision``.
      4. Zero-out outlier positions in the tensor, reshape to [rows, hidden],
         compute per-row AbsMax scale, and quantize to ``regular_precision``.
      5. Pack the bitmask with numpy.packbits.
      6. Store everything in a dict.

    Decompression pipeline:
      1. Unpack the bitmask.
      2. Dequantize regular values (int -> float, * scale), reshape to
         original shape.
      3. Dequantize outlier values and scatter them back into the outlier
         positions via masked_scatter_.

    Compressed payload layout:
      +------------------+---------+-----------------------------+
      | Component        | Size    | Notes                       |
      +------------------+---------+-----------------------------+
      | Bitmask          | N/8     | 1 bit per element           |
      | Outlier values   | varies  | FP16 (2B) or INT8 (1B) ea. |
      | Regular values   | varies  | INT8/INT4/INT2 per element  |
      | Row scales       | R * 4   | FP32 per row                |
      | Metadata         | ~64     | shape, numel, threshold ... |
      +------------------+---------+-----------------------------+

    compression_params : list
        A list of [k, outlier_precision, regular_precision]:
          - k (float): outlier ratio, i.e. fraction of elements treated
            as outliers (e.g. 0.01 = top 1%).
          - outlier_precision (str): precision for outlier storage,
            'fp16' or 'int8'.
          - regular_precision (str): precision for regular (non-outlier)
            storage, 'fp16', 'int8', 'int4', or 'int2'.
    """

    def compress(self, tensor, compression_params):
        """
        Compress a tensor using hybrid mixed-precision quantization.

        Args:
            tensor: Input FP32 tensor of any shape (last dim = hidden size).
            compression_params: list of [k, outlier_precision, regular_precision]
                - k (float): outlier ratio, fraction of elements treated
                  as outliers (e.g. 0.01 = top 1%).
                - outlier_precision (str): precision for outlier storage,
                  'fp16' or 'int8'.
                - regular_precision (str): precision for regular storage,
                  'fp16', 'int8', 'int4', or 'int2'.

        Returns:
            dict with compressed payload and statistics.
        """
        # ---- Parse compression parameters ----
        outlier_ratio, outlier_precision, regular_precision = compression_params

        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        shape = tensor.shape
        original_bytes = tensor.numel() * 4  # FP32 = 4 bytes/element

        # ---- 1. Build outlier bitmask ----
        # Dynamic threshold: percentile(|tensor|, 1 - outlier_ratio)
        percentile = 1.0 - outlier_ratio
        threshold = torch.quantile(
            tensor.abs().float().reshape(-1), percentile
        ).item()

        mask_bool = tensor.abs() > threshold
        num_outliers = mask_bool.sum().item()

        # ---- 2. Quantize outlier values ----
        outlier_raw = torch.masked_select(tensor, mask_bool)

        if outlier_precision == 'fp16':
            outlier_values = outlier_raw.half()
            size_outliers = outlier_values.numel() * 2
            outlier_scale = None
        elif outlier_precision == 'int8':
            abs_max = outlier_raw.abs().max()
            outlier_scale = abs_max / 127.0 if abs_max > 0 else torch.tensor(1.0)
            outlier_values = (outlier_raw / outlier_scale).round().clamp(-127, 127).to(torch.int8)
            size_outliers = outlier_values.numel()
        else:
            raise ValueError(f"Unsupported outlier_precision: {outlier_precision}")

        # ---- 3. Quantize regular values (row-wise AbsMax) ----
        tensor_regular = tensor.clone()
        tensor_regular.masked_fill_(mask_bool, 0.0)

        # Reshape to [rows, hidden] for row-wise scaling
        flattened = tensor_regular.view(-1, shape[-1])
        abs_max = flattened.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)

        regular_padding = 0
        if regular_precision == 'int8':
            scales = abs_max / 127.0
            quantized_regular = (flattened / scales).round().clamp(-127, 127).to(torch.int8)
            size_regular = quantized_regular.numel()

        elif regular_precision == 'int4':
            scales = abs_max / 7.0
            x = flattened / scales
            # Stochastic rounding for int4
            floor_x = x.floor()
            prob = x - floor_x
            quantized = floor_x + torch.bernoulli(prob).to(x.device)
            quantized = quantized.clamp(-7, 7).to(torch.int8)
            shifted = (quantized + 7).to(torch.uint8)  # shift to [0, 14]
            flat = shifted.flatten()
            if flat.numel() % 2 != 0:
                flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8,
                                                    device=flat.device)])
                regular_padding = 1
            # Pack two 4-bit values into one uint8
            quantized_regular = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
            size_regular = quantized_regular.numel() * 0.5

        elif regular_precision == 'int2':
            scales = abs_max / 1.0
            x = flattened / scales
            quantized = x.round().clamp(-1, 1).to(torch.int8)
            shifted = (quantized + 1).to(torch.uint8)  # shift to [0, 2]
            flat = shifted.flatten()
            regular_padding = (4 - flat.numel() % 4) % 4
            if regular_padding:
                flat = torch.cat([flat, torch.zeros(regular_padding,
                                                    dtype=torch.uint8,
                                                    device=flat.device)])
            # Pack four 2-bit values into one uint8
            quantized_regular = ((flat[0::4] << 6) | (flat[1::4] << 4) |
                                 (flat[2::4] << 2) | flat[3::4])
            size_regular = quantized_regular.numel() * 0.25
        else:
            raise ValueError(f"Unsupported regular_precision: {regular_precision}")

        # ---- 4. Pack bitmask (1 bit per element) ----
        mask_np = mask_bool.reshape(-1).cpu().numpy().astype(np.uint8)
        packed_mask = np.packbits(mask_np)

        # ---- 5. Compute compressed size ----
        size_mask = packed_mask.nbytes
        size_scales = scales.numel() * 4
        metadata_bytes = 64
        if outlier_precision == 'int8':
            metadata_bytes += 4  # outlier_scale

        compressed_bytes = (size_mask + size_outliers + size_regular
                            + size_scales + metadata_bytes)
        compression_ratio = compressed_bytes / original_bytes if original_bytes else 1.0

        return {
            'method': 'llmint8_hybrid',
            'packed_mask': packed_mask,
            'outlier_values': outlier_values.cpu().numpy(),
            'outlier_scale': (outlier_scale.item()
                              if outlier_scale is not None else None),
            'main_values': quantized_regular.cpu().numpy(),
            'main_scales': scales.cpu().numpy(),
            'regular_padding': regular_padding,
            'shape': shape,
            'numel': tensor.numel(),
            'threshold': threshold,
            'num_outliers': num_outliers,
            'outlier_ratio_actual': (num_outliers / tensor.numel()
                                     if tensor.numel() else 0.0),
            'compressed': True,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': compression_ratio,
            'outlier_precision': outlier_precision,
            'regular_precision': regular_precision,
        }

    def decompress(self, data, device='cpu'):
        """
        Decompress a payload produced by compress().

        Args:
            data: dict returned by compress().
            device: Target torch device.

        Returns:
            Restored FP32 tensor.
        """
        # Pass-through payload
        if not data.get('compressed', False):
            return data['tensor'].to(device)

        shape = data['shape']
        numel = data['numel']

        # ---- 1. Unpack bitmask ----
        mask_flat = np.unpackbits(data['packed_mask'])[:numel]
        mask = torch.from_numpy(mask_flat.copy()).view(shape).to(device).bool()

        # ---- 2. Dequantize regular values ----
        main_scales = torch.from_numpy(data['main_scales'].copy()).to(device).float()
        reg_prec = data.get('regular_precision', 'int8')

        if reg_prec == 'int8':
            main_vals = torch.from_numpy(data['main_values'].copy()).to(device).float()
            restored = main_vals.view(-1, shape[-1]) * main_scales

        elif reg_prec == 'int4':
            packed = torch.from_numpy(data['main_values'].copy()).to(device)
            high = (packed >> 4).to(torch.int8)
            low = (packed & 0x0F).to(torch.int8)
            unpacked = torch.stack([high, low], dim=1).flatten()
            if data['regular_padding']:
                unpacked = unpacked[:-data['regular_padding']]
            quantized = unpacked - 7  # undo shift
            restored = quantized.float().view(-1, shape[-1]) * main_scales

        elif reg_prec == 'int2':
            packed = torch.from_numpy(data['main_values'].copy()).to(device)
            p0 = (packed >> 6) & 0x03
            p1 = (packed >> 4) & 0x03
            p2 = (packed >> 2) & 0x03
            p3 = packed & 0x03
            unpacked = torch.stack([p0, p1, p2, p3], dim=1).flatten()
            if data['regular_padding']:
                unpacked = unpacked[:-data['regular_padding']]
            quantized = unpacked.to(torch.int8) - 1  # undo shift
            restored = quantized.float().view(-1, shape[-1]) * main_scales

        restored = restored.view(shape)

        # ---- 3. Scatter outlier values back ----
        if data['outlier_values'].size > 0:
            o_prec = data.get('outlier_precision', 'fp16')
            if o_prec == 'fp16':
                outlier_vals = torch.from_numpy(
                    data['outlier_values'].copy()).to(device).float()
            elif o_prec == 'int8':
                outlier_vals = (torch.from_numpy(
                    data['outlier_values'].copy()).to(device).float()
                    * data['outlier_scale'])
            restored.masked_scatter_(mask, outlier_vals)

        return restored

    def get_compressed_size(self, compressed):
        """Return the total compressed payload size in bytes."""
        if not compressed.get('compressed', False):
            return compressed['original_bytes']
        return int(compressed['compressed_bytes'])

    def get_name(self):
        return "LLMInt8"


# ============================================================================
# Convenience factory
# ============================================================================

def get_compressor(name):
    """
    Factory function to create a compressor by name.

    Args:
        name: One of 'topk', 'quantization', 'llmint8'.

    Returns:
        An instance of the requested compressor.

    Example:
        comp = get_compressor('llmint8')
        data = comp.compress(tensor, [0.01, 0.25, 'fp16'])
    """
    name = name.lower().strip()
    if name == 'topk':
        return TopKCompressor()
    elif name in ('quantization', 'quant'):
        return QuantizationCompressor()
    elif name in ('llmint8', 'llm_int8', 'llm.int8'):
        return LLMInt8Compressor()
    else:
        raise ValueError(
            f"Unknown compressor '{name}'. "
            f"Choose from: topk, quantization, llmint8"
        )


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Compressor self-test")
    print("=" * 60)

    # Create a dummy activation tensor (simulating post-ReLU values)
    torch.manual_seed(42)
    tensor = torch.relu(torch.randn(2, 64, 8, 8))  # [batch=2, C=64, H=8, W=8]
    original_bytes = tensor.numel() * 4
    print(f"Input shape: {tensor.shape},  size: {original_bytes} bytes\n")

    # --- TopK ---
    for k in [0.5, 0.25, 0.125]:
        comp = TopKCompressor()
        data = comp.compress(tensor, k)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        print(f"[{comp.get_name()}] k={k:.3f}  "
              f"ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}")

    print()

    # --- Quantization ---
    for k in [0.5, 0.25, 0.125, 0.0625]:
        comp = QuantizationCompressor()
        data = comp.compress(tensor, k)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = comp.get_compressed_size(data) / original_bytes
        label = {0.5: 'FP16', 0.25: 'INT8', 0.125: 'INT4', 0.0625: 'INT2'}[k]
        print(f"[{comp.get_name()}] {label:<5} k={k:.4f}  "
              f"ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}")

    print()

    # --- LLM.int8() ---
    #   compression_params = [k, outlier_precision, regular_precision]
    #   k = outlier ratio (fraction of elements treated as outliers)
    configs = [
        [0.01, 'fp16', 'int8'],   # 1% outliers at FP16, regular at INT8
        [0.05, 'fp16', 'int4'],   # 5% outliers at FP16, regular at INT4
        [0.10, 'int8', 'int2'],   # 10% outliers at INT8, regular at INT2
    ]
    for params in configs:
        comp = LLMInt8Compressor()
        data = comp.compress(tensor, params)
        out = comp.decompress(data)
        err = (tensor - out).abs().mean().item()
        ratio = data['compression_ratio']
        print(f"[{comp.get_name()}] k={params[0]:.2f} "
              f"out={params[1]} reg={params[2]}  "
              f"ratio={ratio:.3f}  MAE={err:.6f}  shape_ok={out.shape == tensor.shape}")

    print("\nAll tests passed.")
