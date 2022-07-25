// This file is part of meshoptimizer library and is distributed under the terms of MIT License.
// Copyright (C) 2016-2021, by Arseny Kapoulkine (arseny.kapoulkine@gmail.com)
var MeshoptEncoder = (function() {
	"use strict";

	// Built with clang version 14.0.4 (https://github.com/llvm/llvm-project 29f1039a7285a5c3a9c353d054140bf2556d4c4d)
	// Built from meshoptimizer 0.17
	var wasm = "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FHRAOCUFHOXBKKAVC+QDF8kJJJJBK/uLEVUCUAICDTGVAICffffI0eGOCBYD1+KJJBhJJJJBBHRALALYD9gGWCDTFARbDBALAWCEFbD9gABARbDBAOCBYD1+KJJBhJJJJBBHRALALYD9gGOCDTFARbDBALAOCEFbD9gABARbDLCUADCDTADCffffI0eCBYD1+KJJBhJJJJBBHRALALYD9gGOCDTFARbDBALAOCEFbD9gABARbDWABYDBCBAVZ+TJJJB8aADCI9uHWDNADtMBABYDBHOAEHLADHRINAOALYDBCDTFGVAVYDBCEFbDBALCLFHLARCUFGRMBKKDNAItMBABYDBHLABYDLHRCBHVAIHOINARAVbDBARCLFHRALYDBAVFHVALCLFHLAOCUFGOMBKKDNADCI6MBAWCEAWCE0eHdABYDLHRABYDWHVCBHLINAECWFYDBHOAECLFYDBHDARAEYDBCDTFGWAWYDBGWCEFbDBAVAWCDTFALbDBARADCDTFGDADYDBGDCEFbDBAVADCDTFALbDBARAOCDTFGOAOYDBGOCEFbDBAVAOCDTFALbDBAECXFHEAdALCEFGL9HMBKKDNAItMBABYDLHEABYDBHLINAEAEYDBALYDB9rbDBALCLFHLAECLFHEAICUFGIMBKKKqBABAEADAIC+01JJBZ+MJJJBKqBABAEADAIC+c+JJJBZ+MJJJBK9dEEUABCfEAICDTZ+TJJJBHLCBHIDNADtMBINDNALAEYDBCDTFGBYDBCU9HMBABAIbDBAICEFHIKAECLFHEADCUFGDMBKKAIK9TEIUCBCBYD+M+KJJBGEABCIFC98gFGBbD+M+KJJBDNDNABzBCZTGD9NMBCUHIABAD9rCffIFCZ4NBCUsMEKAEHIKAIK/lEEEUDNDNAEABvCIgtMBABHIXEKDNDNADCZ9PMBABHIXEKABHIINAIAEYDBbDBAICLFAECLFYDBbDBAICWFAECWFYDBbDBAICXFAECXFYDBbDBAICZFHIAECZFHEADC9wFGDCS0MBKKADCL6MBINAIAEYDBbDBAECLFHEAICLFHIADC98FGDCI0MBKKDNADtMBINAIAErBB86BBAICEFHIAECEFHEADCUFGDMBKKABK/AEEDUDNDNABCIgtMBABHIXEKAECfEgC+B+C+EW2HLDNDNADCZ9PMBABHIXEKABHIINAIALbDBAICXFALbDBAICWFALbDBAICLFALbDBAICZFHIADC9wFGDCS0MBKKADCL6MBINAIALbDBAICLFHIADC98FGDCI0MBKKDNADtMBINAIAE86BBAICEFHIADCUFGDMBKKABK9TEIUCBCBYD+M+KJJBGEABCIFC98gFGBbD+M+KJJBDNDNABzBCZTGD9NMBCUHIABAD9rCffIFCZ4NBCUsMEKAEHIKAIK9+EIUzBHEDNDNCBYD+M+KJJBGDAECZTGI9NMBCUHEADAI9rCffIFCZ4NBCUsMEKADHEKCBABAE9rCIFC98gCBYD+M+KJJBFGDbD+M+KJJBDNADzBCZTGE9NMBADAE9rCffIFCZ4NB8aKKXBABAEZ+YJJJBK+BEEIUDNAB+8GDCl4GICfEgGLCfEsMBDNALMBDNABjBBBB9cMBAECBbDBABSKABjBBJ9fnAEZ+XJJJBHBAEAEYDBCNFbDBABSKAEAICfEgC+CUFbDBADCfff+D94gCJJJ/4Iv++HBKABK+gEBDNDNAECJE9IMBABjBBBUnHBDNAECfE9PMBAEC+BUFHEXDKABjBBBUnHBAECPDAECPD6eC+C9+FHEXEKAEC+BU9KMBABjBBJXnHBDNAEC+b9+9NMBAEC/mBFHEXEKABjBBJXnHBAEC+299AEC+2990eC/MEFHEKABAEClTCJJJ/8IF++nKK+eDDBCJWK+EDB4+H9W9n94+p+Gw+J9o+YE9pBBBBBBEBBBDBBBEBBBDBBBBBBBDBBBBBBBEBBBBBBB+L29Hz/69+9Kz/n/76z/RG97z/Z/O9Xz8j/b85z/+/U9Yz/B/K9hz+2/z9dz9E+L9Mz59a8kz+R/t3z+a+Zyz79ohz/J4++8++y+d9v8+BBBB9S+49+z8r+Hbz9m9m/m8+l/Z/O8+/8+pg89Q/X+j878r+Hq8++m+b/E87BBBBBBJzBBJzBBJz+e/v/n8++y+dSz9I/h/68+XD/r8+/H0838+/w+nOzBBBB+wv9o8+UF888+9I/h/68+9C9g/l89/N/M9M89/d8kO8+BBBBF+8Tz9M836zs+2azl/Zpzz818ez9E+LXz/u98f8+819e/68+BC+EQKXEBBBDBBBAwBB";

	// Used to unpack wasm
	var wasmpack = new Uint8Array([32,0,65,2,1,106,34,33,3,128,11,4,13,64,6,253,10,7,15,116,127,5,8,12,40,16,19,54,20,9,27,255,113,17,42,67,24,23,146,148,18,14,22,45,70,69,56,114,101,21,25,63,75,136,108,28,118,29,73,115]);

	if (typeof WebAssembly !== 'object') {
		// This module requires WebAssembly to function
		return {
			supported: false,
		};
	}

	var instance;

	var promise =
		WebAssembly.instantiate(unpack(wasm), {})
		.then(function(result) {
			instance = result.instance;
			instance.exports.__wasm_call_ctors();
			instance.exports.meshopt_encodeVertexVersion(0);
			instance.exports.meshopt_encodeIndexVersion(1);
		});

	function unpack(data) {
		var result = new Uint8Array(data.length);
		for (var i = 0; i < data.length; ++i) {
			var ch = data.charCodeAt(i);
			result[i] = ch > 96 ? ch - 71 : ch > 64 ? ch - 65 : ch > 47 ? ch + 4 : ch > 46 ? 63 : 62;
		}
		var write = 0;
		for (var i = 0; i < data.length; ++i) {
			result[write++] = (result[i] < 60) ? wasmpack[result[i]] : (result[i] - 60) * 64 + result[++i];
		}
		return result.buffer.slice(0, write);
	}

	function assert(cond) {
		if (!cond) {
			throw new Error("Assertion failed");
		}
	}

	function bytes(view) {
		return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
	}

	function reorder(indices, vertices, optf) {
		var sbrk = instance.exports.sbrk;
		var ip = sbrk(indices.length * 4);
		var rp = sbrk(vertices * 4);
		var heap = new Uint8Array(instance.exports.memory.buffer);
		var indices8 = bytes(indices);
		heap.set(indices8, ip);
		if (optf) {
			optf(ip, ip, indices.length, vertices);
		}
		var unique = instance.exports.meshopt_optimizeVertexFetchRemap(rp, ip, indices.length, vertices);
		// heap may have grown
		heap = new Uint8Array(instance.exports.memory.buffer);
		var remap = new Uint32Array(vertices);
		new Uint8Array(remap.buffer).set(heap.subarray(rp, rp + vertices * 4));
		indices8.set(heap.subarray(ip, ip + indices.length * 4));
		sbrk(ip - sbrk(0));

		for (var i = 0; i < indices.length; ++i)
			indices[i] = remap[indices[i]];

		return [remap, unique];
	}

	function encode(fun, bound, source, count, size) {
		var sbrk = instance.exports.sbrk;
		var tp = sbrk(bound);
		var sp = sbrk(count * size);
		var heap = new Uint8Array(instance.exports.memory.buffer);
		heap.set(bytes(source), sp);
		var res = fun(tp, bound, sp, count, size);
		var target = new Uint8Array(res);
		target.set(heap.subarray(tp, tp + res));
		sbrk(tp - sbrk(0));
		return target;
	}

	function maxindex(source) {
		var result = 0;
		for (var i = 0; i < source.length; ++i) {
			var index = source[i];
			result = result < index ? index : result;
		}
		return result;
	}

	function index32(source, size) {
		assert(size == 2 || size == 4);
		if (size == 4) {
			return new Uint32Array(source.buffer, source.byteOffset, source.byteLength / 4);
		} else {
			var view = new Uint16Array(source.buffer, source.byteOffset, source.byteLength / 2);
			return new Uint32Array(view); // copies each element
		}
	}

	function filter(fun, source, count, stride, bits, insize) {
		var sbrk = instance.exports.sbrk;
		var tp = sbrk(count * stride);
		var sp = sbrk(count * insize);
		var heap = new Uint8Array(instance.exports.memory.buffer);
		heap.set(bytes(source), sp);
		fun(tp, count, stride, bits, sp);
		var target = new Uint8Array(count * stride);
		target.set(heap.subarray(tp, tp + count * stride));
		sbrk(tp - sbrk(0));
		return target;
	}

	return {
		ready: promise,
		supported: true,
		reorderMesh: function(indices, triangles, optsize) {
			var optf = triangles ? (optsize ? instance.exports.meshopt_optimizeVertexCacheStrip : instance.exports.meshopt_optimizeVertexCache) : undefined;
			return reorder(indices, maxindex(indices) + 1, optf);
		},
		encodeVertexBuffer: function(source, count, size) {
			assert(size > 0 && size <= 256);
			assert(size % 4 == 0);
			var bound = instance.exports.meshopt_encodeVertexBufferBound(count, size);
			return encode(instance.exports.meshopt_encodeVertexBuffer, bound, source, count, size);
		},
		encodeIndexBuffer: function(source, count, size) {
			assert(size == 2 || size == 4);
			assert(count % 3 == 0);
			var indices = index32(source, size);
			var bound = instance.exports.meshopt_encodeIndexBufferBound(count, maxindex(indices) + 1);
			return encode(instance.exports.meshopt_encodeIndexBuffer, bound, indices, count, 4);
		},
		encodeIndexSequence: function(source, count, size) {
			assert(size == 2 || size == 4);
			var indices = index32(source, size);
			var bound = instance.exports.meshopt_encodeIndexSequenceBound(count, maxindex(indices) + 1);
			return encode(instance.exports.meshopt_encodeIndexSequence, bound, indices, count, 4);
		},
		encodeGltfBuffer: function(source, count, size, mode) {
			var table = {
				ATTRIBUTES: this.encodeVertexBuffer,
				TRIANGLES: this.encodeIndexBuffer,
				INDICES: this.encodeIndexSequence,
			};
			assert(table[mode]);
			return table[mode](source, count, size);
		},
		encodeFilterOct: function(source, count, stride, bits) {
			assert(stride == 4 || stride == 8);
			assert(bits >= 1 && bits <= 16);
			return filter(instance.exports.meshopt_encodeFilterOct, source, count, stride, bits, 16);
		},
		encodeFilterQuat: function(source, count, stride, bits) {
			assert(stride == 8);
			assert(bits >= 4 && bits <= 16);
			return filter(instance.exports.meshopt_encodeFilterQuat, source, count, stride, bits, 16);
		},
		encodeFilterExp: function(source, count, stride, bits) {
			assert(stride > 0 && stride % 4 == 0);
			assert(bits >= 1 && bits <= 24);
			return filter(instance.exports.meshopt_encodeFilterExp, source, count, stride, bits, stride);
		},
	};
})();

// UMD-style export for MeshoptEncoder
if (typeof exports === 'object' && typeof module === 'object')
	module.exports = MeshoptEncoder;
else if (typeof define === 'function' && define['amd'])
	define([], function() {
		return MeshoptEncoder;
	});
else if (typeof exports === 'object')
	exports["MeshoptEncoder"] = MeshoptEncoder;
else
	(typeof self !== 'undefined' ? self : this).MeshoptEncoder = MeshoptEncoder;
