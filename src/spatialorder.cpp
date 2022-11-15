// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <string.h>

// This work is based on:
// Fabian Giesen. Decoding Morton codes. 2009
namespace meshopt
{

template <typename T>
T part1By2(T x);

// "Insert" two 0 bits after each of the 10 low bits of x
template <>
inline uint32_t part1By2<uint32_t>(uint32_t x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x << 8)) & 0x0300f00f;  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x << 4)) & 0x030c30c3;  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x << 2)) & 0x09249249;  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

template <>
inline uint64_t part1By2<uint64_t>(uint64_t x)
{
    x = x & 0x5555555555555555;
    x = (x | (x >> 1))  & 0x3333333333333333;
    x = (x | (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4))  & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8))  & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return x;
}

static void computeOrder(datatype_t* result, const real_t* vertex_positions_data, size_t vertex_count, size_t vertex_positions_stride)
{
	size_t vertex_stride_real = vertex_positions_stride / sizeof(real_t);

	real_t minv[3] = {REAL_MAX, REAL_MAX, REAL_MAX};
	real_t maxv[3] = {-REAL_MAX, -REAL_MAX, -REAL_MAX};

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const real_t* v = vertex_positions_data + i * vertex_stride_real;

		for (int j = 0; j < 3; ++j)
		{
			real_t vj = v[j];

			minv[j] = minv[j] > vj ? vj : minv[j];
			maxv[j] = maxv[j] < vj ? vj : maxv[j];
		}
	}

	real_t extent = 0.0;

	extent = (maxv[0] - minv[0]) < extent ? extent : (maxv[0] - minv[0]);
	extent = (maxv[1] - minv[1]) < extent ? extent : (maxv[1] - minv[1]);
	extent = (maxv[2] - minv[2]) < extent ? extent : (maxv[2] - minv[2]);

	real_t scale = extent == 0 ? 0.0 : 1.0 / extent;

	// generate Morton order based on the position inside a unit cube
	for (size_t i = 0; i < vertex_count; ++i)
	{
		const real_t* v = vertex_positions_data + i * vertex_stride_real;

		uint32_t x = uint32_t((v[0] - minv[0]) * scale * 1023.0 + 0.5);
		uint32_t y = uint32_t((v[1] - minv[1]) * scale * 1023.0 + 0.5);
		uint32_t z = uint32_t((v[2] - minv[2]) * scale * 1023.0 + 0.5);

		result[i] = part1By2(x) | (part1By2(y) << 1) | (part1By2(z) << 2);
	}
}

static void computeHistogram(datatype_t (&hist)[1024][3], const datatype_t* data, size_t count)
{
	memset(hist, 0, sizeof(hist));

	// compute 3 10-bit histograms in parallel
	for (size_t i = 0; i < count; ++i)
	{
		datatype_t id = data[i];

		hist[(id >> 0) & 1023][0]++;
		hist[(id >> 10) & 1023][1]++;
		hist[(id >> 20) & 1023][2]++;
	}

	datatype_t sumx = 0, sumy = 0, sumz = 0;

	// replace histogram data with prefix histogram sums in-place
	for (int i = 0; i < 1024; ++i)
	{
		datatype_t hx = hist[i][0], hy = hist[i][1], hz = hist[i][2];

		hist[i][0] = sumx;
		hist[i][1] = sumy;
		hist[i][2] = sumz;

		sumx += hx;
		sumy += hy;
		sumz += hz;
	}

	assert(sumx == count && sumy == count && sumz == count);
}

static void radixPass(datatype_t* destination, const datatype_t* source, const datatype_t* keys, size_t count, datatype_t (&hist)[1024][3], int pass)
{
	int bitoff = pass * 10;

	for (size_t i = 0; i < count; ++i)
	{
		datatype_t id = (keys[source[i]] >> bitoff) & 1023;

		destination[hist[id][pass]++] = source[i];
	}
}

} // namespace meshopt

void meshopt_spatialSortRemap(datatype_t* destination, const real_t* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(real_t) == 0);

	meshopt_Allocator allocator;

	datatype_t* keys = allocator.allocate<datatype_t>(vertex_count);
	computeOrder(keys, vertex_positions, vertex_count, vertex_positions_stride);

	datatype_t hist[1024][3];
	computeHistogram(hist, keys, vertex_count);

	datatype_t* scratch = allocator.allocate<datatype_t>(vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
		destination[i] = datatype_t(i);

	// 3-pass radix sort computes the resulting order into scratch
	radixPass(scratch, destination, keys, vertex_count, hist, 0);
	radixPass(destination, scratch, keys, vertex_count, hist, 1);
	radixPass(scratch, destination, keys, vertex_count, hist, 2);

	// since our remap table is mapping old=>new, we need to reverse it
	for (size_t i = 0; i < vertex_count; ++i)
		destination[scratch[i]] = datatype_t(i);
}

void meshopt_spatialSortTriangles(datatype_t* destination, const datatype_t* indices, size_t index_count, const real_t* vertex_positions, size_t vertex_count, size_t vertex_positions_stride)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(real_t) == 0);

	(void)vertex_count;

	size_t face_count = index_count / 3;
	size_t vertex_stride_real = vertex_positions_stride / sizeof(real_t);

	meshopt_Allocator allocator;

	real_t* centroids = allocator.allocate<real_t>(face_count * 3);

	for (size_t i = 0; i < face_count; ++i)
	{
		datatype_t a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		assert(a < vertex_count && b < vertex_count && c < vertex_count);

		const real_t* va = vertex_positions + a * vertex_stride_real;
		const real_t* vb = vertex_positions + b * vertex_stride_real;
		const real_t* vc = vertex_positions + c * vertex_stride_real;

		centroids[i * 3 + 0] = (va[0] + vb[0] + vc[0]) / 3.0;
		centroids[i * 3 + 1] = (va[1] + vb[1] + vc[1]) / 3.0;
		centroids[i * 3 + 2] = (va[2] + vb[2] + vc[2]) / 3.0;
	}

	datatype_t* remap = allocator.allocate<datatype_t>(face_count);

	meshopt_spatialSortRemap(remap, centroids, face_count, sizeof(real_t) * 3);

	// support in-order remap
	if (destination == indices)
	{
		datatype_t* indices_copy = allocator.allocate<datatype_t>(index_count);
		memcpy(indices_copy, indices, index_count * sizeof(datatype_t));
		indices = indices_copy;
	}

	for (size_t i = 0; i < face_count; ++i)
	{
		datatype_t a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
		datatype_t r = remap[i];

		destination[r * 3 + 0] = a;
		destination[r * 3 + 1] = b;
		destination[r * 3 + 2] = c;
	}
}
