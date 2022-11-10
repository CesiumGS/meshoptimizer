// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <cmath>
#include <string.h>

// This work is based on:
// Pedro Sander, Diego Nehab and Joshua Barczak. Fast Triangle Reordering for Vertex Locality and Reduced Overdraw. 2007
namespace meshopt
{

static void calculateSortData(real_t* sort_data, const datatype_t* indices, size_t index_count, const real_t* vertex_positions, size_t vertex_positions_stride, const datatype_t* clusters, size_t cluster_count)
{
	size_t vertex_stride_real_t = vertex_positions_stride / sizeof(real_t);

	real_t mesh_centroid[3] = {};

	for (size_t i = 0; i < index_count; ++i)
	{
		const real_t* p = vertex_positions + vertex_stride_real_t * indices[i];

		mesh_centroid[0] += p[0];
		mesh_centroid[1] += p[1];
		mesh_centroid[2] += p[2];
	}

	mesh_centroid[0] /= index_count;
	mesh_centroid[1] /= index_count;
	mesh_centroid[2] /= index_count;

	for (size_t cluster = 0; cluster < cluster_count; ++cluster)
	{
		size_t cluster_begin = clusters[cluster] * 3;
		size_t cluster_end = (cluster + 1 < cluster_count) ? clusters[cluster + 1] * 3 : index_count;
		assert(cluster_begin < cluster_end);

		real_t cluster_area = 0;
		real_t cluster_centroid[3] = {};
		real_t cluster_normal[3] = {};

		for (size_t i = cluster_begin; i < cluster_end; i += 3)
		{
			const real_t* p0 = vertex_positions + vertex_stride_real_t * indices[i + 0];
			const real_t* p1 = vertex_positions + vertex_stride_real_t * indices[i + 1];
			const real_t* p2 = vertex_positions + vertex_stride_real_t * indices[i + 2];

			real_t p10[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
			real_t p20[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};

			real_t normalx = p10[1] * p20[2] - p10[2] * p20[1];
			real_t normaly = p10[2] * p20[0] - p10[0] * p20[2];
			real_t normalz = p10[0] * p20[1] - p10[1] * p20[0];

			real_t area = std::sqrt(normalx * normalx + normaly * normaly + normalz * normalz);

			cluster_centroid[0] += (p0[0] + p1[0] + p2[0]) * (area / 3);
			cluster_centroid[1] += (p0[1] + p1[1] + p2[1]) * (area / 3);
			cluster_centroid[2] += (p0[2] + p1[2] + p2[2]) * (area / 3);
			cluster_normal[0] += normalx;
			cluster_normal[1] += normaly;
			cluster_normal[2] += normalz;
			cluster_area += area;
		}

		real_t inv_cluster_area = cluster_area == 0 ? 0 : 1 / cluster_area;

		cluster_centroid[0] *= inv_cluster_area;
		cluster_centroid[1] *= inv_cluster_area;
		cluster_centroid[2] *= inv_cluster_area;

		real_t cluster_normal_length = std::sqrt(cluster_normal[0] * cluster_normal[0] + cluster_normal[1] * cluster_normal[1] + cluster_normal[2] * cluster_normal[2]);
		real_t inv_cluster_normal_length = cluster_normal_length == 0 ? 0 : 1 / cluster_normal_length;

		cluster_normal[0] *= inv_cluster_normal_length;
		cluster_normal[1] *= inv_cluster_normal_length;
		cluster_normal[2] *= inv_cluster_normal_length;

		real_t centroid_vector[3] = {cluster_centroid[0] - mesh_centroid[0], cluster_centroid[1] - mesh_centroid[1], cluster_centroid[2] - mesh_centroid[2]};

		sort_data[cluster] = centroid_vector[0] * cluster_normal[0] + centroid_vector[1] * cluster_normal[1] + centroid_vector[2] * cluster_normal[2];
	}
}

static void calculateSortOrderRadix(datatype_t* sort_order, const real_t* sort_data, unsigned short* sort_keys, size_t cluster_count)
{
	// compute sort data bounds and renormalize, using fixed point snorm
	real_t sort_data_max = 1e-3f;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		real_t dpa = std::abs(sort_data[i]);

		sort_data_max = (sort_data_max < dpa) ? dpa : sort_data_max;
	}

	const int sort_bits = 11;

	for (size_t i = 0; i < cluster_count; ++i)
	{
		// note that we flip distribution since high dot product should come first
		real_t sort_key = 0.5f - 0.5f * (sort_data[i] / sort_data_max);

		sort_keys[i] = meshopt_quantizeUnorm(sort_key, sort_bits) & ((1 << sort_bits) - 1);
	}

	// fill histogram for counting sort
	datatype_t histogram[1 << sort_bits];
	memset(histogram, 0, sizeof(histogram));

	for (size_t i = 0; i < cluster_count; ++i)
	{
		histogram[sort_keys[i]]++;
	}

	// compute offsets based on histogram data
	size_t histogram_sum = 0;

	for (size_t i = 0; i < 1 << sort_bits; ++i)
	{
		size_t count = histogram[i];
		histogram[i] = datatype_t(histogram_sum);
		histogram_sum += count;
	}

	assert(histogram_sum == cluster_count);

	// compute sort order based on offsets
	for (size_t i = 0; i < cluster_count; ++i)
	{
		sort_order[histogram[sort_keys[i]]++] = datatype_t(i);
	}
}

static datatype_t updateCache(datatype_t a, datatype_t b, datatype_t c, datatype_t cache_size, datatype_t* cache_timestamps, datatype_t& timestamp)
{
	datatype_t cache_misses = 0;

	// if vertex is not in cache, put it in cache
	if (timestamp - cache_timestamps[a] > cache_size)
	{
		cache_timestamps[a] = timestamp++;
		cache_misses++;
	}

	if (timestamp - cache_timestamps[b] > cache_size)
	{
		cache_timestamps[b] = timestamp++;
		cache_misses++;
	}

	if (timestamp - cache_timestamps[c] > cache_size)
	{
		cache_timestamps[c] = timestamp++;
		cache_misses++;
	}

	return cache_misses;
}

static size_t generateHardBoundaries(datatype_t* destination, const datatype_t* indices, size_t index_count, size_t vertex_count, datatype_t cache_size, datatype_t* cache_timestamps)
{
	memset(cache_timestamps, 0, vertex_count * sizeof(datatype_t));

	datatype_t timestamp = cache_size + 1;

	size_t face_count = index_count / 3;

	size_t result = 0;

	for (size_t i = 0; i < face_count; ++i)
	{
		datatype_t m = updateCache(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], cache_size, &cache_timestamps[0], timestamp);

		// when all three vertices are not in the cache it's usually relatively safe to assume that this is a new patch in the mesh
		// that is disjoint from previous vertices; sometimes it might come back to reference existing vertices but that frequently
		// suggests an inefficiency in the vertex cache optimization algorithm
		// usually the first triangle has 3 misses unless it's degenerate - thus we make sure the first cluster always starts with 0
		if (i == 0 || m == 3)
		{
			destination[result++] = datatype_t(i);
		}
	}

	assert(result <= index_count / 3);

	return result;
}

static size_t generateSoftBoundaries(datatype_t* destination, const datatype_t* indices, size_t index_count, size_t vertex_count, const datatype_t* clusters, size_t cluster_count, datatype_t cache_size, real_t threshold, datatype_t* cache_timestamps)
{
	memset(cache_timestamps, 0, vertex_count * sizeof(datatype_t));

	datatype_t timestamp = 0;

	size_t result = 0;

	for (size_t it = 0; it < cluster_count; ++it)
	{
		size_t start = clusters[it];
		size_t end = (it + 1 < cluster_count) ? clusters[it + 1] : index_count / 3;
		assert(start < end);

		// reset cache
		timestamp += cache_size + 1;

		// measure cluster ACMR
		datatype_t cluster_misses = 0;

		for (size_t i = start; i < end; ++i)
		{
			datatype_t m = updateCache(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], cache_size, &cache_timestamps[0], timestamp);

			cluster_misses += m;
		}

		real_t cluster_threshold = threshold * (real_t(cluster_misses) / real_t(end - start));

		// first cluster always starts from the hard cluster boundary
		destination[result++] = datatype_t(start);

		// reset cache
		timestamp += cache_size + 1;

		datatype_t running_misses = 0;
		datatype_t running_faces = 0;

		for (size_t i = start; i < end; ++i)
		{
			datatype_t m = updateCache(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], cache_size, &cache_timestamps[0], timestamp);

			running_misses += m;
			running_faces += 1;

			if (real_t(running_misses) / real_t(running_faces) <= cluster_threshold)
			{
				// we have reached the target ACMR with the current triangle so we need to start a new cluster on the next one
				// note that this may mean that we add 'end` to destination for the last triangle, which will imply that the last
				// cluster is empty; however, the 'pop_back' after the loop will clean it up
				destination[result++] = datatype_t(i + 1);

				// reset cache
				timestamp += cache_size + 1;

				running_misses = 0;
				running_faces = 0;
			}
		}

		// each time we reach the target ACMR we flush the cluster
		// this means that the last cluster is by definition not very good - there are frequent cases where we are left with a few triangles
		// in the last cluster, producing a very bad ACMR and significantly penalizing the overall results
		// thus we remove the last cluster boundary, merging the last complete cluster with the last incomplete one
		// there are sometimes cases when the last cluster is actually good enough - in which case the code above would have added 'end'
		// to the cluster boundary array which we need to remove anyway - this code will do that automatically
		if (destination[result - 1] != start)
		{
			result--;
		}
	}

	assert(result >= cluster_count);
	assert(result <= index_count / 3);

	return result;
}

} // namespace meshopt

void meshopt_optimizeOverdraw(datatype_t* destination, const datatype_t* indices, size_t index_count, const real_t* vertex_positions, size_t vertex_count, size_t vertex_positions_stride, real_t threshold)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(vertex_positions_stride >= 12 && vertex_positions_stride <= 256);
	assert(vertex_positions_stride % sizeof(real_t) == 0);

	meshopt_Allocator allocator;

	// guard for empty meshes
	if (index_count == 0 || vertex_count == 0)
		return;

	// support in-place optimization
	if (destination == indices)
	{
		datatype_t* indices_copy = allocator.allocate<datatype_t>(index_count);
		memcpy(indices_copy, indices, index_count * sizeof(datatype_t));
		indices = indices_copy;
	}

	datatype_t cache_size = 16;

	datatype_t* cache_timestamps = allocator.allocate<datatype_t>(vertex_count);

	// generate hard boundaries from full-triangle cache misses
	datatype_t* hard_clusters = allocator.allocate<datatype_t>(index_count / 3);
	size_t hard_cluster_count = generateHardBoundaries(hard_clusters, indices, index_count, vertex_count, cache_size, cache_timestamps);

	// generate soft boundaries
	datatype_t* soft_clusters = allocator.allocate<datatype_t>(index_count / 3 + 1);
	size_t soft_cluster_count = generateSoftBoundaries(soft_clusters, indices, index_count, vertex_count, hard_clusters, hard_cluster_count, cache_size, threshold, cache_timestamps);

	const datatype_t* clusters = soft_clusters;
	size_t cluster_count = soft_cluster_count;

	// fill sort data
	real_t* sort_data = allocator.allocate<real_t>(cluster_count);
	calculateSortData(sort_data, indices, index_count, vertex_positions, vertex_positions_stride, clusters, cluster_count);

	// sort clusters using sort data
	unsigned short* sort_keys = allocator.allocate<unsigned short>(cluster_count);
	datatype_t* sort_order = allocator.allocate<datatype_t>(cluster_count);
	calculateSortOrderRadix(sort_order, sort_data, sort_keys, cluster_count);

	// fill output buffer
	size_t offset = 0;

	for (size_t it = 0; it < cluster_count; ++it)
	{
		datatype_t cluster = sort_order[it];
		assert(cluster < cluster_count);

		size_t cluster_begin = clusters[cluster] * 3;
		size_t cluster_end = (cluster + 1 < cluster_count) ? clusters[cluster + 1] * 3 : index_count;
		assert(cluster_begin < cluster_end);

		memcpy(destination + offset, indices + cluster_begin, (cluster_end - cluster_begin) * sizeof(datatype_t));
		offset += cluster_end - cluster_begin;
	}

	assert(offset == index_count);
}
