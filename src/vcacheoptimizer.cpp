// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <string.h>

// This work is based on:
// Tom Forsyth. Linear-Speed Vertex Cache Optimisation. 2006
// Pedro Sander, Diego Nehab and Joshua Barczak. Fast Triangle Reordering for Vertex Locality and Reduced Overdraw. 2007
namespace meshopt
{

const size_t kCacheSizeMax = 16;
const size_t kValenceMax = 8;

struct VertexScoreTable
{
	real_t cache[1 + kCacheSizeMax];
	real_t live[1 + kValenceMax];
};

// Tuned to minimize the ACMR of a GPU that has a cache profile similar to NVidia and AMD
static const VertexScoreTable kVertexScoreTable = {
    {0.0, 0.779, 0.791, 0.789, 0.981, 0.843, 0.726, 0.847, 0.882, 0.867, 0.799, 0.642, 0.613, 0.600, 0.568, 0.372, 0.234},
    {0.0, 0.995, 0.713, 0.450, 0.404, 0.059, 0.005, 0.147, 0.006},
};

// Tuned to minimize the encoded index buffer size
static const VertexScoreTable kVertexScoreTableStrip = {
    {0.0, 1.000, 1.000, 1.000, 0.453, 0.561, 0.490, 0.459, 0.179, 0.526, 0.000, 0.227, 0.184, 0.490, 0.112, 0.050, 0.131},
    {0.0, 0.956, 0.786, 0.577, 0.558, 0.618, 0.549, 0.499, 0.489},
};

struct TriangleAdjacency
{
	datatype_t* counts;
	datatype_t* offsets;
	datatype_t* data;
};

static void buildTriangleAdjacency(TriangleAdjacency& adjacency, const datatype_t* indices, size_t index_count, size_t vertex_count, meshopt_Allocator& allocator)
{
	size_t face_count = index_count / 3;

	// allocate arrays
	adjacency.counts = allocator.allocate<datatype_t>(vertex_count);
	adjacency.offsets = allocator.allocate<datatype_t>(vertex_count);
	adjacency.data = allocator.allocate<datatype_t>(index_count);

	// fill triangle counts
	memset(adjacency.counts, 0, vertex_count * sizeof(datatype_t));

	for (size_t i = 0; i < index_count; ++i)
	{
		assert(indices[i] < vertex_count);

		adjacency.counts[indices[i]]++;
	}

	// fill offset table
	datatype_t offset = 0;

	for (size_t i = 0; i < vertex_count; ++i)
	{
		adjacency.offsets[i] = offset;
		offset += adjacency.counts[i];
	}

	assert(offset == index_count);

	// fill triangle data
	for (size_t i = 0; i < face_count; ++i)
	{
		datatype_t a = indices[i * 3 + 0], b = indices[i * 3 + 1], c = indices[i * 3 + 2];

		adjacency.data[adjacency.offsets[a]++] = datatype_t(i);
		adjacency.data[adjacency.offsets[b]++] = datatype_t(i);
		adjacency.data[adjacency.offsets[c]++] = datatype_t(i);
	}

	// fix offsets that have been disturbed by the previous pass
	for (size_t i = 0; i < vertex_count; ++i)
	{
		assert(adjacency.offsets[i] >= adjacency.counts[i]);

		adjacency.offsets[i] -= adjacency.counts[i];
	}
}

static datatype_t getNextVertexDeadEnd(const datatype_t* dead_end, datatype_t& dead_end_top, datatype_t& input_cursor, const datatype_t* live_triangles, size_t vertex_count)
{
	// check dead-end stack
	while (dead_end_top)
	{
		datatype_t vertex = dead_end[--dead_end_top];

		if (live_triangles[vertex] > 0)
			return vertex;
	}

	// input order
	while (input_cursor < vertex_count)
	{
		if (live_triangles[input_cursor] > 0)
			return input_cursor;

		++input_cursor;
	}

	return ALL_BITS_ONE;
}

static datatype_t getNextVertexNeighbour(const datatype_t* next_candidates_begin, const datatype_t* next_candidates_end, const datatype_t* live_triangles, const datatype_t* cache_timestamps, datatype_t timestamp, datatype_t cache_size)
{
	datatype_t best_candidate = ALL_BITS_ONE;
	int64_t best_priority = -1;

	for (const datatype_t* next_candidate = next_candidates_begin; next_candidate != next_candidates_end; ++next_candidate)
	{
		datatype_t vertex = *next_candidate;

		// otherwise we don't need to process it
		if (live_triangles[vertex] > 0)
		{
			int64_t priority = 0;

			// will it be in cache after fanning?
			if (2 * live_triangles[vertex] + timestamp - cache_timestamps[vertex] <= cache_size)
			{
				priority = timestamp - cache_timestamps[vertex]; // position in cache
			}

			if (priority > best_priority)
			{
				best_candidate = vertex;
				best_priority = priority;
			}
		}
	}

	return best_candidate;
}

static real_t vertexScore(const VertexScoreTable* table, int cache_position, datatype_t live_triangles)
{
	assert(cache_position >= -1 && cache_position < int(kCacheSizeMax));

	datatype_t live_triangles_clamped = live_triangles < kValenceMax ? live_triangles : kValenceMax;

	return table->cache[1 + cache_position] + table->live[live_triangles_clamped];
}

static datatype_t getNextTriangleDeadEnd(datatype_t& input_cursor, const unsigned char* emitted_flags, size_t face_count)
{
	// input order
	while (input_cursor < face_count)
	{
		if (!emitted_flags[input_cursor])
			return input_cursor;

		++input_cursor;
	}

	return ALL_BITS_ONE;
}

} // namespace meshopt

void meshopt_optimizeVertexCacheTable(datatype_t* destination, const datatype_t* indices, size_t index_count, size_t vertex_count, const meshopt::VertexScoreTable* table)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);

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
	assert(cache_size <= kCacheSizeMax);

	size_t face_count = index_count / 3;

	// build adjacency information
	TriangleAdjacency adjacency = {};
	buildTriangleAdjacency(adjacency, indices, index_count, vertex_count, allocator);

	// live triangle counts
	datatype_t* live_triangles = allocator.allocate<datatype_t>(vertex_count);
	memcpy(live_triangles, adjacency.counts, vertex_count * sizeof(datatype_t));

	// emitted flags
	unsigned char* emitted_flags = allocator.allocate<unsigned char>(face_count);
	memset(emitted_flags, 0, face_count);

	// compute initial vertex scores
	real_t* vertex_scores = allocator.allocate<real_t>(vertex_count);

	for (size_t i = 0; i < vertex_count; ++i)
		vertex_scores[i] = vertexScore(table, -1, live_triangles[i]);

	// compute triangle scores
	real_t* triangle_scores = allocator.allocate<real_t>(face_count);

	for (size_t i = 0; i < face_count; ++i)
	{
		datatype_t a = indices[i * 3 + 0];
		datatype_t b = indices[i * 3 + 1];
		datatype_t c = indices[i * 3 + 2];

		triangle_scores[i] = vertex_scores[a] + vertex_scores[b] + vertex_scores[c];
	}

	datatype_t cache_holder[2 * (kCacheSizeMax + 3)];
	datatype_t* cache = cache_holder;
	datatype_t* cache_new = cache_holder + kCacheSizeMax + 3;
	size_t cache_count = 0;

	datatype_t current_triangle = 0;
	datatype_t input_cursor = 1;

	datatype_t output_triangle = 0;

	while (current_triangle != ALL_BITS_ONE)
	{
		assert(output_triangle < face_count);

		datatype_t a = indices[current_triangle * 3 + 0];
		datatype_t b = indices[current_triangle * 3 + 1];
		datatype_t c = indices[current_triangle * 3 + 2];

		// output indices
		destination[output_triangle * 3 + 0] = a;
		destination[output_triangle * 3 + 1] = b;
		destination[output_triangle * 3 + 2] = c;
		output_triangle++;

		// update emitted flags
		emitted_flags[current_triangle] = true;
		triangle_scores[current_triangle] = 0;

		// new triangle
		size_t cache_write = 0;
		cache_new[cache_write++] = a;
		cache_new[cache_write++] = b;
		cache_new[cache_write++] = c;

		// old triangles
		for (size_t i = 0; i < cache_count; ++i)
		{
			datatype_t index = cache[i];

			if (index != a && index != b && index != c)
			{
				cache_new[cache_write++] = index;
			}
		}

		datatype_t* cache_temp = cache;
		cache = cache_new, cache_new = cache_temp;
		cache_count = cache_write > cache_size ? cache_size : cache_write;

		// update live triangle counts
		live_triangles[a]--;
		live_triangles[b]--;
		live_triangles[c]--;

		// remove emitted triangle from adjacency data
		// this makes sure that we spend less time traversing these lists on subsequent iterations
		for (size_t k = 0; k < 3; ++k)
		{
			datatype_t index = indices[current_triangle * 3 + k];

			datatype_t* neighbours = &adjacency.data[0] + adjacency.offsets[index];
			size_t neighbours_size = adjacency.counts[index];

			for (size_t i = 0; i < neighbours_size; ++i)
			{
				datatype_t tri = neighbours[i];

				if (tri == current_triangle)
				{
					neighbours[i] = neighbours[neighbours_size - 1];
					adjacency.counts[index]--;
					break;
				}
			}
		}

		datatype_t best_triangle = ALL_BITS_ONE;
		real_t best_score = 0;

		// update cache positions, vertex scores and triangle scores, and find next best triangle
		for (size_t i = 0; i < cache_write; ++i)
		{
			datatype_t index = cache[i];

			int cache_position = i >= cache_size ? -1 : int(i);

			// update vertex score
			real_t score = vertexScore(table, cache_position, live_triangles[index]);
			real_t score_diff = score - vertex_scores[index];

			vertex_scores[index] = score;

			// update scores of vertex triangles
			const datatype_t* neighbours_begin = &adjacency.data[0] + adjacency.offsets[index];
			const datatype_t* neighbours_end = neighbours_begin + adjacency.counts[index];

			for (const datatype_t* it = neighbours_begin; it != neighbours_end; ++it)
			{
				datatype_t tri = *it;
				assert(!emitted_flags[tri]);

				real_t tri_score = triangle_scores[tri] + score_diff;
				assert(tri_score > 0);

				if (best_score < tri_score)
				{
					best_triangle = tri;
					best_score = tri_score;
				}

				triangle_scores[tri] = tri_score;
			}
		}

		// step through input triangles in order if we hit a dead-end
		current_triangle = best_triangle;

		if (current_triangle == ALL_BITS_ONE)
		{
			current_triangle = getNextTriangleDeadEnd(input_cursor, &emitted_flags[0], face_count);
		}
	}

	assert(input_cursor == face_count);
	assert(output_triangle == face_count);
}

void meshopt_optimizeVertexCache(datatype_t* destination, const datatype_t* indices, size_t index_count, size_t vertex_count)
{
	meshopt_optimizeVertexCacheTable(destination, indices, index_count, vertex_count, &meshopt::kVertexScoreTable);
}

void meshopt_optimizeVertexCacheStrip(datatype_t* destination, const datatype_t* indices, size_t index_count, size_t vertex_count)
{
	meshopt_optimizeVertexCacheTable(destination, indices, index_count, vertex_count, &meshopt::kVertexScoreTableStrip);
}

void meshopt_optimizeVertexCacheFifo(datatype_t* destination, const datatype_t* indices, size_t index_count, size_t vertex_count, datatype_t cache_size)
{
	using namespace meshopt;

	assert(index_count % 3 == 0);
	assert(cache_size >= 3);

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

	size_t face_count = index_count / 3;

	// build adjacency information
	TriangleAdjacency adjacency = {};
	buildTriangleAdjacency(adjacency, indices, index_count, vertex_count, allocator);

	// live triangle counts
	datatype_t* live_triangles = allocator.allocate<datatype_t>(vertex_count);
	memcpy(live_triangles, adjacency.counts, vertex_count * sizeof(datatype_t));

	// cache time stamps
	datatype_t* cache_timestamps = allocator.allocate<datatype_t>(vertex_count);
	memset(cache_timestamps, 0, vertex_count * sizeof(datatype_t));

	// dead-end stack
	datatype_t* dead_end = allocator.allocate<datatype_t>(index_count);
	datatype_t dead_end_top = 0;

	// emitted flags
	unsigned char* emitted_flags = allocator.allocate<unsigned char>(face_count);
	memset(emitted_flags, 0, face_count);

	datatype_t current_vertex = 0;

	datatype_t timestamp = cache_size + 1;
	datatype_t input_cursor = 1; // vertex to restart from in case of dead-end

	datatype_t output_triangle = 0;

	while (current_vertex != ALL_BITS_ONE)
	{
		const datatype_t* next_candidates_begin = &dead_end[0] + dead_end_top;

		// emit all vertex neighbours
		const datatype_t* neighbours_begin = &adjacency.data[0] + adjacency.offsets[current_vertex];
		const datatype_t* neighbours_end = neighbours_begin + adjacency.counts[current_vertex];

		for (const datatype_t* it = neighbours_begin; it != neighbours_end; ++it)
		{
			datatype_t triangle = *it;

			if (!emitted_flags[triangle])
			{
				datatype_t a = indices[triangle * 3 + 0], b = indices[triangle * 3 + 1], c = indices[triangle * 3 + 2];

				// output indices
				destination[output_triangle * 3 + 0] = a;
				destination[output_triangle * 3 + 1] = b;
				destination[output_triangle * 3 + 2] = c;
				output_triangle++;

				// update dead-end stack
				dead_end[dead_end_top + 0] = a;
				dead_end[dead_end_top + 1] = b;
				dead_end[dead_end_top + 2] = c;
				dead_end_top += 3;

				// update live triangle counts
				live_triangles[a]--;
				live_triangles[b]--;
				live_triangles[c]--;

				// update cache info
				// if vertex is not in cache, put it in cache
				if (timestamp - cache_timestamps[a] > cache_size)
					cache_timestamps[a] = timestamp++;

				if (timestamp - cache_timestamps[b] > cache_size)
					cache_timestamps[b] = timestamp++;

				if (timestamp - cache_timestamps[c] > cache_size)
					cache_timestamps[c] = timestamp++;

				// update emitted flags
				emitted_flags[triangle] = true;
			}
		}

		// next candidates are the ones we pushed to dead-end stack just now
		const datatype_t* next_candidates_end = &dead_end[0] + dead_end_top;

		// get next vertex
		current_vertex = getNextVertexNeighbour(next_candidates_begin, next_candidates_end, &live_triangles[0], &cache_timestamps[0], timestamp, cache_size);

		if (current_vertex == ALL_BITS_ONE)
		{
			current_vertex = getNextVertexDeadEnd(&dead_end[0], dead_end_top, input_cursor, &live_triangles[0], vertex_count);
		}
	}

	assert(output_triangle == face_count);
}
