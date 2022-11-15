// This file is part of gltfpack; see gltfpack.h for version/license details
#include "gltfpack.h"

#include <algorithm>

#include <float.h>
#include <math.h>
#include <string.h>

static real_t getDelta(const Attr& l, const Attr& r, cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return std::max(std::max(std::abs(l.f[0] - r.f[0]), std::abs(l.f[1] - r.f[1])), std::abs(l.f[2] - r.f[2]));

	case cgltf_animation_path_type_rotation:
		return acosf(std::min(1.0, std::abs(l.f[0] * r.f[0] + l.f[1] * r.f[1] + l.f[2] * r.f[2] + l.f[3] * r.f[3])));

	case cgltf_animation_path_type_scale:
		return std::max(std::max(std::abs(l.f[0] / r.f[0] - 1), std::abs(l.f[1] / r.f[1] - 1)), std::abs(l.f[2] / r.f[2] - 1));

	case cgltf_animation_path_type_weights:
		return std::abs(l.f[0] - r.f[0]);

	default:
		assert(!"Uknown animation path");
		return 0;
	}
}

static real_t getDeltaTolerance(cgltf_animation_path_type type)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		return 0.0001f; // 0.1mm linear

	case cgltf_animation_path_type_rotation:
		return 0.1f * (3.1415926f / 180.f); // 0.1 degrees

	case cgltf_animation_path_type_scale:
		return 0.001f; // 0.1% ratio

	case cgltf_animation_path_type_weights:
		return 0.001f; // 0.1% linear

	default:
		assert(!"Uknown animation path");
		return 0;
	}
}

static Attr interpolateLinear(const Attr& l, const Attr& r, real_t t, cgltf_animation_path_type type)
{
	if (type == cgltf_animation_path_type_rotation)
	{
		// Approximating slerp, https://zeux.io/2015/07/23/approximating-slerp/
		// We also handle quaternion double-cover
		real_t ca = l.f[0] * r.f[0] + l.f[1] * r.f[1] + l.f[2] * r.f[2] + l.f[3] * r.f[3];

		real_t d = std::abs(ca);
		real_t A = 1.0904f + d * (-3.2452f + d * (3.55645f - d * 1.43519f));
		real_t B = 0.848013f + d * (-1.06021f + d * 0.215638f);
		real_t k = A * (t - 0.5) * (t - 0.5) + B;
		real_t ot = t + t * (t - 0.5) * (t - 1) * k;

		real_t t0 = 1 - ot;
		real_t t1 = ca > 0 ? ot : -ot;

		Attr lerp = {{
		    l.f[0] * t0 + r.f[0] * t1,
		    l.f[1] * t0 + r.f[1] * t1,
		    l.f[2] * t0 + r.f[2] * t1,
		    l.f[3] * t0 + r.f[3] * t1,
		}};

		real_t len = std::sqrt(lerp.f[0] * lerp.f[0] + lerp.f[1] * lerp.f[1] + lerp.f[2] * lerp.f[2] + lerp.f[3] * lerp.f[3]);

		if (len > 0.0)
		{
			lerp.f[0] /= len;
			lerp.f[1] /= len;
			lerp.f[2] /= len;
			lerp.f[3] /= len;
		}

		return lerp;
	}
	else
	{
		Attr lerp = {{
		    l.f[0] * (1 - t) + r.f[0] * t,
		    l.f[1] * (1 - t) + r.f[1] * t,
		    l.f[2] * (1 - t) + r.f[2] * t,
		    l.f[3] * (1 - t) + r.f[3] * t,
		}};

		return lerp;
	}
}

static Attr interpolateHermite(const Attr& v0, const Attr& t0, const Attr& v1, const Attr& t1, real_t t, real_t dt, cgltf_animation_path_type type)
{
	real_t s0 = 1 + t * t * (2 * t - 3);
	real_t s1 = t + t * t * (t - 2);
	real_t s2 = 1 - s0;
	real_t s3 = t * t * (t - 1);

	real_t ts1 = dt * s1;
	real_t ts3 = dt * s3;

	Attr lerp = {{
	    s0 * v0.f[0] + ts1 * t0.f[0] + s2 * v1.f[0] + ts3 * t1.f[0],
	    s0 * v0.f[1] + ts1 * t0.f[1] + s2 * v1.f[1] + ts3 * t1.f[1],
	    s0 * v0.f[2] + ts1 * t0.f[2] + s2 * v1.f[2] + ts3 * t1.f[2],
	    s0 * v0.f[3] + ts1 * t0.f[3] + s2 * v1.f[3] + ts3 * t1.f[3],
	}};

	if (type == cgltf_animation_path_type_rotation)
	{
		real_t len = std::sqrt(lerp.f[0] * lerp.f[0] + lerp.f[1] * lerp.f[1] + lerp.f[2] * lerp.f[2] + lerp.f[3] * lerp.f[3]);

		if (len > 0.0)
		{
			lerp.f[0] /= len;
			lerp.f[1] /= len;
			lerp.f[2] /= len;
			lerp.f[3] /= len;
		}
	}

	return lerp;
}

static void resampleKeyframes(std::vector<Attr>& data, const std::vector<real_t>& input, const std::vector<Attr>& output, cgltf_animation_path_type type, cgltf_interpolation_type interpolation, size_t components, int frames, real_t mint, int freq)
{
	size_t cursor = 0;

	for (int i = 0; i < frames; ++i)
	{
		real_t time = mint + real_t(i) / freq;

		while (cursor + 1 < input.size())
		{
			real_t next_time = input[cursor + 1];

			if (next_time > time)
				break;

			cursor++;
		}

		if (cursor + 1 < input.size())
		{
			real_t cursor_time = input[cursor + 0];
			real_t next_time = input[cursor + 1];

			real_t range = next_time - cursor_time;
			real_t inv_range = (range == 0.0) ? 0.0 : 1.0 / (next_time - cursor_time);
			real_t t = std::max(0.0, std::min(1.0, (time - cursor_time) * inv_range));

			for (size_t j = 0; j < components; ++j)
			{
				switch (interpolation)
				{
				case cgltf_interpolation_type_linear:
				{
					const Attr& v0 = output[(cursor + 0) * components + j];
					const Attr& v1 = output[(cursor + 1) * components + j];
					data.push_back(interpolateLinear(v0, v1, t, type));
				}
				break;

				case cgltf_interpolation_type_step:
				{
					const Attr& v = output[cursor * components + j];
					data.push_back(v);
				}
				break;

				case cgltf_interpolation_type_cubic_spline:
				{
					const Attr& v0 = output[(cursor * 3 + 1) * components + j];
					const Attr& b0 = output[(cursor * 3 + 2) * components + j];
					const Attr& a1 = output[(cursor * 3 + 3) * components + j];
					const Attr& v1 = output[(cursor * 3 + 4) * components + j];
					data.push_back(interpolateHermite(v0, b0, v1, a1, t, range, type));
				}
				break;

				default:
					assert(!"Unknown interpolation type");
				}
			}
		}
		else
		{
			size_t offset = (interpolation == cgltf_interpolation_type_cubic_spline) ? cursor * 3 + 1 : cursor;

			for (size_t j = 0; j < components; ++j)
			{
				const Attr& v = output[offset * components + j];
				data.push_back(v);
			}
		}
	}
}

static real_t getMaxDelta(const std::vector<Attr>& data, cgltf_animation_path_type type, int frames, const Attr* value, size_t components)
{
	assert(data.size() == frames * components);

	real_t result = 0;

	for (int i = 0; i < frames; ++i)
	{
		for (size_t j = 0; j < components; ++j)
		{
			real_t delta = getDelta(value[j], data[i * components + j], type);

			result = (result < delta) ? delta : result;
		}
	}

	return result;
}

static void getBaseTransform(Attr* result, size_t components, cgltf_animation_path_type type, cgltf_node* node)
{
	switch (type)
	{
	case cgltf_animation_path_type_translation:
		memcpy(result->f, node->translation, 3 * sizeof(real_t));
		break;

	case cgltf_animation_path_type_rotation:
		memcpy(result->f, node->rotation, 4 * sizeof(real_t));
		break;

	case cgltf_animation_path_type_scale:
		memcpy(result->f, node->scale, 3 * sizeof(real_t));
		break;

	case cgltf_animation_path_type_weights:
		if (node->weights_count)
		{
			assert(node->weights_count == components);
			memcpy(result->f, node->weights, components * sizeof(real_t));
		}
		else if (node->mesh && node->mesh->weights_count)
		{
			assert(node->mesh->weights_count == components);
			memcpy(result->f, node->mesh->weights, components * sizeof(real_t));
		}
		break;

	default:
		assert(!"Unknown animation path");
	}
}

static real_t getWorldScale(cgltf_node* node)
{
	real_t transform[16];
	cgltf_node_transform_world(node, transform);

	// 3x3 determinant computes scale^3
	real_t a0 = transform[5] * transform[10] - transform[6] * transform[9];
	real_t a1 = transform[4] * transform[10] - transform[6] * transform[8];
	real_t a2 = transform[4] * transform[9] - transform[5] * transform[8];
	real_t det = transform[0] * a0 - transform[1] * a1 + transform[2] * a2;

	return powf(std::abs(det), 1.0 / 3.0);
}

void processAnimation(Animation& animation, const Settings& settings)
{
	real_t mint = FLT_MAX, maxt = 0;

	for (size_t i = 0; i < animation.tracks.size(); ++i)
	{
		const Track& track = animation.tracks[i];
		assert(!track.time.empty());

		mint = std::min(mint, track.time.front());
		maxt = std::max(maxt, track.time.back());
	}

	mint = std::min(mint, maxt);

	// round the number of frames to nearest but favor the "up" direction
	// this means that at 10 Hz resampling, we will try to preserve the last frame <10ms
	// but if the last frame is <2ms we favor just removing this data
	int frames = 1 + int((maxt - mint) * settings.anim_freq + 0.8f);

	animation.start = mint;
	animation.frames = frames;

	std::vector<Attr> base;

	for (size_t i = 0; i < animation.tracks.size(); ++i)
	{
		Track& track = animation.tracks[i];

		std::vector<Attr> result;
		resampleKeyframes(result, track.time, track.data, track.path, track.interpolation, track.components, frames, mint, settings.anim_freq);

		track.time.clear();
		track.data.swap(result);

		real_t tolerance = getDeltaTolerance(track.path);

		// translation tracks use world space tolerance; in the future, we should compute all errors as linear using hierarchy
		if (track.node && track.path == cgltf_animation_path_type_translation)
		{
			real_t scale = getWorldScale(track.node);
			tolerance /= scale == 0.0 ? 1.0 : scale;
		}

		real_t deviation = getMaxDelta(track.data, track.path, frames, &track.data[0], track.components);

		if (deviation <= tolerance)
		{
			// track is constant (equal to first keyframe), we only need the first keyframe
			track.constant = true;
			track.data.resize(track.components);

			// track.dummy is true iff track redundantly sets up the value to be equal to default node transform
			base.resize(track.components);
			getBaseTransform(&base[0], track.components, track.path, track.node);

			track.dummy = getMaxDelta(track.data, track.path, 1, &base[0], track.components) <= tolerance;
		}
	}
}
