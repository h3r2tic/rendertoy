struct ViewConstants {
    mat4 view_to_clip;
    mat4 clip_to_view;
    mat4 view_to_sample;
    mat4 sample_to_view;
    mat4 world_to_view;
    mat4 view_to_world;
    vec2 sample_offset_pixels;
    vec2 sample_offset_clip;
};
