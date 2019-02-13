vec3 uniform_sample_sphere(vec2 urand) {
    const float PI2 = 6.28318530718;
    float z = 1.0 - 2.0 * urand.x;
    float xy = sqrt(max(0.0, 1.0 - z * z));
    float sn = sin(PI2 * urand.y);
	float cs = cos(PI2 * urand.y);
	return vec3(sn * xy, cs * xy, z);
}

// "perp_hm" from https://blog.selfshadow.com/2011/10/17/perp-vectors/
vec3 build_orthonormal_basis(vec3 u) {
    vec3 a = abs(u);
    vec3 v;
    if (a.x <= a.y && a.x <= a.z)
        v = vec3(0, -u.z, u.y);
    else if (a.y <= a.x && a.y <= a.z)
        v = vec3(-u.z, 0, u.x);
    else
        v = vec3(-u.y, u.x, 0);
    return v;
}
