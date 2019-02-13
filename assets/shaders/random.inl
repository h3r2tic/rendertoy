uint hash(uint x) {
	x += (x << 10u);
	x ^= (x >>  6u);
	x += (x <<  3u);
	x ^= (x >> 11u);
	x += (x << 15u);
	return x;
}

float rand_float(uint h) {
	const uint mantissaMask = 0x007FFFFFu;
	const uint one = 0x3F800000u;

	h &= mantissaMask;
	h |= one;

	float  r2 = uintBitsToFloat( h );
	return r2 - 1.0;
}
