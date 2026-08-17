/** @type {import('next').NextConfig} */
const nextConfig = {
  // Strict Mode monta dos veces y rompe el runtime WASM de MediaPipe (Module.arguments).
  reactStrictMode: false,
  async headers() {
    return [
      {
        source: "/mediapipe/holistic/:path*.wasm",
        headers: [{ key: "Content-Type", value: "application/wasm" }],
      },
      {
        source: "/mediapipe/holistic/:path*.data",
        headers: [{ key: "Content-Type", value: "application/octet-stream" }],
      },
      {
        source: "/mediapipe/holistic/:path*.js",
        headers: [{ key: "Content-Type", value: "application/javascript" }],
      },
    ];
  },
  async rewrites() {
    // Siempre proxy interno al backend local (evita loop 502 con Tailscale Serve HTTPS).
    const apiUrl = process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${apiUrl}/api/:path*` }];
  },
};

module.exports = nextConfig;
