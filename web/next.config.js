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
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      { source: "/api/:path*", destination: `${apiUrl}/api/:path*` },
    ];
  },
};

module.exports = nextConfig;
