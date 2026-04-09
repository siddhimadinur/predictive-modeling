/** @type {import('next').NextConfig} */
const nextConfig = {
  rewrites: async () => [
    {
      source: "/api/:path*",
      destination: "/api/:path*",
    },
  ],
};

export default nextConfig;
