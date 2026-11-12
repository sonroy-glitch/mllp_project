import path from "node:path";

const nextConfig = {
  outputFileTracingRoot: path.resolve(import.meta.dirname),
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
};

export default nextConfig;
