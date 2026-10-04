/** @type {import('next').NextConfig} */
const nextConfig = {
  // make sure the JSON data files are bundled into the serverless functions on Vercel
  outputFileTracingIncludes: { "/**": ["./data/*.json"] },
  poweredByHeader: false,
};
export default nextConfig;
