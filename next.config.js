/** @type {import('next').NextConfig} */
module.exports = {
  reactStrictMode: true,
  // Python serverless functions in api/ are handled by Vercel's Python runtime;
  // Next never bundles them.
};
