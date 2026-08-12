/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        surface: { DEFAULT: "#1a1a2e", light: "#16213e", dark: "#0f0f1a" },
        accent: { DEFAULT: "#e94560", muted: "#533483" },
      },
    },
  },
  plugins: [],
};
