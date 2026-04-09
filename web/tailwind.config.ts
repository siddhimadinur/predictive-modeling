import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        sunset: {
          50: "#FFF5F0",
          100: "#FFE8DB",
          200: "#FFD0B5",
          300: "#FFB088",
          400: "#FF8C57",
          500: "#E85D26",
          600: "#C44A1C",
          700: "#9C3A16",
          800: "#7A2E12",
          900: "#5C220E",
        },
        pacific: {
          50: "#F0F7FB",
          100: "#D4E8F4",
          200: "#A8D1E9",
          300: "#6DB5D9",
          400: "#3A97C4",
          500: "#1B6B93",
          600: "#155676",
          700: "#10415A",
          800: "#0B2D3E",
          900: "#061A24",
        },
      },
    },
  },
  plugins: [],
};

export default config;
