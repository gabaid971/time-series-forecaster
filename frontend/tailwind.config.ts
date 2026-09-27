import type { Config } from "tailwindcss";

// Semantic colors bound to the theme variables of app/globals.css
const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: token("canvas"),
        panel: token("panel"),
        ink: token("ink"),
        shade: token("shade"),
        fg: {
          DEFAULT: token("fg"),
          muted: token("fg-muted"),
          subtle: token("fg-subtle"),
        },
        accent: {
          DEFAULT: token("accent"),
          text: token("accent-text"),
        },
        "on-accent": token("on-accent"),
        positive: token("positive"),
        negative: token("negative"),
        warning: token("warning"),
        info: token("info"),
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
    },
  },
  plugins: [],
};
export default config;
