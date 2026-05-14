/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f4f4f3",
        ink: "#3a3a3a",
        muted: "#9a9a9a",
        line: "#c4c4c4",
        "dark-bg":    "#18181b",
        "dark-ink":   "#c4c4c0",
        "dark-muted": "#6b6b68",
        "dark-line":  "#3a3a38",
      },
      fontFamily: {
        sans: ["Dosis", "system-ui", "sans-serif"],
        mono: ["DynaPuff", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
