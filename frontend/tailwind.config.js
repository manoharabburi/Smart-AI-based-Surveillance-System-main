/**** Tailwind Config ****/
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}'
  ],
  theme: {
    extend: {
      keyframes: {
        flash: {
          '0%,100%': { backgroundColor: 'rgba(239,68,68,0.15)' },
          '50%': { backgroundColor: 'rgba(239,68,68,0.45)' }
        }
      },
      animation: {
        flash: 'flash 1s ease-in-out 3'
      }
    },
  },
  plugins: []
};

