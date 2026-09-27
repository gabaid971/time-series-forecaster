# Time Series Forecaster Frontend

This is the frontend for the Time Series Forecaster application, built with Next.js and Tailwind CSS.
See the [root README](../README.md) for the full setup.

## Getting Started

1.  Configure the backend URL:

    ```bash
    cp .env.example .env.local
    ```

2.  Install dependencies:

    ```bash
    npm install
    ```

3.  Run the development server:

    ```bash
    npm run dev
    ```

4.  Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

-   `app/`: Contains the application routes and pages.
-   `components/`: UI components (strategy step, charts, lag analysis panel).
-   `lib/`: API client and formatters.
-   `types/`: TypeScript type definitions for the application.
-   `public/`: Static assets.
