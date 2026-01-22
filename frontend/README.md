# Semi-Supervised Labeling Framework - Frontend

A modern React + TypeScript frontend built with Vite.js for the Semi-Supervised Labeling Framework.

## Features

- **Main Workflow**: Complete pipeline with 4 steps (Embeddings → Clustering → Labeling → KNN Training)
- **KNN Prediction**: Use trained KNN models to predict labels
- **LLM Labeling**: Direct LLM-based labeling for individual items
- **Evaluation**: Compare and evaluate two labeling methods

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router** - Routing
- **Axios** - HTTP client
- **Lucide React** - Icons

## Setup

### Development

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Create `.env` file**:
   ```env
   VITE_API_URL=http://localhost:8000/api/v1
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

   The app will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
```

The built files will be in the `dist` directory.

### Docker

See `DOCKER.md` for Docker setup instructions.

## Project Structure

```
src/
├── components/       # Reusable UI components
├── pages/           # Page components
├── services/        # API services
├── types/           # TypeScript types
├── App.tsx          # Main app component
├── main.tsx         # Entry point
└── index.css        # Global styles
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The frontend communicates with the backend API at `http://localhost:8000/api/v1`. The API URL can be configured via the `VITE_API_URL` environment variable.

## Pages

1. **Workflow Page** (`/`) - Main 4-step workflow
2. **KNN Prediction** (`/knn-prediction`) - Use KNN model for predictions
3. **LLM Labeling** (`/llm-labeling`) - Direct LLM labeling
4. **Evaluation** (`/evaluation`) - Compare two methods
