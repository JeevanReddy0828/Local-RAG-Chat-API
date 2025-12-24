# RAG Chat React UI

A minimal, responsive React frontend for the Local RAG Chat API.

## Features

- 🌙 Dark theme with minimal design
- 📱 Fully responsive (mobile-friendly)
- ⚡ Real-time streaming responses (SSE)
- 📄 Document upload with drag-and-drop
- 📊 Index statistics display
- 💾 Session management
- ⌨️ Keyboard shortcuts (Enter to send)

## Quick Start

```bash
cd ui-react
npm install
npm run dev
```

Opens at http://localhost:3000

> Make sure the FastAPI backend is running on http://localhost:8000

## Build for Production

```bash
npm run build
```

Output goes to `dist/` folder.

## Project Structure

```
ui-react/
├── src/
│   ├── App.jsx       # Main React component
│   └── main.jsx      # Entry point
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## API Proxy

In development, Vite proxies API requests to the backend:

- `/upload` → `http://localhost:8000/upload`
- `/chat` → `http://localhost:8000/chat`
- `/chat/stream` → `http://localhost:8000/chat/stream`
- `/stats` → `http://localhost:8000/stats`

For production, configure your web server to proxy these routes.
