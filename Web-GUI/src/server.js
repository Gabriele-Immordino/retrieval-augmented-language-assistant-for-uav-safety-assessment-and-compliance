const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();

// Configuration
const PORT = process.env.PORT || 3000;
const BACKEND_URL = process.env.BACKEND_URL || 'http://160.85.96.226:8080';

// Static assets middleware (no body parsing before proxy to preserve raw stream)
app.use(express.static(path.join(__dirname, '../public')));

// Proxy configuration
const apiProxy = createProxyMiddleware({
    target: BACKEND_URL,
    changeOrigin: true,
    pathRewrite: {
        '^/api': '/api',
    },
    onProxyReq: (proxyReq, req, res) => {
        // Log proxied requests
        console.log(`[PROXY] ${req.method} ${req.path} -> ${BACKEND_URL}${req.path}`);
    },
    onError: (err, req, res) => {
        console.error(`[PROXY ERROR] ${err.message}`);
        res.status(503).json({
            error: {
                code: 'backend_unavailable',
                message: 'Backend service is unavailable',
            }
        });
    }
});

// Apply proxy to all /api routes
app.use('/api', apiProxy);

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Home and entry routes
app.get(['/', '/home'], (req, res) => {
    res.sendFile(path.join(__dirname, '../public/home.html'));
});

// Classification page route
app.get('/classification', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/classification.html'));
});

// Chat page route
app.get('/chat', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/chatbot.html'));
});

// Serve home.html for all other routes (SPA routing)
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/home.html'));
});

// Error handler
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: {
            code: 'internal_error',
            message: 'An internal server error occurred',
        }
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Web GUI server running on port ${PORT}`);
    console.log(`Backend URL: ${BACKEND_URL}`);
    console.log(`Visit http://localhost:${PORT} to access the chatbot`);
});
