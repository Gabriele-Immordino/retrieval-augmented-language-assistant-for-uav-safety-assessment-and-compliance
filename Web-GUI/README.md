# Skysafe Chatbot Web GUI

A modern, responsive web interface for the Skysafe Chatbot, built with Express.js and vanilla JavaScript.

## Features

- 💬 Real-time chat interface with markdown rendering
- 🧭 User-specific chat history fetch (per-user)
- 🔄 Loading animation while waiting for responses
- 📚 Source references sidebar
- 🎨 Modern gradient UI with smooth animations
- 📱 Fully responsive design (mobile, tablet, desktop)
- 🔒 XSS protection with HTML sanitization
- 📊 Classification page for initial operation indicators
- 🐳 Docker support for easy deployment

## Project Structure

```
Web-GUI/
├── public/                 # Static files served to clients
│   ├── index.html         # Main HTML page
│   ├── classification.html # Indicators/classification page
│   ├── css/
│   │   └── style.css      # Main stylesheet
│   └── js/
│       ├── app.js         # Chat client logic
│       └── classification.js # Classification client logic
├── src/
│   └── server.js          # Express.js server
├── pictures/              # Image assets folder
├── package.json           # Node.js dependencies
├── Dockerfile             # Docker build configuration
├── docker-compose.example.yml
├── RESOURCES.md           # Asset documentation
└── README.md              # This file
```

## Prerequisites

- Node.js 14.0.0 or higher
- npm 6.0.0 or higher
- (Optional) Docker and Docker Compose for containerized deployment

## Installation

### Local Development

1. Install dependencies:
```bash
cd Web-GUI
npm install
```

2. Start the development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

### Development with Auto-Reload

Install nodemon for development:
```bash
npm install -g nodemon
# or
npm install --save-dev nodemon
```

Then run:
```bash
npm run dev
```

## Configuration

### Environment Variables

Create a `.env` file in the `Web-GUI` directory:

```env
PORT=3000
BACKEND_URL=http://localhost:8080
NODE_ENV=production
```

- `PORT`: Port number for the web server (default: 3000)
- `BACKEND_URL`: URL of the Skysafe backend server (default: http://localhost:8080)
- `NODE_ENV`: Node environment (development/production)

## API Integration

- Chat: `/api/v1/chat` (POST)
- Chat history: `/api/v1/chat_history?user_id={userId}` (GET)
- Classification: `/api/v1/classify` (POST)

### Request Format

```json
{
  "user_id": "user123",
  "user_name": "Alice",
  "preprocess_query": true,
  "chat_history": [
    {"role": "user", "content": "What is Skysafe?"},
    {"role": "assistant", "content": "Skysafe is a..."},
    {"role": "user", "content": "How does it ensure safety?"}
  ]
}
```

### Chat History Response (GET)

```json
{
  "user_id": "user123",
  "chat_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
  ]
}
```

### Classification Request (POST)

```json
{
  "user_id": "user123",
  "user_name": "Alice",
  "operation_input": {
    "maximum_takeoff_mass_category": "lt_25kg",
    "vlos_or_bvlos": "VLOS",
    "ground_environment": "sparsely_populated",
    "airspace_type": "uncontrolled",
    "maximum_altitude_category": "gt_50m_le_120m"
  },
  "indicators": [
    "likely_regulatory_pathway",
    "initial_ground_risk_orientation",
    "initial_air_risk_orientation",
    "expected_assessment_depth"
  ]
}
```

### Response Format

```json
{
  "user_id": "user123",
  "user_name": "Alice",
  "original_question": "How does it ensure safety?",
  "answer": "# Safety Features\n\nSkysafe ensures safety through...",
  "sources": ["Source 1", "Source 2"],
  "reasoning": "Processing..."
}
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t skysafe-web-gui:latest .
```

### Run Container

```bash
docker run -p 3000:3000 \
  -e BACKEND_URL=http://backend:8080 \
  skysafe-web-gui:latest
```

### Docker Compose

Use the provided `docker-compose.example.yml`:

```bash
# Copy example file
cp docker-compose.example.yml docker-compose.yml

# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The application will be available at `http://localhost:3000`

## Features in Detail

### Chat Interface
- Messages scroll automatically as new content arrives
- User messages appear on the right (blue)
- Assistant messages appear on the left (gray)
- Welcome message shown on first load

### Markdown Rendering
The assistant responses support:
- Headers (# ## ###)
- **Bold** and *italic* text
- Bullet lists and numbered lists
- Code blocks with syntax highlighting
- Blockquotes
- Links

### Loading Animation
- Animated spinner shown while waiting for backend
- "Thinking..." status text
- Input disabled during processing
- Prevents multiple simultaneous requests

### Sources Sidebar
- Displays relevant sources for the answer
- Updates with each response
- Shows "No sources available" when empty
- Click-ready for future expansion

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari 14+, Chrome Android)

## Performance

- Uses CDN for markdown and sanitization libraries
- Optimized CSS with minimal repaints
- Lazy loading of markdown parser
- Efficient chat history management
- Responsive design without external UI frameworks

## Security

- HTML sanitization using DOMPurify
- XSS protection for all user input
- CORS-aware proxy to backend
- No sensitive data stored in client-side storage

## Troubleshooting

### Backend Connection Issues

**Problem**: "Backend service is unavailable"
- Ensure backend server is running on the configured URL
- Check `BACKEND_URL` environment variable
- Verify firewall/network settings

**Problem**: CORS errors
- The Express proxy should handle this automatically
- Check that backend is accessible from the Web GUI container

### Chat Not Sending

**Problem**: Messages not being sent
- Check browser console for errors (F12 → Console)
- Verify backend is responding with valid JSON
- Try clearing browser cache and reloading

### Markdown Not Rendering

**Problem**: Markdown text appears as plain text
- CDN libraries may not have loaded
- Check browser console for errors
- Verify internet connection for CDN resources

## Development

### Code Structure

- `public/index.html` - Main page template
- `public/css/style.css` - All styling (no external CSS framework)
- `public/js/app.js` - Client-side logic (vanilla JS, no frameworks)
- `src/server.js` - Express server and proxy configuration

### Adding Features

1. **New UI Components**: Edit `public/index.html` and `public/css/style.css`
2. **Client Logic**: Add functions to `public/js/app.js`
3. **Server Features**: Modify `src/server.js`

## Dependencies

### Production
- `express`: Web framework
- `http-proxy-middleware`: Backend proxy

### Development
- `nodemon`: Auto-reload development server

## License

MIT

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review browser console for error messages
3. Verify backend service is running
4. Check environment variables are set correctly

## Future Enhancements

- [ ] User authentication
- [ ] Chat persistence with database
- [ ] File upload support
- [ ] Dark mode theme
- [ ] Multi-language support
- [ ] Advanced formatting options
- [ ] Speech-to-text input
- [ ] Export conversation feature
