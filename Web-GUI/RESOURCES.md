# Skysafe Chatbot - Web GUI Resources

## Overview
This document lists all external resources and image assets used in the Skysafe Chatbot Web GUI.

## CDN Resources (External)

### Markdown Rendering Library
- **Name**: marked.js
- **Purpose**: Convert markdown text to HTML
- **URL**: `https://cdn.jsdelivr.net/npm/marked/marked.min.js`
- **Size**: ~40 KB (minified)
- **License**: MIT

### HTML Sanitization Library
- **Name**: DOMPurify
- **Purpose**: Sanitize HTML to prevent XSS attacks
- **URL**: `https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js`
- **Size**: ~17 KB (minified)
- **License**: Apache 2.0 / MPL 2.0

## CSS Resources

### System Fonts (No External Dependencies)
The application uses system fonts for optimal performance:
- Primary: `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif`
- Monospace (code blocks): `'Courier New', monospace`

## Local Image Assets

### Recommended Folder Structure
```
pictures/
├── logo/
│   ├── skysafe-logo.svg (256x256 px)
│   └── skysafe-logo-dark.svg (256x256 px)
├── icons/
│   ├── loading-spinner.svg (64x64 px)
│   ├── send-icon.svg (24x24 px)
│   ├── source-icon.svg (20x20 px)
│   └── close-icon.svg (24x24 px)
└── backgrounds/
    └── gradient-bg.png (1920x1080 px)
```

### Asset Specifications

#### Logo
- **File**: `skysafe-logo.svg`
- **Format**: SVG
- **Recommended Size**: 256x256 px
- **Purpose**: Header branding
- **Color**: Primary brand color (#667eea)

#### Loading Spinner
- **File**: `loading-spinner.svg`
- **Format**: SVG (animated)
- **Recommended Size**: 64x64 px
- **Purpose**: Loading indicator animation
- **Note**: Currently implemented with CSS animation; SVG optional

#### Send Icon
- **File**: `send-icon.svg`
- **Format**: SVG
- **Recommended Size**: 24x24 px
- **Purpose**: Send button icon
- **Color**: White (#FFFFFF)

#### Source Icon
- **File**: `source-icon.svg`
- **Format**: SVG
- **Recommended Size**: 20x20 px
- **Purpose**: Source reference indicator
- **Color**: Primary brand color (#667eea)

## Current Implementation

### No External Images Required
The current implementation uses:
- **CSS Gradients** for backgrounds
- **CSS Animations** for loading spinner
- **System Fonts** for all typography
- **Inline SVG** (if needed)

This keeps the application lightweight and fast.

## Adding Custom Images

To add custom images to the application:

1. Place image files in `pictures/` folder
2. Reference them in CSS or HTML as:
   ```css
   background-image: url('/pictures/filename.png');
   ```
   ```html
   <img src="/pictures/filename.png" alt="description">
   ```

## Performance Considerations

- All images should be optimized for web:
  - Use SVG for icons and logos (scalable, small size)
  - Use WebP format for photographs where supported
  - Use PNG with optimization for complex images
  - Maximum recommended image size: 100 KB
  - Recommended total assets per page: < 500 KB

## Browser Compatibility

All resources and libraries used are compatible with:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari 14+, Chrome Mobile)

## Future Enhancements

Potential assets to add:
- User avatar icons
- Animated wave SVG for header
- Custom fonts (e.g., for better typography)
- Dark mode theme resources
- Favicon and Apple touch icon
