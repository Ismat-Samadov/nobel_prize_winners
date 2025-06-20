#!/bin/bash

# Deployment setup script for Active Learning NER

echo "🚀 Active Learning NER - Deployment Setup"
echo "=========================================="

echo ""
echo "Choose deployment mode:"
echo "1) Demo Mode (Simple, no ML dependencies - Recommended for first deployment)"
echo "2) Unified Mode (Full ML stack in one service)"
echo "3) Separate Services (Frontend + Backend separately)"
echo "4) Local Development"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Setting up Demo Mode deployment..."
        cp render-simple.yaml render.yaml
        echo "✅ Configured for demo deployment"
        echo "📝 This mode uses:"
        echo "   - Simplified rule-based NER (no heavy ML libraries)"
        echo "   - Single service deployment"
        echo "   - Fast build times"
        echo ""
        echo "🚀 To deploy:"
        echo "   1. Commit and push to GitHub"
        echo "   2. Connect repo to Render"
        echo "   3. Deploy using render.yaml"
        ;;
    2)
        echo "Setting up Unified Mode deployment..."
        cp render-unified.yaml render.yaml
        echo "✅ Configured for unified deployment"
        echo "📝 This mode uses:"
        echo "   - Full BERT-based NER model"
        echo "   - Single service deployment"
        echo "   - Longer build times (includes ML libraries)"
        echo ""
        echo "⚠️  Note: May require paid Render plan for sufficient resources"
        ;;
    3)
        echo "Setting up Separate Services deployment..."
        # render.yaml is already set up for separate services
        echo "✅ Already configured for separate services"
        echo "📝 This mode uses:"
        echo "   - Frontend service (Node.js)"
        echo "   - Backend service (Python)"
        echo "   - Independent scaling"
        echo ""
        echo "💰 Note: Uses 2 Render services (higher cost)"
        ;;
    4)
        echo "Setting up Local Development..."
        echo ""
        echo "Choose local development mode:"
        echo "a) Demo mode (fast, no ML dependencies)"
        echo "b) Full mode (complete ML stack)"
        echo ""
        read -p "Enter choice (a/b): " dev_choice
        
        case $dev_choice in
            a)
                echo "🔧 Local Demo Development:"
                echo "npm run build:frontend"
                echo "mkdir -p api/static && cp -r frontend/build/* api/static/"
                echo "cd api && python -m uvicorn main_unified_simple:app --reload"
                echo ""
                echo "Access at: http://localhost:8000"
                ;;
            b)
                echo "🔧 Local Full Development:"
                echo "Terminal 1: npm run dev:api"
                echo "Terminal 2: npm run dev:frontend"
                echo ""
                echo "Or use: npm run dev"
                echo ""
                echo "Frontend: http://localhost:3000"
                echo "Backend: http://localhost:8000"
                ;;
        esac
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "📚 Additional Resources:"
echo "   - DEPLOYMENT_OPTIONS.md - Detailed deployment comparison"
echo "   - README_DEPLOYMENT.md - Comprehensive deployment guide"
echo "   - README.md - Full project documentation"
echo ""
echo "🎉 Setup complete!"