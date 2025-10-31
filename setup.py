"""
Automated Setup Script
Helps configure the project quickly
"""
import os
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("✗ Python 3.10 or higher is required")
        print("Please upgrade Python and try again")
        return False
    
    print("✓ Python version is compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'neo4j',
        'ultralytics',
        'pydantic',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall them with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies are installed")
    return True

def setup_env_file():
    """Create .env file from template"""
    print_header("Setting Up Environment File")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✓ .env file already exists")
        overwrite = input("Do you want to reconfigure it? (y/n): ").strip().lower()
        if overwrite != 'y':
            return True
    
    if not env_example_path.exists():
        print("✗ .env.example not found")
        return False
    
    print("\nPlease enter your Neo4j Aura credentials:")
    print("(You can get these from https://console.neo4j.io)")
    
    neo4j_uri = input("\nNeo4j URI (e.g., neo4j+s://xxxxx.databases.neo4j.io): ").strip()
    neo4j_user = input("Neo4j Username (default: neo4j): ").strip() or "neo4j"
    neo4j_password = input("Neo4j Password: ").strip()
    
    if not neo4j_uri or not neo4j_password:
        print("\n✗ URI and password are required")
        return False
    
    # Create .env file
    env_content = f"""# Neo4j Aura Configuration
NEO4J_URI={neo4j_uri}
NEO4J_USER={neo4j_user}
NEO4J_PASSWORD={neo4j_password}

# Server Configuration (Optional)
HOST=0.0.0.0
PORT=8000
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("\n✓ .env file created successfully")
    return True

def check_yolo_model():
    """Check if YOLO model exists"""
    print_header("Checking YOLO Model")
    
    model_path = Path("best.pt")
    
    if model_path.exists():
        print(f"✓ YOLO model found: {model_path}")
        return True
    
    print("✗ YOLO model (best.pt) not found")
    print("\nOptions:")
    print("1. Train a new model: python train_yolo_hijab.py")
    print("2. Copy your existing model to: best.pt")
    print("3. The app will use a default YOLOv8 model as placeholder")
    
    return False

def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = [
        "uploads",
        "test_images",
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")
    
    return True

def test_neo4j_connection():
    """Test Neo4j connection"""
    print_header("Testing Neo4j Connection")
    
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
        
        load_dotenv()
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, user, password]):
            print("✗ Environment variables not set")
            return False
        
        print(f"Connecting to: {uri}")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        
        print("✓ Successfully connected to Neo4j Aura")
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nPlease check:")
        print("- Your Neo4j Aura instance is running")
        print("- Credentials in .env are correct")
        print("- Your IP is whitelisted in Neo4j Aura")
        return False

def print_next_steps():
    """Print next steps"""
    print_header("Setup Complete!")
    
    print("\n✅ Your project is ready!")
    print("\nNext steps:")
    print("\n1. Start the server:")
    print("   python run_server.py")
    print("\n2. Open API documentation:")
    print("   http://localhost:8000/docs")
    print("\n3. Test the API:")
    print("   python test_api.py")
    print("\n4. (Optional) Train YOLO model:")
    print("   python train_yolo_hijab.py")
    print("\n5. Read the documentation:")
    print("   - README.md - Complete guide")
    print("   - QUICKSTART.md - Quick setup")
    print("   - DEPLOYMENT.md - Deploy to cloud")
    
    print("\n" + "=" * 70)

def main():
    """Main setup function"""
    print("\n" + "=" * 70)
    print("  HIJAB DETECTION BACKEND - AUTOMATED SETUP")
    print("=" * 70)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    deps_installed = check_dependencies()
    if not deps_installed:
        install = input("\nInstall dependencies now? (y/n): ").strip().lower()
        if install == 'y':
            os.system("pip install -r requirements.txt")
        else:
            print("\nPlease install dependencies and run setup again")
            return
    
    # Create directories
    create_directories()
    
    # Setup .env file
    if not setup_env_file():
        print("\n⚠ Environment setup incomplete")
        print("You can manually create .env file later")
    
    # Test Neo4j connection
    test_neo4j_connection()
    
    # Check YOLO model
    check_yolo_model()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Setup cancelled by user")
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        print("Please check the error and try again")
