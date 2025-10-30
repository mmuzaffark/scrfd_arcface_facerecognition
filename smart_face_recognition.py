#!/usr/bin/env python3
"""
Smart Face Recognition System with NumPy-based Similarity Search + Database
Solves classical face recognition problems: face mismatch, false positives, scalability
No FAISS dependency - uses NumPy for fast similarity search
"""

import os
import cv2
import sqlite3
import numpy as np
import math
# import faiss  # Using NumPy-based similarity search instead
import logging
import json
import time
import requests
import aiohttp
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from json_storage import save_clustering_results
import hashlib
from urllib.parse import urlparse
import tempfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from compare_face_from_api import FaceComparisonFromAPI
from qdrant_manager import QdrantManager
import uvicorn
import base64
from PIL import Image
import io
from contextlib import contextmanager
from functools import wraps
import gc

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Only log if function takes more than 0.1 seconds
        duration = end_time - start_time
        if duration > 0.1:
            print(f"⏱️  {func.__name__} took {duration:.2f} seconds")
        
        return result
    return wrapper


def load_api_config():
    """Load API configuration from api_config.txt file"""
    config = {
        'api_url': 'https://api.analytics.thefusionapps.com/api/v2/retail/visit',
        'auth_token': '',
        'api_key': '',
        'default_start_date': '2025-10-22',  # October 22, 2025 (2161 visits available)
        'default_end_date': '2025-10-22',
        'default_start_time': '00:00:00',
        'default_end_time': '23:59:59',
        'default_page': 0,
        'default_limit': 50,  # Smaller page size to avoid timeouts
        'default_all_branch': True,
        'default_max_visits': 1000  # Increased to get 1000+ images
    }
    
    try:
        if os.path.exists('api_config.txt'):
            with open('api_config.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == 'api_url':
                            config['api_url'] = value
                        elif key == 'auth_token':
                            config['auth_token'] = value
                        elif key == 'api_key':
                            config['api_key'] = value
                        elif key == 'default_start_date':
                            config['default_start_date'] = value
                        elif key == 'default_end_date':
                            config['default_end_date'] = value
                        elif key == 'default_start_time':
                            config['default_start_time'] = value
                        elif key == 'default_end_time':
                            config['default_end_time'] = value
                        elif key == 'default_page':
                            config['default_page'] = int(value)
                        elif key == 'default_limit':
                            config['default_limit'] = int(value)
                        elif key == 'default_all_branch':
                            config['default_all_branch'] = value.lower() == 'true'
                        elif key == 'default_max_visits':
                            config['default_max_visits'] = int(value)
                        elif key == 'api_key':
                            config['api_key'] = value
    except Exception as e:
        print(f"Warning: Could not load api_config.txt: {e}")
    
    return config


class SmartFaceRecognition:
    def __init__(self, 
                 database_path: str = None,
                 model_name: str = None,
                 gpu_id: int = None,
                 confidence_thresh: float = None,
                 similarity_thresh: float = None,
                 quality_thresh: float = None,
                 config_file: str = "config.json"):
        """
        Initialize Smart Face Recognition System
        
        Args:
            database_path: Path to SQLite database (overrides config)
            model_name: InsightFace model name (overrides config)
            gpu_id: GPU ID (0 for GPU, -1 for CPU) (overrides config)
            confidence_thresh: Face detection confidence threshold (overrides config)
            similarity_thresh: Face similarity threshold for matching (overrides config)
            quality_thresh: Face quality threshold for registration (overrides config)
            config_file: Path to configuration JSON file
        """
        # Load configuration from JSON file
        self.config = self.load_config(config_file)
        
        # Initialize logger after config is loaded
        self.setup_logging()
        
        # Set configuration values (command line args override config file)
        self.database_path = database_path or self.config['system']['database_path']
        self.model_name = model_name or self.config['system']['model_name']
        self.gpu_id = gpu_id if gpu_id is not None else self.config['system']['gpu_id']
        self.confidence_thresh = confidence_thresh if confidence_thresh is not None else self.config['face_detection']['confidence_threshold']
        self.similarity_thresh = similarity_thresh if similarity_thresh is not None else self.config['face_recognition']['similarity_threshold']
        self.quality_thresh = quality_thresh if quality_thresh is not None else self.config['face_detection']['quality_threshold']
        
        # Multi-metric similarity settings
        self.use_multi_metric_similarity = self.config.get('face_comparison', {}).get('use_multi_metric_similarity', False)
        self.similarity_weights = self.config.get('face_comparison', {}).get('similarity_weights', {
            'cosine': 0.5,
            'euclidean': 0.25,
            'manhattan': 0.15,
            'pearson': 0.1
        })
        
        # Initialize components
        self.app = None
        self.face_quality_cache = {}
        self.image_cache = {}  # Cache for processed images
        self.image_cache_dir = self.config['system']['image_cache_dir']  # Directory for cached images
        
        # Initialize vector database
        self.vector_db = QdrantManager(self.config)
        
        # Performance optimization settings
        self.max_cache_size = 1000  # Maximum number of images to cache
        self.batch_size = 5  # Reduced batch size for GPU memory management
        self._db_connection = None  # Reusable database connection
        
        # GPU memory management
        self.gpu_memory_available = True
        self.gpu_memory_error_count = 0
        
        # Thread lock for model access (InsightFace is not thread-safe)
        self.model_lock = threading.Lock()
        
        # Initialize webhook manager
        # JSON storage manager is imported globally
        
        # Setup
        self.setup_database()
        self.run_database_migrations()
        self.setup_image_cache()
        self.initialize_model()
        self.load_embeddings()
        
        # Log multi-metric similarity status
        self.logger.info(f"Multi-metric similarity: {'Enabled' if self.use_multi_metric_similarity else 'Disabled'}")
        
        # Ensemble configuration
        self.use_choronological = self.config.get('face_recognition', {}).get('use_choronological', False)
        self.ensemble_models_config = self.config.get('face_recognition', {}).get('ensemble_models', ['buffalo_l', 'arcface_r100_v2'])
        
        # Import ensemble module (lazy import)
        self.ensemble_handler = None
        if self.use_choronological:
            try:
                from ensemble_face_recognition import EnsembleFaceRecognition
                self.ensemble_handler = EnsembleFaceRecognition(self.config, self.logger)
                self.logger.info("Ensemble face recognition enabled")
            except Exception as e:
                self.logger.error(f"Failed to load ensemble module: {e}")
                self.use_choronological = False
        
        # Advanced clustering
        self.clustering_handler = None
        try:
            from clustering_advanced import AdvancedClustering, CLUSTERING_AVAILABLE
            if CLUSTERING_AVAILABLE:
                self.clustering_handler = AdvancedClustering(self.config, self.logger)
                clustering_method = self.config.get('face_recognition', {}).get('clustering', {}).get('method', 'threshold')
                self.logger.info(f"Advanced clustering enabled: method={clustering_method}")
            else:
                self.logger.warning("scipy/sklearn not available, advanced clustering disabled")
        except Exception as e:
            self.logger.error(f"Failed to load clustering module: {e}")
    
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with automatic cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(self.database_path)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=10000")  # Increase cache size
            conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def cleanup_memory(self):
        """Clean up memory by clearing caches and forcing garbage collection"""
        # Clear image cache if it gets too large
        if len(self.image_cache) > self.max_cache_size:
            # Keep only the most recent 50% of cached images
            items_to_remove = len(self.image_cache) - (self.max_cache_size // 2)
            keys_to_remove = list(self.image_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.image_cache[key]
        
        # Clear face quality cache
        self.face_quality_cache.clear()
        
        # Force garbage collection
        gc.collect()
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        try:
            import gc
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info("GPU memory cache cleared")
            except ImportError:
                pass
            
            # Additional cleanup for InsightFace/ONNX
            try:
                if hasattr(self, 'app') and self.app is not None:
                    # DON'T delete the model - pristine faces can't work without it
                    # Just force garbage collection to clean up memory
                    gc.collect()
                    self.logger.info("Memory cleaned without deleting model")
            except Exception as e:
                self.logger.warning(f"Error during model cleanup: {e}")
                
        except Exception as e:
            self.logger.warning(f"Could not clear GPU memory: {e}")
    
    def check_gpu_memory(self):
        """Check if GPU memory is available"""
        if not self.gpu_memory_available:
            return False
        
        try:
            import torch
            if torch.cuda.is_available():
                # Check if GPU memory is over 90% used
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                memory_total = torch.cuda.get_device_properties(0).total_memory
                
                usage_percent = (memory_reserved / memory_total) * 100
                if usage_percent > 90:
                    self.logger.warning(f"GPU memory usage high: {usage_percent:.1f}%")
                    return False
                return True
        except Exception as e:
            self.logger.warning(f"Could not check GPU memory: {e}")
            return False
        
        self.logger.info(f"Memory cleanup completed. Image cache size: {len(self.image_cache)}")
    
    def load_config(self, config_file: str) -> dict:
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found, using defaults")
            # Return default configuration if file not found
            return {
                'system': {
                    'database_path': 'face_database.db',
                    'model_name': 'buffalo_l',
                    'gpu_id': 0,
                    'image_cache_dir': 'image_cache'
                },
                'face_detection': {
                    'confidence_threshold': 0.5,
                    'quality_threshold': 0.3
                },
                'face_recognition': {
                    'similarity_threshold': 0.4
                }
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file {config_file}: {e}")
            raise
        except Exception as e:
            print(f"Error loading configuration file {config_file}: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    @monitor_performance
    def setup_database(self):
        """Setup SQLite database for person metadata"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create persons table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT,
                    face_quality REAL,
                    face_hash TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    match_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create face_quality table for quality assessment
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    quality_score REAL,
                    blur_score REAL,
                    pose_score REAL,
                    lighting_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            # Create person_visits table for web interface
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    visit_id TEXT,
                    customer_id TEXT,
                    entry_time TEXT,
                    image_url TEXT,
                    saved_image_path TEXT,
                    similarity REAL,
                    branchId TEXT,
                    camera TEXT,
                    entryEventIds TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            # Add new columns to existing person_visits table if they don't exist
            try:
                cursor.execute("ALTER TABLE person_visits ADD COLUMN branchId TEXT DEFAULT ''")
                self.logger.info("Added branchId column to person_visits table")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE person_visits ADD COLUMN camera TEXT DEFAULT ''")
                self.logger.info("Added camera column to person_visits table")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            try:
                cursor.execute("ALTER TABLE person_visits ADD COLUMN entryEventIds TEXT DEFAULT ''")
                self.logger.info("Added entryEventIds column to person_visits table")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            conn.commit()
            self.logger.info("Database setup completed")
    
    def run_database_migrations(self):
        """Run database migrations to update schema"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Migration 1: Add reason column to low_similarity_images table
            try:
                cursor.execute('ALTER TABLE low_similarity_images ADD COLUMN reason TEXT')
                conn.commit()
                self.logger.info("Added reason column to low_similarity_images table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    self.logger.info("Reason column already exists in low_similarity_images table")
                else:
                    self.logger.warning(f"Could not add reason column: {e}")
            
            # Migration 2: Remove embedding column from persons table (moved to Qdrant)
            try:
                # Check if embedding column exists
                cursor.execute("PRAGMA table_info(persons)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'embedding' in columns:
                    # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
                    self.logger.info("Migrating persons table to remove embedding column...")
                    
                    # Create new table without embedding column
                    cursor.execute('''
                        CREATE TABLE persons_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            image_path TEXT,
                            face_quality REAL,
                            face_hash TEXT UNIQUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            match_count INTEGER DEFAULT 0
                        )
                    ''')
                    
                    # Copy data from old table to new table (excluding embedding)
                    cursor.execute('''
                        INSERT INTO persons_new (id, name, image_path, face_quality, face_hash, created_at, last_seen, match_count)
                        SELECT id, name, image_path, face_quality, face_hash, created_at, last_seen, match_count
                        FROM persons
                    ''')
                    
                    # Drop old table and rename new table
                    cursor.execute('DROP TABLE persons')
                    cursor.execute('ALTER TABLE persons_new RENAME TO persons')
                    
                    conn.commit()
                    self.logger.info("Successfully migrated persons table - embedding column removed")
                else:
                    self.logger.info("Persons table already migrated - no embedding column found")
                    
            except Exception as e:
                self.logger.warning(f"Could not migrate persons table: {e}")
            
            conn.close()
        except Exception as e:
            self.logger.error(f"Error running database migrations: {e}")
    
    def setup_image_cache(self):
        """Setup image cache directory"""
        os.makedirs(self.image_cache_dir, exist_ok=True)
        self.logger.info(f"Image cache directory: {self.image_cache_dir}")
    
    @monitor_performance
    def clear_database(self):
        """Clear all data from the database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
        
        try:
            # Clear all tables
            cursor.execute('DELETE FROM person_visits')
            cursor.execute('DELETE FROM face_quality')
            cursor.execute('DELETE FROM persons')
            
            # Reset auto-increment counters
            cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("persons", "face_quality", "person_visits")')
            
            conn.commit()
            self.logger.info("SQLite database cleared successfully")
            
            # Clear Qdrant vector database
            try:
                self.vector_db.clear_all()
                self.logger.info("Qdrant vector database cleared successfully")
            except Exception as e:
                self.logger.error(f"Error clearing Qdrant database: {e}")
        
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            conn.rollback()
            raise
        
        # Clear memory caches
        self.cleanup_memory()
    
    def initialize_model(self):
        """Initialize InsightFace model with GPU error handling and CPU fallback"""
        self.logger.info(f"Loading {self.model_name} model...")
        
        # Check if CPU mode is forced
        force_cpu = self.config.get('system', {}).get('force_cpu_mode', False)
        
        if force_cpu:
            self.logger.info("CPU mode forced by configuration")
            self.gpu_id = -1
        
        # Try GPU first only if not forced to CPU
        if not force_cpu and self.gpu_id >= 0:
            try:
                self.logger.info(f"Attempting to initialize model on GPU {self.gpu_id}")
                self.app = FaceAnalysis(name=self.model_name)
                det_size = tuple(self.config['face_detection']['detection_size'])
                self.app.prepare(ctx_id=self.gpu_id, det_size=det_size)
                self.logger.info("Model loaded successfully on GPU")
                return
            except Exception as e:
                self.logger.warning(f"GPU initialization failed: {e}")
                self.logger.warning("Falling back to CPU mode...")
                # Clear any partial initialization
                if hasattr(self, 'app'):
                    del self.app
        
        # Initialize on CPU
        try:
            self.logger.info("Initializing model on CPU")
            self.app = FaceAnalysis(name=self.model_name)
            det_size = tuple(self.config['face_detection']['detection_size'])
            self.app.prepare(ctx_id=-1, det_size=det_size)  # -1 for CPU
            self.gpu_id = -1  # Update gpu_id to reflect CPU usage
            self.logger.info("Model loaded successfully on CPU")
        except Exception as e:
            self.logger.error(f"Failed to initialize model on CPU: {e}")
            raise RuntimeError(f"Could not initialize InsightFace model: {e}")
    
    def compute_face_hash(self, embedding: np.ndarray) -> str:
        """Compute hash for face embedding to detect duplicates"""
        return hashlib.md5(embedding.tobytes()).hexdigest()
    
    def get_cached_image_path(self, image_url: str) -> str:
        """Get cached image path or create cache entry"""
        # Ensure image cache directory exists
        os.makedirs(self.image_cache_dir, exist_ok=True)
        
        # Create a hash of the URL for cache filename
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        cached_path = os.path.join(self.image_cache_dir, f"{url_hash}.jpg")
        
        # If cached image doesn't exist, download and cache it
        if not os.path.exists(cached_path):
            try:
                image = self.download_image_from_url(image_url, save_path=cached_path)
                if image is not None:
                    self.logger.info(f"Cached image: {image_url} -> {cached_path}")
                else:
                    # If download failed, return None
                    return None
            except Exception as e:
                self.logger.error(f"Error caching image {image_url}: {e}")
                return None
        
        return cached_path
    
    def process_image_for_web(self, image_path: str, max_size: tuple = None) -> Optional[str]:
        """
        Process image for web display with proper resizing and format
        
        Args:
            image_path: Path to the image file
            max_size: Maximum size (width, height) for the processed image
            
        Returns:
            Base64 encoded image string or None if processing failed
        """
        try:
            if not os.path.exists(image_path):
                return None
            
            # Use config default if max_size not provided
            if max_size is None:
                max_size = tuple(self.config['image_processing']['web_max_size'])
            
            # Open and process image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize while maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                jpeg_quality = self.config['image_processing']['jpeg_quality']
                img.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
                buffer.seek(0)
                
                # Encode to base64
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_base64}"
                
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def download_image_from_url(self, url: str, timeout: int = None, save_path: str = None, retry_count: int = 3) -> Optional[np.ndarray]:
        """
        Download image from URL and return as OpenCV image with retry logic
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            save_path: Optional path to save the downloaded image
            retry_count: Number of retry attempts
            
        Returns:
            OpenCV image array or None if failed
        """
        for attempt in range(retry_count):
            try:
                self.logger.info(f"Downloading image from: {url} (attempt {attempt + 1}/{retry_count})")
                
                # Use config default if timeout not provided
                if timeout is None:
                    timeout = self.config['image_processing']['download_timeout']
                
                # Set headers to request image content
                headers = {
                    'User-Agent': self.config['http_headers']['user_agent'],
                    'Accept': self.config['http_headers']['accept'],
                    'Accept-Language': self.config['http_headers']['accept_language'],
                    'Cache-Control': self.config['http_headers']['cache_control']
                }
                
                # Download image
                response = requests.get(url, timeout=timeout, stream=True, headers=headers)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                self.logger.debug(f"Content-Type: {content_type}")
                
                # Check if response is JSON (error response)
                if 'application/json' in content_type:
                    self.logger.warning(f"Received JSON response instead of image from: {url}")
                    try:
                        json_response = response.json()
                        self.logger.warning(f"JSON response: {json_response}")
                    except:
                        self.logger.warning(f"Could not parse JSON response")
                    if attempt < retry_count - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    return None
                
                # Check if response is actually an image
                if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                    self.logger.warning(f"Unexpected content type: {content_type}")
                    # Try to decode anyway in case it's an image with wrong headers
                
                # Convert to OpenCV format
                image_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    self.logger.warning(f"Failed to decode image from URL: {url}")
                    # Try to save raw content for debugging
                    if save_path:
                        raw_path = save_path.replace('.jpg', '_raw.bin')
                        with open(raw_path, 'wb') as f:
                            f.write(response.content)
                        self.logger.info(f"Saved raw content to: {raw_path}")
                    if attempt < retry_count - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    return None
                
                # Save image if path provided
                if save_path:
                    success = cv2.imwrite(save_path, image)
                    if success:
                        self.logger.info(f"Saved image to: {save_path}")
                    else:
                        self.logger.warning(f"Failed to save image to: {save_path}")
                    
                self.logger.info(f"Successfully downloaded image: {image.shape}")
                return image
                
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Timeout downloading {url} (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2)  # Wait longer for timeout retries
                    continue
                return None
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Network error downloading {url} (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                return None
            except Exception as e:
                self.logger.error(f"Error downloading image from {url} (attempt {attempt + 1}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                return None
        
        self.logger.error(f"Failed to download image after {retry_count} attempts: {url}")
        return None
    
    async def download_image_async(self, session: aiohttp.ClientSession, url: str, timeout: int = None) -> Optional[np.ndarray]:
        """
        Async download image from URL and return as OpenCV image
        
        Args:
            session: aiohttp ClientSession for connection pooling
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            OpenCV image array or None if failed
        """
        try:
            # Use config default if timeout not provided
            if timeout is None:
                timeout = self.config['image_processing']['download_timeout']
            
            # Set headers to request image content
            headers = {
                'User-Agent': self.config['http_headers']['user_agent'],
                'Accept': self.config['http_headers']['accept'],
                'Accept-Language': self.config['http_headers']['accept_language'],
                'Cache-Control': self.config['http_headers']['cache_control']
            }
            
            # Download image asynchronously
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                
                # Check if response is JSON (error response)
                if 'application/json' in content_type:
                    self.logger.warning(f"Received JSON response instead of image from: {url}")
                    return None
                
                # Check if response is actually an image
                if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                    self.logger.warning(f"Unexpected content type: {content_type}")
                
                # Read image data
                image_data = await response.read()
                
                # Convert to OpenCV format
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    self.logger.warning(f"Failed to decode image from URL: {url}")
                    return None
                
                return image
                
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout downloading {url}")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error downloading {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    async def download_images_batch(self, urls: List[str], max_concurrent: int = None) -> Dict[str, Optional[np.ndarray]]:
        """
        Download multiple images concurrently using async/await
        
        Args:
            urls: List of image URLs to download
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            Dictionary mapping URL to image array (or None if failed)
        """
        if max_concurrent is None:
            max_concurrent = self.config['image_processing']['max_concurrent_downloads']
        
        # Create connection pool with limits
        connector = aiohttp.TCPConnector(
            limit=self.config['image_processing']['connection_pool_size'],
            limit_per_host=max_concurrent,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config['image_processing']['download_timeout'])
        
        results = {}
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create semaphore to limit concurrent downloads
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def download_with_semaphore(url):
                async with semaphore:
                    return url, await self.download_image_async(session, url)
            
            # Download all images concurrently
            tasks = [download_with_semaphore(url) for url in urls]
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for task_result in completed_tasks:
                if isinstance(task_result, Exception):
                    self.logger.error(f"Task failed with exception: {task_result}")
                    continue
                
                url, image = task_result
                results[url] = image
                
                if image is not None:
                    self.logger.info(f"Successfully downloaded: {url} ({image.shape})")
                else:
                    self.logger.warning(f"Failed to download: {url}")
        
        return results
    
    def load_visit_data(self, json_file_path: str) -> List[Dict]:
        """
        Load visit data from JSON file
        
        Args:
            json_file_path: Path to JSON file containing visit data
            
        Returns:
            List of visit records with image URLs
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            visits = data.get('visits', [])
            self.logger.info(f"Loaded {len(visits)} visits from JSON file")
            
            # Filter visits with valid image URLs
            valid_visits = []
            for visit in visits:
                if visit.get('image') and visit.get('image').startswith('http'):
                    valid_visits.append(visit)
            
            self.logger.info(f"Found {len(valid_visits)} visits with valid image URLs")
            return valid_visits
            
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {json_file_path}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in {json_file_path}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading visit data: {e}")
            return []
    
    def fetch_face_comparison_data_from_api(self, api_url: str, start_date: str = None, end_date: str = None, 
                                           page: int = 0, limit: int = 100, start_time: str = None, 
                                           end_time: str = None, all_branch: bool = True, api_key: str = None, 
                                           auth_token: str = None) -> List[Dict]:
        """
        Fetch face comparison data from external API - New structure with url1 and url2
        
        Args:
            api_url: Base API URL
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            page: Page number for pagination
            limit: Number of records per page
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            all_branch: Whether to include all branches
            api_key: API key for authentication (optional)
            auth_token: Bearer token for authentication (optional)
            
        Returns:
            List of face comparison records
        """
        try:
            # Build query parameters for new analytics API
            params = {
                'page': page,
                'limit': limit,
                'allBranch': str(all_branch).lower()
            }
            
            # Add date parameter (new API uses single date parameter)
            if start_date:
                params['date'] = start_date
            
            # Add time parameters if provided
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            # Add other optional parameters
            params.update({
                'nolimit': 'false',
                'isZone': 'false',
                'BlackListed': 'false',
                'Vip': 'false',
                'Vendor': 'false',
                'isDeleted': 'false'
            })
            
            self.logger.info(f"Fetching face comparison data from API: {api_url}")
            self.logger.info(f"Parameters: {params}")
            
            # Prepare headers for authentication
            headers = {}
            if api_key:
                headers['X-API-Key'] = api_key
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            # Make API request
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            # Handle different HTTP status codes
            if response.status_code == 401:
                self.logger.error("API request failed: 401 Unauthorized. Please check your API credentials.")
                return []
            elif response.status_code == 403:
                self.logger.error("API request failed: 403 Forbidden. You don't have permission to access this resource.")
                return []
            elif response.status_code == 404:
                self.logger.error("API request failed: 404 Not Found. The API endpoint may be incorrect.")
                return []
            elif response.status_code == 429:
                self.logger.error("API request failed: 429 Too Many Requests. Rate limit exceeded.")
                return []
            elif not response.ok:
                self.logger.error(f"API request failed: {response.status_code} {response.reason}")
                return []
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            self.logger.info(f"API response received: {len(data) if isinstance(data, list) else 'object'} records")
            
            # Debug: Log the structure of the response (only in debug mode)
            if self.logger.level <= 10:  # DEBUG level
                if isinstance(data, list) and len(data) > 0:
                    self.logger.debug(f"First record structure: {list(data[0].keys()) if data[0] else 'Empty record'}")
                elif isinstance(data, dict):
                    self.logger.debug(f"Response keys: {list(data.keys())}")
                    if 'data' in data:
                        self.logger.debug(f"Data field type: {type(data['data'])}, length: {len(data['data']) if isinstance(data['data'], list) else 'N/A'}")
            
            # Extract visits from response - Updated for new API structure
            raw_visits = data if isinstance(data, list) else data.get('visits', data.get('data', []))
            
            if not raw_visits:
                self.logger.warning("No visits found in API response")
                return []
            
            # Transform visits to face comparison format - process both entry and exit events
            comparison_records = []
            for i, visit in enumerate(raw_visits):
                try:
                    visit_id = visit.get('id', f"visit_{i}")
                    customer_id = visit.get('customerId', 'unknown')
                    branch_id = visit.get('branchId', '')
                    ref_image = visit.get('refImage')  # Reference image from database
                    
                    if not ref_image:
                        self.logger.debug(f"No reference image found for visit {visit_id}")
                        continue
                    
                    # Process entry events
                    entry_events = visit.get('entryEventIds', [])
                    for event in entry_events:
                        if isinstance(event, dict) and event.get('fileName'):
                            image_url = f"https://cdn.analytics.thefusionapps.com/v11/{event['fileName']}"
                            
                            comparison_record = {
                                'comparison_id': f"{visit_id}_entry_{event.get('eventId', 'unknown')}",
                                'event_id': event.get('eventId'),
                                'approve': visit.get('isConverted', False),
                                'image1_url': image_url,
                                'image2_url': ref_image,
                                'branch_id': branch_id,
                                'created_at': event.get('timestamp', visit.get('entryTime')),
                                'customer_info': [customer_id],
                                'matched_info': [ref_image],
                                'message': f"Entry event comparison for customer {customer_id}",
                                'is_first_visit': visit.get('isFirstVisit', False),
                                'is_vip': visit.get('isVip', False),
                                'is_blacklisted': visit.get('isBlackListed', False),
                                'event_type': 'entry',
                                'camera': event.get('camera', ''),
                                'raw_data': event
                            }
                            
                            comparison_records.append(comparison_record)
                    
                    # Process exit events
                    exit_events = visit.get('exitEventIds', [])
                    for event in exit_events:
                        if isinstance(event, dict) and event.get('fileName'):
                            image_url = f"https://cdn.analytics.thefusionapps.com/v11/{event['fileName']}"
                            
                            comparison_record = {
                                'comparison_id': f"{visit_id}_exit_{event.get('eventId', 'unknown')}",
                                'event_id': event.get('eventId'),
                                'approve': visit.get('isConverted', False),
                                'image1_url': image_url,
                                'image2_url': ref_image,
                                'branch_id': branch_id,
                                'created_at': event.get('timestamp', visit.get('exitTime', visit.get('entryTime'))),
                                'customer_info': [customer_id],
                                'matched_info': [ref_image],
                                'message': f"Exit event comparison for customer {customer_id}",
                                'is_first_visit': visit.get('isFirstVisit', False),
                                'is_vip': visit.get('isVip', False),
                                'is_blacklisted': visit.get('isBlackListed', False),
                                'event_type': 'exit',
                                'camera': event.get('camera', ''),
                                'raw_data': event
                            }
                            
                            comparison_records.append(comparison_record)
                    
                    # Also process main visit image if it exists
                    main_image = visit.get('image')
                    if main_image and main_image.startswith('http'):
                        comparison_record = {
                            'comparison_id': f"{visit_id}_main",
                            'event_id': None,
                            'approve': visit.get('isConverted', False),
                            'image1_url': main_image,
                            'image2_url': ref_image,
                            'branch_id': branch_id,
                            'created_at': visit.get('entryTime'),
                            'customer_info': [customer_id],
                            'matched_info': [ref_image],
                            'message': f"Main visit comparison for customer {customer_id}",
                            'is_first_visit': visit.get('isFirstVisit', False),
                            'is_vip': visit.get('isVip', False),
                            'is_blacklisted': visit.get('isBlackListed', False),
                            'event_type': 'main',
                            'camera': visit.get('camera', ''),
                            'raw_data': visit
                        }
                        
                        comparison_records.append(comparison_record)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing visit: {e}")
                    continue
            
            self.logger.info(f"Transformed {len(comparison_records)} face comparison records")
            return comparison_records
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching face comparison data from API: {e}")
            return []

    def fetch_visit_data_from_api(self, api_url: str, start_date: str = None, end_date: str = None, 
                                 page: int = 0, limit: int = 100, start_time: str = None, 
                                 end_time: str = None, all_branch: bool = True, selected_branch: str = 'EM-DEPT', 
                                 api_key: str = None, auth_token: str = None, max_visits: int = None) -> List[Dict]:
        """
        Fetch visit data from external API with automatic pagination
        
        Args:
            api_url: Base API URL
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            page: Page number for pagination (starting from 0)
            limit: Number of records per page
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            all_branch: Whether to include all branches
            selected_branch: Specific branch to filter by (EM-DEPT, CBE-TRS, etc.)
            api_key: API key for authentication (optional)
            auth_token: Bearer token for authentication (optional)
            max_visits: Maximum number of visits to fetch (will paginate automatically)
            
        Returns:
            List of visit records with image URLs
        """
        try:
            all_visits = []
            current_page = page
            total_fetched = 0
            
            # Create session for connection reuse with headers that match the working curl command
            session = requests.Session()
            session.headers.update({
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9,en-IN;q=0.8',
                'Connection': 'keep-alive',
                'DNT': '1',
                'Origin': 'https://analytics.develop.thefusionapps.com',
                'Referer': 'https://analytics.develop.thefusionapps.com/',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
                'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36 Edg/141.0.0.0',
                'sec-ch-ua': '"Microsoft Edge";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
                'sec-ch-ua-mobile': '?1',
                'sec-ch-ua-platform': '"Android"',
                'Keep-Alive': 'timeout=60, max=1000'
            })
            
            # Determine if we need pagination
            if max_visits and max_visits > limit:
                self.logger.info(f"Requesting {max_visits} visits with pagination (limit per page: {limit})")
            else:
                self.logger.info(f"Fetching single page: page {page}, limit {limit}")
            
            while True:
                # Build query parameters to match the working API call exactly
                # Simplified parameters based on working example
                params = {
                    'date': start_date,
                    'page': current_page,
                    'limit': limit,
                    'allBranch': 'false' if not all_branch else 'true',
                    'nolimit': 'false',
                    'isZone': 'false',
                    'BlackListed': 'false',
                    'Vip': 'false',
                    'Vendor': 'false',
                    'isDeleted': 'false'
                }
                
                # Log parameters for debugging
                self.logger.info(f"API call parameters: {params}")
                
                self.logger.info(f"Fetching page {current_page} from API: {api_url}")
                self.logger.info(f"Parameters: {params}")
                
                # Add authentication headers to the existing session headers
                if api_key:
                    session.headers['X-API-Key'] = api_key
                if auth_token:
                    session.headers['Authorization'] = f'Bearer {auth_token}'
                
                # Add the If-None-Match header from the working curl command
                session.headers['If-None-Match'] = 'W/"1798fd-vds5yT1u5VoZqudXPZh9yPz5D9M"'
                
                # Make API request with retry logic for timeout handling
                max_retries = 3
                retry_delay = 1
                response = None
                
                for attempt in range(max_retries):
                    try:
                        response = session.get(api_url, params=params, timeout=60)  # Use session with increased timeout
                        break  # Success, exit retry loop
                    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Timeout on page {current_page}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            self.logger.error(f"All retry attempts failed for page {current_page}: {e}")
                            raise e
                
                # Handle different HTTP status codes
                if response.status_code == 401:
                    self.logger.error("API request failed: 401 Unauthorized. Please check your API credentials.")
                    break
                elif response.status_code == 403:
                    self.logger.error("API request failed: 403 Forbidden. You don't have permission to access this resource.")
                    break
                elif response.status_code == 404:
                    self.logger.error("API request failed: 404 Not Found. The API endpoint may be incorrect.")
                    break
                elif response.status_code == 429:
                    self.logger.error("API request failed: 429 Too Many Requests. Rate limit exceeded.")
                    break
                elif not response.ok:
                    self.logger.error(f"API request failed: {response.status_code} {response.reason}")
                    break
                
                response.raise_for_status()
                
                data = response.json()
                
                # Enhanced response parsing - handle multiple API response formats
                page_visits = []
                if isinstance(data, list):
                    page_visits = data
                    self.logger.info(f"Received list response with {len(page_visits)} items")
                elif isinstance(data, dict):
                    # Try all possible response keys for data array
                    possible_keys = ['list', 'data', 'visits', 'results', 'items', 'records', 'content']
                    for key in possible_keys:
                        if key in data and isinstance(data[key], list):
                            page_visits = data[key]
                            self.logger.info(f"Found data in '{key}' key with {len(page_visits)} items")
                            break
                    
                    # If still no data found, check if the dict itself contains visit-like data
                    if not page_visits and 'id' in data:
                        page_visits = [data]  # Single record
                        self.logger.info("Single record response detected")
                    
                    if not page_visits:
                        self.logger.warning(f"No visit data found in response. Available keys: {list(data.keys())}")
                else:
                    self.logger.error(f"Unexpected API response format: {type(data)}")
                    break
                
                page_count = len(page_visits)
                
                # Log API total if available
                if isinstance(data, dict) and 'total' in data:
                    api_total = data.get('total', 0)
                    self.logger.info(f"Page {current_page} received: {page_count} records (API total: {api_total})")
                else:
                    self.logger.info(f"Page {current_page} received: {page_count} records")
                
                # If no records on this page, we've reached the end
                if page_count == 0:
                    self.logger.info("No more records available, stopping pagination")
                    break
                
                # Transform API data to match expected format - process only main image (1:1 mapping)
                page_transformed_visits = []
                for visit in page_visits:
                    # Extract basic visit information
                    visit_id = visit.get('id', visit.get('visitId', visit.get('visit_id')))
                    customer_id = visit.get('customerId', visit.get('customer_id', visit.get('customerId')))
                    branch_id = visit.get('branchId', visit.get('branch_id', ''))
                    entry_time = visit.get('timestamp', visit.get('entryTime', visit.get('entry_time')))
                    exit_time = visit.get('exitTime', visit.get('exit_time'))
                    entry_events = visit.get('entryEventIds', [])
                    exit_events = visit.get('exitEventIds', [])
                    
                    # Process only the main image to ensure 1:1 mapping (389 visits → 389 events)
                    main_image_url = None
                    if visit.get('image') and visit['image'].startswith('http'):
                        main_image_url = visit['image']
                    elif visit.get('imageUrl') and visit['imageUrl'].startswith('http'):
                        main_image_url = visit['imageUrl']
                    
                    if main_image_url:
                        # Extract filename from URL for consistency
                        filename = main_image_url.split('/')[-1] if main_image_url else None
                        
                        transformed_visit = {
                            'id': f"{visit_id}_main",
                            'originalVisitId': visit_id,
                            'customerId': customer_id,
                            'image': main_image_url,
                            'entryTime': entry_time,
                            'event': 'main',
                            'camera': visit.get('camera', visit.get('cameraName', '')),
                            'branchId': branch_id,
                            'eventId': visit_id,  # Use visit ID as event ID for main image
                            'fileName': filename,
                            'age': visit.get('faceResponse', {}).get('age', {}).get('low') if visit.get('faceResponse') else None,
                            'gender': visit.get('faceResponse', {}).get('gender', {}).get('value') if visit.get('faceResponse') else None,
                            'similarity': visit.get('similarity', visit.get('confidence', 0.0)),
                            'entryEventIds': entry_events,
                            'exitEventIds': exit_events
                        }
                        
                        page_transformed_visits.append(transformed_visit)
                    else:
                        # If no main image, try to get the first available image from events
                        first_image_url = None
                        first_event = None
                        
                        # Check entry events first
                        for event in entry_events:
                            if isinstance(event, dict) and event.get('fileName'):
                                first_image_url = f"https://cdn.analytics.thefusionapps.com/v11/{event['fileName']}"
                                first_event = event
                                break
                        
                        # If no entry event, check exit events
                        if not first_image_url:
                            for event in exit_events:
                                if isinstance(event, dict) and event.get('fileName'):
                                    first_image_url = f"https://cdn.analytics.thefusionapps.com/v11/{event['fileName']}"
                                    first_event = event
                                    break
                        
                        # If still no image, check refImage
                        if not first_image_url:
                            ref_image_url = visit.get('refImage')
                            if ref_image_url and ref_image_url.startswith('http'):
                                first_image_url = ref_image_url
                        
                        if first_image_url:
                            # Extract filename from URL
                            filename = first_image_url.split('/')[-1] if first_image_url else None
                            
                            transformed_visit = {
                                'id': f"{visit_id}_fallback",
                                'originalVisitId': visit_id,
                                'customerId': customer_id,
                                'image': first_image_url,
                                'entryTime': first_event.get('timestamp', entry_time) if first_event else entry_time,
                                'event': 'fallback',
                                'camera': first_event.get('camera', visit.get('camera', visit.get('cameraName', ''))) if first_event else visit.get('camera', visit.get('cameraName', '')),
                                'branchId': branch_id,
                                'eventId': first_event.get('eventId', visit_id) if first_event else visit_id,
                                'fileName': filename,
                                'age': visit.get('faceResponse', {}).get('age', {}).get('low') if visit.get('faceResponse') else None,
                                'gender': visit.get('faceResponse', {}).get('gender', {}).get('value') if visit.get('faceResponse') else None,
                                'similarity': first_event.get('confidence', visit.get('similarity', visit.get('confidence', 0.0))) if first_event else visit.get('similarity', visit.get('confidence', 0.0)),
                                'entryEventIds': entry_events,
                                'exitEventIds': exit_events
                            }
                            
                            page_transformed_visits.append(transformed_visit)
                
                # Add valid visits from this page
                all_visits.extend(page_transformed_visits)
                total_fetched += len(page_transformed_visits)
                
                self.logger.info(f"Page {current_page}: {len(page_transformed_visits)} valid visits with images (total so far: {total_fetched})")
                
                # Progress tracking and memory management
                if current_page % 10 == 0:
                    self.logger.info(f"📊 Progress: {total_fetched} visits processed, continuing from page {current_page + 1}")
                    # Clean up memory every 10 pages to prevent memory issues
                    self.cleanup_memory()
                
                # Check if we have enough visits
                if max_visits and total_fetched >= max_visits:
                    self.logger.info(f"Reached target of {max_visits} visits, stopping pagination")
                    break
                
                # Check if we've reached the end of available data
                # The API might return fewer records than requested on the last page
                if page_count < limit:
                    self.logger.info(f"Received {page_count} records (less than limit {limit}), reached end of data")
                    break
                
                # Additional check: if the API response indicates no more data
                if isinstance(data, dict):
                    # Check if API provides total count and we've reached it
                    api_total = data.get('total', 0)
                    if api_total > 0 and total_fetched >= api_total:
                        self.logger.info(f"Reached API total limit of {api_total} visits")
                        break
                
                # Move to next page
                current_page += 1
                
                # Safety check to prevent infinite loops
                if current_page > 100:  # Reasonable limit
                    self.logger.warning("Reached maximum page limit (100), stopping pagination")
                    break
            
            # Trim to exact max_visits if specified
            if max_visits and len(all_visits) > max_visits:
                all_visits = all_visits[:max_visits]
                self.logger.info(f"Trimmed to exactly {max_visits} visits")
            
            self.logger.info(f"Final result: {len(all_visits)} visits with valid image URLs from {current_page - page} pages")
            
            # Store full JSON response in reference file
            self._store_full_json_response(all_visits, api_url, start_date, end_date, page, limit)
            
            # Close session to free resources
            session.close()
            
            return all_visits
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from API: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching visit data from API: {e}")
            return []
    
    def _store_full_json_response(self, visits_data: List[Dict], api_url: str, start_date: str, end_date: str, page: int, limit: int):
        """
        Store the full JSON response in a reference file similar to visit-cluster.json format
        
        Args:
            visits_data: List of processed visit data
            api_url: API URL that was called
            start_date: Start date used in the request
            end_date: End date used in the request (not used in simplified API)
            page: Page number used
            limit: Limit per page used
        """
        try:
            from datetime import datetime
            
            # Create the reference data structure similar to visit-cluster.json
            # Store only the parameters that are actually used in the simplified API call
            reference_data = {
                "metadata": {
                    "api_url": api_url,
                    "date": start_date,  # Use 'date' to match the API parameter name
                    "page": page,
                    "limit": limit,
                    "allBranch": "false",  # Always false in our implementation
                    "nolimit": "false",
                    "isZone": "false",
                    "BlackListed": "false",
                    "Vip": "false",
                    "Vendor": "false",
                    "isDeleted": "false",
                    "total_visits": len(visits_data),
                    "generated_at": datetime.now().isoformat(),
                    "generated_by": "smart_face_recognition_system"
                },
                "total": len(visits_data),
                "visits": visits_data
            }
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visit-cluster-reference_{timestamp}.json"
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(reference_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Full JSON response stored in reference file: {filename}")
            self.logger.info(f"Reference file contains {len(visits_data)} visits")
            
        except Exception as e:
            self.logger.error(f"Error storing full JSON response: {e}")
    
    def analyze_api_data_completeness(self, api_url: str, start_date: str = None, end_date: str = None, 
                                    page: int = 0, limit: int = 100, start_time: str = None, 
                                    end_time: str = None, all_branch: bool = True, selected_branch: str = 'EM-DEPT', 
                                    api_key: str = None, auth_token: str = None, max_visits: int = None) -> Dict:
        """
        Analyze API data completeness to identify missing data
        
        Args:
            api_url: Base API URL
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            page: Page number for pagination (starting from 0)
            limit: Number of records per page
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            all_branch: Whether to include all branches
            selected_branch: Specific branch to filter by
            api_key: API key for authentication (optional)
            auth_token: Bearer token for authentication (optional)
            max_visits: Maximum number of visits to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            analysis_result = {
                "summary": {
                    "total_pages_analyzed": 0,
                    "total_visits_found": 0,
                    "total_visits_with_images": 0,
                    "total_visits_missing_images": 0,
                    "missing_data_percentage": 0.0,
                    "api_errors": [],
                    "pagination_issues": [],
                    "data_quality_issues": []
                },
                "detailed_analysis": {
                    "page_by_page": [],
                    "missing_image_urls": [],
                    "duplicate_visits": [],
                    "invalid_timestamps": [],
                    "empty_events": []
                },
                "recommendations": []
            }
            
            all_visits = []
            current_page = page
            total_fetched = 0
            seen_visit_ids = set()
            seen_image_urls = set()
            
            # Create session for connection reuse with headers that match the working curl command
            session = requests.Session()
            session.headers.update({
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9,en-IN;q=0.8',
                'Connection': 'keep-alive',
                'DNT': '1',
                'Origin': 'https://analytics.develop.thefusionapps.com',
                'Referer': 'https://analytics.develop.thefusionapps.com/',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
                'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Mobile Safari/537.36 Edg/141.0.0.0',
                'sec-ch-ua': '"Microsoft Edge";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
                'sec-ch-ua-mobile': '?1',
                'sec-ch-ua-platform': '"Android"',
                'Keep-Alive': 'timeout=60, max=1000'
            })
            
            self.logger.info(f"Starting data completeness analysis for {max_visits or 'all'} visits")
            
            while True:
                # Build query parameters to match the working API call exactly
                params = {
                    'date': start_date,
                    'page': current_page,
                    'limit': limit,
                    'allBranch': 'false' if not all_branch else 'true',
                    'nolimit': 'false',
                    'isZone': 'false',
                    'BlackListed': 'false',
                    'Vip': 'false',
                    'Vendor': 'false',
                    'isDeleted': 'false'
                }
                
                # Add authentication headers to the existing session headers
                if api_key:
                    session.headers['X-API-Key'] = api_key
                if auth_token:
                    session.headers['Authorization'] = f'Bearer {auth_token}'
                
                # Add the If-None-Match header from the working curl command
                session.headers['If-None-Match'] = 'W/"1798fd-vds5yT1u5VoZqudXPZh9yPz5D9M"'
                
                try:
                    response = session.get(api_url, params=params, timeout=60)
                    
                    # Handle different HTTP status codes
                    if response.status_code == 401:
                        analysis_result["summary"]["api_errors"].append(f"Page {current_page}: 401 Unauthorized")
                        break
                    elif response.status_code == 403:
                        analysis_result["summary"]["api_errors"].append(f"Page {current_page}: 403 Forbidden")
                        break
                    elif response.status_code == 404:
                        analysis_result["summary"]["api_errors"].append(f"Page {current_page}: 404 Not Found")
                        break
                    elif response.status_code == 429:
                        analysis_result["summary"]["api_errors"].append(f"Page {current_page}: 429 Rate Limited")
                        break
                    elif not response.ok:
                        analysis_result["summary"]["api_errors"].append(f"Page {current_page}: {response.status_code} {response.reason}")
                        break
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Parse response data
                    page_visits = []
                    if isinstance(data, list):
                        page_visits = data
                    elif isinstance(data, dict):
                        possible_keys = ['list', 'data', 'visits', 'results', 'items', 'records', 'content']
                        for key in possible_keys:
                            if key in data and isinstance(data[key], list):
                                page_visits = data[key]
                                break
                        
                        if not page_visits and 'id' in data:
                            page_visits = [data]
                    
                    page_count = len(page_visits)
                    page_analysis = {
                        "page_number": current_page,
                        "visits_received": page_count,
                        "visits_with_images": 0,
                        "visits_missing_images": 0,
                        "duplicate_visits": 0,
                        "invalid_timestamps": 0,
                        "empty_events": 0,
                        "api_total": data.get('total', 0) if isinstance(data, dict) else 0
                    }
                    
                    # Analyze each visit in this page
                    for visit in page_visits:
                        visit_id = visit.get('id', visit.get('visitId', visit.get('visit_id')))
                        
                        # Check for duplicates
                        if visit_id in seen_visit_ids:
                            page_analysis["duplicate_visits"] += 1
                            analysis_result["detailed_analysis"]["duplicate_visits"].append({
                                "visit_id": visit_id,
                                "page": current_page
                            })
                        else:
                            seen_visit_ids.add(visit_id)
                        
                        # Check for valid timestamps
                        entry_time = visit.get('timestamp', visit.get('entryTime', visit.get('entry_time')))
                        if not entry_time or entry_time == 'null' or entry_time == '':
                            page_analysis["invalid_timestamps"] += 1
                            analysis_result["detailed_analysis"]["invalid_timestamps"].append({
                                "visit_id": visit_id,
                                "page": current_page
                            })
                        
                        # Check for images in entry events
                        entry_events = visit.get('entryEventIds', [])
                        has_image = False
                        
                        if not entry_events or len(entry_events) == 0:
                            page_analysis["empty_events"] += 1
                            analysis_result["detailed_analysis"]["empty_events"].append({
                                "visit_id": visit_id,
                                "page": current_page
                            })
                        else:
                            for event in entry_events:
                                if isinstance(event, dict) and event.get('fileName'):
                                    image_url = f"https://cdn.analytics.thefusionapps.com/v11/{event['fileName']}"
                                    if image_url not in seen_image_urls:
                                        seen_image_urls.add(image_url)
                                        has_image = True
                                        break
                        
                        if has_image:
                            page_analysis["visits_with_images"] += 1
                        else:
                            page_analysis["visits_missing_images"] += 1
                            analysis_result["detailed_analysis"]["missing_image_urls"].append({
                                "visit_id": visit_id,
                                "page": current_page,
                                "entry_events_count": len(entry_events) if entry_events else 0
                            })
                    
                    # Add page analysis to results
                    analysis_result["detailed_analysis"]["page_by_page"].append(page_analysis)
                    
                    # Update summary
                    analysis_result["summary"]["total_pages_analyzed"] += 1
                    analysis_result["summary"]["total_visits_found"] += page_count
                    analysis_result["summary"]["total_visits_with_images"] += page_analysis["visits_with_images"]
                    analysis_result["summary"]["total_visits_missing_images"] += page_analysis["visits_missing_images"]
                    
                    # Check if we've reached the end
                    if page_count == 0:
                        analysis_result["summary"]["pagination_issues"].append(f"No data on page {current_page}")
                        break
                    
                    if page_count < limit:
                        analysis_result["summary"]["pagination_issues"].append(f"Page {current_page} returned {page_count} records (less than limit {limit})")
                        break
                    
                    # Check if we have enough visits
                    if max_visits and analysis_result["summary"]["total_visits_found"] >= max_visits:
                        break
                    
                    current_page += 1
                    
                    # Safety check
                    if current_page > 100:
                        analysis_result["summary"]["pagination_issues"].append("Reached maximum page limit (100)")
                        break
                        
                except Exception as e:
                    analysis_result["summary"]["api_errors"].append(f"Page {current_page}: {str(e)}")
                    break
            
            # Calculate missing data percentage
            total_visits = analysis_result["summary"]["total_visits_found"]
            if total_visits > 0:
                missing_count = analysis_result["summary"]["total_visits_missing_images"]
                analysis_result["summary"]["missing_data_percentage"] = (missing_count / total_visits) * 100
            
            # Generate recommendations
            if analysis_result["summary"]["missing_data_percentage"] > 50:
                analysis_result["recommendations"].append("High missing data percentage - check API parameters and authentication")
            
            if len(analysis_result["summary"]["api_errors"]) > 0:
                analysis_result["recommendations"].append("API errors detected - verify credentials and endpoint")
            
            if len(analysis_result["detailed_analysis"]["duplicate_visits"]) > 0:
                analysis_result["recommendations"].append("Duplicate visits found - consider adding deduplication logic")
            
            if len(analysis_result["detailed_analysis"]["empty_events"]) > 0:
                analysis_result["recommendations"].append("Empty events found - check data quality at source")
            
            # Close session
            session.close()
            
            self.logger.info(f"Data completeness analysis completed: {total_visits} visits analyzed, {analysis_result['summary']['missing_data_percentage']:.1f}% missing images")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing API data completeness: {e}")
            return {
                "error": str(e),
                "summary": {"total_visits_found": 0, "missing_data_percentage": 100.0}
            }
    
    def compare_face_images(self, image1_url: str, image2_url: str) -> Dict:
        """
        Compare two face images and return comparison result
        
        Args:
            image1_url: URL of first image
            image2_url: URL of second image
            
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info(f"Comparing faces: {image1_url} vs {image2_url}")
            
            # Download both images
            img1 = self.download_image_from_url(image1_url)
            img2 = self.download_image_from_url(image2_url)
            
            if img1 is None or img2 is None:
                return {
                    'same_person': False,
                    'confidence': 0.0,
                    'error': 'Could not download one or both images',
                    'image1_url': image1_url,
                    'image2_url': image2_url
                }
            
            # Workaround for InsightFace library issue - detect faces directly
            try:
                # Convert BGR to RGB for InsightFace
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # Check if model is initialized
                if self.app is None:
                    self.logger.error("Face recognition model not initialized")
                    return None
                
                # Check if model is initialized
                if not hasattr(self, 'app') or self.app is None:
                    self.logger.error("Face recognition model not initialized")
                    return None
                
                # Detect faces in both images (use max_num for better detection)
                faces1 = self.app.get(img1_rgb, max_num=10)
                faces2 = self.app.get(img2_rgb, max_num=10)
                
                if len(faces1) == 0 or len(faces2) == 0:
                    return {
                        'same_person': False,
                        'confidence': 0.0,
                        'error': 'Could not detect faces in one or both images',
                        'image1_url': image1_url,
                        'image2_url': image2_url
                    }
                
                # Get face embeddings directly from detected faces
                face1_embedding = faces1[0].embedding
                face2_embedding = faces2[0].embedding
                
                # Calculate similarity
                similarity = self.calculate_face_similarity(face1_embedding, face2_embedding)
                
                # Determine if same person based on threshold
                same_person = similarity > self.similarity_thresh
                
                self.logger.info(f"Face comparison result: {same_person} (confidence: {similarity:.4f}, threshold: {self.similarity_thresh})")
                
                return {
                    'same_person': same_person,
                    'confidence': float(similarity),
                    'threshold_used': self.similarity_thresh,
                    'image1_url': image1_url,
                    'image2_url': image2_url,
                    'error': None
                }
                
            except Exception as face_error:
                self.logger.error(f"Error in face detection/embedding: {face_error}")
                return {
                    'same_person': False,
                    'confidence': 0.0,
                    'error': f'Face detection error: {str(face_error)}',
                    'image1_url': image1_url,
                    'image2_url': image2_url
                }
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {e}")
            return {
                'same_person': False,
                'confidence': 0.0,
                'error': str(e),
                'image1_url': image1_url,
                'image2_url': image2_url
            }
    
    @monitor_performance
    def calculate_face_similarity(self, face1, face2) -> float:
        """
        Calculate similarity between two face embeddings
        
        Args:
            face1: First face embedding
            face2: Second face embedding
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Use multi-metric similarity if enabled, otherwise use cosine similarity
            if self.use_multi_metric_similarity:
                return self.calculate_multi_metric_similarity(face1, face2)
            else:
                # Calculate cosine similarity (backward compatible)
                similarity = np.dot(face1, face2) / (np.linalg.norm(face1) * np.linalg.norm(face2))
                return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating face similarity: {e}")
            return 0.0
    
    def calculate_similarity_vectorized(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Vectorized similarity calculation for multiple face pairs
        
        Args:
            embeddings1: First set of face embeddings (N x D)
            embeddings2: Second set of face embeddings (M x D)
            
        Returns:
            Similarity matrix (N x M) with cosine similarities
        """
        try:
            # Normalize embeddings for cosine similarity
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # Avoid division by zero
            norm1 = np.where(norm1 == 0, 1, norm1)
            norm2 = np.where(norm2 == 0, 1, norm2)
            
            # Normalized embeddings
            normalized1 = embeddings1 / norm1
            normalized2 = embeddings2 / norm2
            
            # Vectorized cosine similarity calculation
            similarity_matrix = np.dot(normalized1, normalized2.T)
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error in vectorized similarity calculation: {e}")
            return np.zeros((len(embeddings1), len(embeddings2)))
    
    def find_best_matches_vectorized(self, query_embeddings: np.ndarray, 
                                   candidate_embeddings: np.ndarray, 
                                   candidate_ids: List[int],
                                   k: int = 5) -> List[List[Dict]]:
        """
        Find best matches for multiple queries using vectorized operations
        
        Args:
            query_embeddings: Query face embeddings (N x D)
            candidate_embeddings: Candidate face embeddings (M x D)
            candidate_ids: List of candidate person IDs
            k: Number of top matches to return per query
            
        Returns:
            List of best matches for each query
        """
        try:
            # Calculate similarity matrix
            similarity_matrix = self.calculate_similarity_vectorized(query_embeddings, candidate_embeddings)
            
            # Find top k matches for each query
            results = []
            for i in range(len(query_embeddings)):
                # Get similarities for this query
                similarities = similarity_matrix[i]
                
                # Get top k indices
                top_k_indices = np.argsort(similarities)[-k:][::-1]
                
                # Create result for this query
                query_results = []
                for idx in top_k_indices:
                    if similarities[idx] > self.similarity_thresh:
                        query_results.append({
                            'person_id': candidate_ids[idx],
                            'similarity': float(similarities[idx]),
                            'name': f"Person_{candidate_ids[idx]}"  # You might want to get actual names
                        })
                
                results.append(query_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in vectorized matching: {e}")
            return [[] for _ in range(len(query_embeddings))]
    
    @monitor_performance
    def calculate_multi_metric_similarity(self, face1_embedding: np.ndarray, face2_embedding: np.ndarray) -> float:
        """
        Calculate similarity using multiple distance metrics with weighted combination
        
        Args:
            face1_embedding: First face embedding
            face2_embedding: Second face embedding
            
        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        try:
            # Cosine similarity (existing logic)
            cosine_sim = np.dot(face1_embedding, face2_embedding) / (np.linalg.norm(face1_embedding) * np.linalg.norm(face2_embedding))
            
            # Euclidean distance similarity
            euclidean_dist = np.linalg.norm(face1_embedding - face2_embedding)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Manhattan distance similarity
            manhattan_dist = np.sum(np.abs(face1_embedding - face2_embedding))
            manhattan_sim = 1 / (1 + manhattan_dist)
            
            # Pearson correlation
            pearson_corr = np.corrcoef(face1_embedding, face2_embedding)[0, 1]
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
            
            # Apply weighted combination using config weights
            combined_similarity = (self.similarity_weights['cosine'] * cosine_sim + 
                                 self.similarity_weights['euclidean'] * euclidean_sim + 
                                 self.similarity_weights['manhattan'] * manhattan_sim + 
                                 self.similarity_weights['pearson'] * pearson_corr)
            
            return float(combined_similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-metric similarity: {e}")
            return 0.0
    
    def calculate_multi_metric_similarity_vectorized(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Vectorized multi-metric similarity calculation for multiple face pairs
        
        Args:
            embeddings1: First set of face embeddings (N x D)
            embeddings2: Second set of face embeddings (M x D)
            
        Returns:
            Similarity matrix (N x M) with combined similarities
        """
        try:
            # Normalize embeddings for cosine similarity
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            
            # Avoid division by zero
            norm1 = np.where(norm1 == 0, 1, norm1)
            norm2 = np.where(norm2 == 0, 1, norm2)
            
            # Cosine similarity matrix
            cosine_sim = np.dot(embeddings1, embeddings2.T) / (norm1 * norm2.T)
            
            # Euclidean distance similarity matrix
            # Reshape for broadcasting: (N, 1, D) - (1, M, D) = (N, M, D)
            diff = embeddings1[:, np.newaxis, :] - embeddings2[np.newaxis, :, :]
            euclidean_dist = np.linalg.norm(diff, axis=2)
            euclidean_sim = 1 / (1 + euclidean_dist)
            
            # Manhattan distance similarity matrix
            manhattan_dist = np.sum(np.abs(diff), axis=2)
            manhattan_sim = 1 / (1 + manhattan_dist)
            
            # Pearson correlation matrix (more complex for vectorized operation)
            # For now, use a simplified version or fall back to cosine for large matrices
            if len(embeddings1) * len(embeddings2) > 10000:  # Large matrix, use cosine as approximation
                pearson_sim = cosine_sim
            else:
                # Calculate Pearson correlation for smaller matrices
                pearson_sim = np.zeros_like(cosine_sim)
                for i in range(len(embeddings1)):
                    for j in range(len(embeddings2)):
                        corr = np.corrcoef(embeddings1[i], embeddings2[j])[0, 1]
                        pearson_sim[i, j] = 0.0 if np.isnan(corr) else corr
            
            # Apply weighted combination
            combined_similarity = (self.similarity_weights['cosine'] * cosine_sim + 
                                 self.similarity_weights['euclidean'] * euclidean_sim + 
                                 self.similarity_weights['manhattan'] * manhattan_sim + 
                                 self.similarity_weights['pearson'] * pearson_sim)
            
            return combined_similarity
            
        except Exception as e:
            self.logger.error(f"Error in vectorized multi-metric similarity: {e}")
            return np.zeros((len(embeddings1), len(embeddings2)))
    
    def extract_face_embedding(self, image) -> np.ndarray:
        """
        Extract face embedding from image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Face embedding vector or None if no face detected
        """
        try:
            # Ensure image is in the correct format for InsightFace
            if not isinstance(image, np.ndarray):
                self.logger.error(f"Invalid image format: {type(image)}")
                return None
            
            # Convert BGR to RGB if needed (InsightFace expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Check if model is initialized, try to reinitialize if not
            if self.app is None:
                self.logger.warning("Face recognition model not initialized, attempting to reinitialize...")
                try:
                    self.initialize_model()
                except Exception as e:
                    self.logger.error(f"Failed to reinitialize model: {e}")
                    return None
            
            # Detect faces in the image with GPU error handling
            try:
                # Check if model is initialized
                if not hasattr(self, 'app') or self.app is None:
                    self.logger.error("Face recognition model not initialized")
                    return None
                
                faces = self.app.get(image_rgb, max_num=10)
            except Exception as gpu_error:
                if "CUDNN" in str(gpu_error) or "CUDA" in str(gpu_error) or "ONNXRuntimeError" in str(gpu_error):
                    self.logger.warning(f"GPU error detected: {gpu_error}")
                    self.logger.warning("Attempting to reinitialize model on CPU...")
                    
                    # Try to reinitialize on CPU
                    try:
                        self.app = FaceAnalysis(name=self.model_name)
                        det_size = tuple(self.config['face_detection']['detection_size'])
                        self.app.prepare(ctx_id=-1, det_size=det_size)  # -1 for CPU
                        self.gpu_id = -1
                        self.logger.info("Model reinitialized on CPU, retrying face detection...")
                        
                        # Retry face detection on CPU
                        faces = self.app.get(image_rgb, max_num=10)
                    except Exception as cpu_error:
                        self.logger.error(f"CPU fallback also failed: {cpu_error}")
                        return None
                else:
                    # Re-raise non-GPU errors
                    raise gpu_error
            
            if len(faces) == 0:
                self.logger.warning("No faces detected in image")
                return None
            
            # Get the first (largest) face
            face = faces[0]
            
            # Return the embedding
            return face.embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def extract_face_embedding_with_choronological(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using ensemble if enabled, otherwise use single model
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Face embedding vector or None
        """
        # Use ensemble if enabled
        if self.use_choronological and self.ensemble_handler is not None:
            return self.ensemble_handler.extract_ensemble_embedding(image)
        
        # Fall back to regular extraction
        return self.extract_face_embedding(image)

    def process_face_comparisons(self, comparison_records: List[Dict], max_comparisons: int = None) -> Dict:
        """
        Process face comparisons from API data
        
        Args:
            comparison_records: List of comparison records from API
            max_comparisons: Maximum number of comparisons to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not comparison_records:
                return {
                    'total_comparisons': 0,
                    'processed': 0,
                    'same_person': 0,
                    'different_person': 0,
                    'errors': 0,
                    'results': []
                }
            
            # Limit comparisons if specified
            if max_comparisons and len(comparison_records) > max_comparisons:
                comparison_records = comparison_records[:max_comparisons]
            
            self.logger.info(f"Processing {len(comparison_records)} face comparisons")
            
            results = []
            same_person_count = 0
            different_person_count = 0
            error_count = 0
            
            for i, record in enumerate(comparison_records):
                try:
                    self.logger.info(f"Processing comparison {i+1}/{len(comparison_records)}: {record['comparison_id']}")
                    
                    # Compare the two faces
                    comparison_result = self.compare_face_images(
                        record['image1_url'], 
                        record['image2_url']
                    )
                    
                    # Add metadata to result
                    result = {
                        'comparison_id': record['comparison_id'],
                        'event_id': record['event_id'],
                        'branch_id': record['branch_id'],
                        'created_at': record['created_at'],
                        'customer_info': record['customer_info'],
                        'matched_info': record['matched_info'],
                        'api_approve': record['approve'],
                        'our_result': comparison_result['same_person'],
                        'confidence': comparison_result['confidence'],
                        'threshold_used': comparison_result.get('threshold_used', self.similarity_thresh),
                        'image1_url': comparison_result['image1_url'],
                        'image2_url': comparison_result['image2_url'],
                        'error': comparison_result['error'],
                        'match_status': 'SAME' if comparison_result['same_person'] else 'DIFFERENT',
                        'api_vs_our_match': record['approve'] == comparison_result['same_person'],
                        'raw_data': record.get('raw_data', {})
                    }
                    
                    results.append(result)
                    
                    # Update counters
                    if comparison_result['error']:
                        error_count += 1
                    elif comparison_result['same_person']:
                        same_person_count += 1
                    else:
                        different_person_count += 1
                    
                    self.logger.info(f"Comparison {i+1} result: {result['match_status']} (confidence: {comparison_result['confidence']:.4f})")
                    
                except Exception as e:
                    self.logger.error(f"Error processing comparison {record['comparison_id']}: {e}")
                    error_count += 1
                    results.append({
                        'comparison_id': record['comparison_id'],
                        'error': str(e),
                        'match_status': 'ERROR'
                    })
            
            # Calculate accuracy if we have API approval data
            api_matches = sum(1 for r in results if r.get('api_vs_our_match') is True)
            total_with_api_data = sum(1 for r in results if 'api_vs_our_match' in r and r['api_vs_our_match'] is not None)
            accuracy = (api_matches / total_with_api_data * 100) if total_with_api_data > 0 else 0
            
            processing_results = {
                'total_comparisons': len(comparison_records),
                'processed': len(results),
                'same_person': same_person_count,
                'different_person': different_person_count,
                'errors': error_count,
                'accuracy_vs_api': accuracy,
                'api_matches': api_matches,
                'total_with_api_data': total_with_api_data,
                'results': results
            }
            
            self.logger.info(f"Face comparison processing completed:")
            self.logger.info(f"  Total: {processing_results['total_comparisons']}")
            self.logger.info(f"  Same Person: {processing_results['same_person']}")
            self.logger.info(f"  Different Person: {processing_results['different_person']}")
            self.logger.info(f"  Errors: {processing_results['errors']}")
            self.logger.info(f"  Accuracy vs API: {processing_results['accuracy_vs_api']:.2f}%")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Error processing face comparisons: {e}")
            return {
                'total_comparisons': 0,
                'processed': 0,
                'same_person': 0,
                'different_person': 0,
                'errors': 1,
                'error_message': str(e),
                'results': []
            }
    
    def assess_face_quality(self, face) -> Dict[str, float]:
        """
        Assess face quality to prevent false positives and mismatches
        
        Args:
            face: InsightFace face object
            
        Returns:
            Dictionary with quality scores
        """
        quality_scores = {
            'overall': 0.0,
            'blur': 0.0,
            'pose': 0.0,
            'lighting': 0.0,
            'size': 0.0
        }
        
        try:
            # Detection confidence
            det_score = getattr(face, 'det_score', 0.0)
            
            # Face size assessment
            bbox = face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            
            # Size score (prefer larger faces)
            size_normalization = self.config['face_quality']['size_normalization']
            size_score = min(1.0, face_area / size_normalization)
            
            # Blur assessment (simplified)
            blur_score = min(1.0, det_score * 1.2)  # Higher detection score = less blur
            
            # Pose assessment (using keypoints if available)
            pose_score = 1.0
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps
                # Check if keypoints are well distributed
                if len(kps) >= 5:
                    # Simple pose assessment based on keypoint distribution
                    x_coords = kps[:, 0]
                    y_coords = kps[:, 1]
                    x_range = np.max(x_coords) - np.min(x_coords)
                    y_range = np.max(y_coords) - np.min(y_coords)
                    pose_score = min(1.0, (x_range + y_range) / 100)
            
            # Lighting assessment (simplified)
            lighting_score = min(1.0, det_score * 1.1)
            
            # Overall quality score using config weights
            weights = self.config['face_quality']['weights']
            overall_score = (det_score * weights['detection_score'] + 
                           size_score * weights['size_score'] + 
                           blur_score * weights['blur_score'] + 
                           pose_score * weights['pose_score'] + 
                           lighting_score * weights['lighting_score'])
            
            quality_scores = {
                'overall': float(overall_score),
                'blur': float(blur_score),
                'pose': float(pose_score),
                'lighting': float(lighting_score),
                'size': float(size_score)
            }
            
        except Exception as e:
            self.logger.warning(f"Error assessing face quality: {e}")
            quality_scores['overall'] = self.config['face_quality']['min_overall_score']
        
        return quality_scores
    
    def get_face_pose_angles(self, face) -> Dict[str, float]:
        """
        Extract face pose angles from InsightFace detection
        
        Args:
            face: InsightFace face object
            
        Returns:
            Dict with yaw, pitch, roll angles in degrees
        """
        try:
            # InsightFace provides pose information in radians
            yaw = getattr(face, 'yaw', 0)  # Left/right rotation
            pitch = getattr(face, 'pitch', 0)  # Up/down rotation
            roll = getattr(face, 'roll', 0)  # Rotation around face axis
            
            # Convert from radians to degrees
            yaw_deg = math.degrees(yaw) if yaw else 0
            pitch_deg = math.degrees(pitch) if pitch else 0
            roll_deg = math.degrees(roll) if roll else 0
            
            return {
                'yaw': yaw_deg,      # -90 to +90 (left to right)
                'pitch': pitch_deg,  # -90 to +90 (up to down)  
                'roll': roll_deg     # -180 to +180 (rotation)
            }
        except Exception as e:
            self.logger.warning(f"Error extracting pose angles: {e}")
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
    
    def is_side_face(self, face) -> bool:
        """
        Check if face is too side-facing to be useful for recognition
        Uses both pose angles and advanced bbox analysis
        
        Args:
            face: InsightFace face object
            
        Returns:
            True if face should be rejected (side-facing)
        """
        try:
            # Check if side face rejection is disabled in config
            if self.config['side_face_detection'].get('disable_side_face_rejection', False):
                self.logger.debug("Side face rejection is disabled - accepting all faces")
                return False
            
            # Method 1: Try pose angles if available
            pose_angles = self.get_face_pose_angles(face)
            yaw_angle = abs(pose_angles.get('yaw', 0))
            pitch_angle = abs(pose_angles.get('pitch', 0))
            
            if yaw_angle > 0 or pitch_angle > 0:  # If pose data is available
                yaw_threshold = self.config['face_detection']['yaw_threshold']
                pitch_threshold = self.config['face_detection']['pitch_threshold']
                if yaw_angle > yaw_threshold:
                    self.logger.info(f"Side face detected: yaw={yaw_angle:.1f}° (threshold: {yaw_threshold}°)")
                    return True
                if pitch_angle > pitch_threshold:
                    self.logger.info(f"Extreme angle detected: pitch={pitch_angle:.1f}° (threshold: {pitch_threshold}°)")
                    return True
                return False
            
            # Method 2: Advanced bbox analysis using InsightFace bbox
            bbox = getattr(face, 'bbox', None)
            det_score = getattr(face, 'det_score', 0.0)
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox_data = {
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'top': y1,
                    'left': x1
                }
                
                is_side, reason, score = self.analyze_bbox_for_side_face(bbox_data, det_score)
                if is_side:
                    self.logger.info(f"Side face detected: {reason} (score: {score})")
                    return True
                
            return False
        except Exception as e:
            self.logger.warning(f"Error checking side face: {e}")
            return False
    
    def analyze_bbox_for_side_face(self, bbox_data, det_score=None):
        """
        Analyze bbox data from JSON to detect side faces using advanced methods
        Based on research on side face detection techniques
        
        Args:
            bbox_data: Dictionary with 'width', 'height', 'top', 'left' keys
            det_score: Detection confidence score (optional)
            
        Returns:
            Tuple of (is_side_face, reason, side_face_score)
        """
        if not bbox_data:
            return False, "No bbox data", 0
        
        width = bbox_data.get('width', 0)
        height = bbox_data.get('height', 0)
        top = bbox_data.get('top', 0)
        left = bbox_data.get('left', 0)
        
        if width <= 0 or height <= 0:
            return False, "Invalid bbox dimensions", 0
        
        # Calculate metrics
        aspect_ratio = width / height
        area = width * height
        perimeter = 2 * (width + height)
        compactness = (4 * 3.14159 * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Advanced scoring system based on research
        side_face_score = 0
        reasons = []
        
        # 1. Aspect ratio analysis using config thresholds
        aspect_thresholds = self.config['side_face_detection']['aspect_ratio_thresholds']
        if aspect_ratio < aspect_thresholds['extreme_profile']:
            side_face_score += 4
            reasons.append(f"Extreme profile (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio < aspect_thresholds['very_strong_profile']:
            side_face_score += 3
            reasons.append(f"Very strong profile (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio < aspect_thresholds['strong_profile']:
            side_face_score += 2
            reasons.append(f"Strong profile (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > aspect_thresholds['very_wide']:
            side_face_score += 3
            reasons.append(f"Very wide face (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > aspect_thresholds['wide']:
            side_face_score += 2
            reasons.append(f"Wide face (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > aspect_thresholds['moderately_wide']:
            side_face_score += 1
            reasons.append(f"Moderately wide (ratio: {aspect_ratio:.2f})")
        
        # 2. Size analysis using config thresholds
        area_thresholds = self.config['side_face_detection']['area_thresholds']
        if area < area_thresholds['extremely_small']:
            side_face_score += 3
            reasons.append(f"Extremely small area: {area}")
        elif area < area_thresholds['very_small']:
            side_face_score += 2
            reasons.append(f"Very small area: {area}")
        elif area < area_thresholds['small']:
            side_face_score += 1
            reasons.append(f"Small area: {area}")
        elif area > area_thresholds['very_large']:
            side_face_score += 2
            reasons.append(f"Very large area: {area}")
        elif area > area_thresholds['large']:
            side_face_score += 1
            reasons.append(f"Large area: {area}")
        
        # 3. Compactness analysis using config thresholds
        compactness_thresholds = self.config['side_face_detection']['compactness_thresholds']
        if compactness < compactness_thresholds['very_low']:
            side_face_score += 2
            reasons.append(f"Very low compactness: {compactness:.2f}")
        elif compactness < compactness_thresholds['low']:
            side_face_score += 1
            reasons.append(f"Low compactness: {compactness:.2f}")
        
        # 4. Detection confidence analysis using config thresholds
        confidence_thresholds = self.config['side_face_detection']['confidence_thresholds']
        if det_score and det_score < confidence_thresholds['very_low']:
            side_face_score += 2
            reasons.append(f"Very low confidence: {det_score:.3f}")
        elif det_score and det_score < confidence_thresholds['low']:
            side_face_score += 1
            reasons.append(f"Low confidence: {det_score:.3f}")
        
        # 5. Position analysis (faces at edges might be side views)
        edge_threshold = self.config['side_face_detection']['edge_position_threshold']
        if left < edge_threshold or top < edge_threshold:
            side_face_score += 1
            reasons.append(f"Face very near edge (left: {left}, top: {top})")
        
        # Decision threshold using config
        decision_threshold = self.config['side_face_detection']['decision_threshold']
        is_side_face = side_face_score >= decision_threshold
        reason = "; ".join(reasons) if reasons else "Normal face"
        
        return is_side_face, reason, side_face_score
    
    def check_side_face_from_json_bbox(self, visit_data) -> tuple:
        """
        Check for side faces using bbox data from JSON before processing
        
        Args:
            visit_data: Visit data dictionary from JSON
            
        Returns:
            Tuple of (is_side_face, reason, bbox_data)
        """
        try:
            # Extract bbox data from entryEventIds
            entry_events = visit_data.get('entryEventIds', [])
            if not entry_events:
                return False, "No entry events", None
            
            # Use the first entry event's bbox
            first_event = entry_events[0]
            bbox_data = first_event.get('box', {})
            
            if not bbox_data:
                return False, "No bbox data in entry event", None
            
            # Analyze the bbox for side face characteristics
            is_side, reason, score = self.analyze_bbox_for_side_face(bbox_data)
            
            return is_side, reason, bbox_data
            
        except Exception as e:
            self.logger.warning(f"Error checking side face from JSON bbox: {e}")
            return False, f"Error: {e}", None
    
    def extract_face_embedding(self, image_source: str, save_image: bool = False, output_dir: str = None) -> Optional[Dict]:
        """
        Extract face embedding from image with quality assessment
        
        Args:
            image_source: Path to local image file or URL to image
            save_image: Whether to save downloaded images locally
            output_dir: Directory to save images (if save_image is True)
            
        Returns:
            Dictionary with embedding and quality info, or None if no face found
        """
        try:
            # Determine if it's a URL or local file path
            if image_source.startswith('http'):
                # Generate save path if needed
                save_path = None
                if save_image and output_dir:
                    # Extract filename from URL
                    url_parts = image_source.split('/')
                    filename = url_parts[-1] if url_parts else f"image_{int(time.time())}.jpg"
                    # Ensure it has proper extension
                    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        filename += '.jpg'
                    save_path = os.path.join(output_dir, filename)
                
                # Download image from URL
                image = self.download_image_from_url(image_source, save_path=save_path)
                if image is None:
                    self.logger.warning(f"Could not download image from URL: {image_source}")
                    return None
            else:
                # Load local image file
                image = cv2.imread(image_source)
                if image is None:
                    self.logger.warning(f"Could not load local image: {image_source}")
                    return None
            
            # Check if model is initialized, try to reinitialize if not
            if self.app is None:
                self.logger.warning("Face recognition model not initialized, attempting to reinitialize...")
                try:
                    with self.model_lock:
                        self.initialize_model()
                except Exception as e:
                    self.logger.error(f"Failed to reinitialize model: {e}")
                    return None
            
            # Detect faces (with thread lock for safety - InsightFace is not thread-safe)
            with self.model_lock:
                # Check if app exists before using it
                if not hasattr(self, 'app') or self.app is None:
                    self.logger.warning("Model not initialized, attempting to reinitialize...")
                    self.initialize_model()
                
                if not hasattr(self, 'app') or self.app is None:
                    self.logger.error("Failed to initialize model")
                    return None
                
                # Use max_num=10 to detect more faces (better for side faces and profiles)
                faces = self.app.get(image, max_num=10)
            if not faces:
                self.logger.warning(f"No faces detected in: {image_source}")
                return None
            
            # Select best face
            best_face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
            
            # Check confidence threshold
            if getattr(best_face, 'det_score', 0.0) < self.confidence_thresh:
                self.logger.warning(f"Face confidence too low in: {image_source}")
                return None
            
            # Check if face is side-facing (reject side faces)
            if self.is_side_face(best_face):
                self.logger.warning(f"Side face rejected in: {image_source}")
                return None
            
            # Get embedding - USE ENSEMBLE IF ENABLED
            if self.use_choronological and self.ensemble_handler is not None:
                # Use ensemble for better accuracy
                self.logger.debug("Using ensemble models for embedding extraction")
                ensemble_embedding = self.ensemble_handler.extract_ensemble_embedding(image)
                if ensemble_embedding is not None:
                    embedding = ensemble_embedding
                    self.logger.debug("Ensemble embedding extracted successfully")
                else:
                    self.logger.warning("Ensemble failed to extract embedding, falling back to single model")
                    # Fall back to single model
                    embedding = getattr(best_face, 'normed_embedding', None)
                    if embedding is None:
                        embedding = getattr(best_face, 'embedding', None)
                        if embedding is not None:
                            embedding = embedding / np.linalg.norm(embedding)
            else:
                # Single model extraction (original code)
                embedding = getattr(best_face, 'normed_embedding', None)
                if embedding is None:
                    embedding = getattr(best_face, 'embedding', None)
                    if embedding is not None:
                        embedding = embedding / np.linalg.norm(embedding)
            
            if embedding is None:
                self.logger.warning(f"Could not extract embedding from: {image_source}")
                return None
            
            # CRITICAL: Normalize embedding before returning (ensures L2 norm = 1)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Assess face quality
            quality_scores = self.assess_face_quality(best_face)
            det_score = getattr(best_face, 'det_score', 0.0)
            
            # Check if we should skip face detection quality checks
            skip_quality = self.config['face_detection'].get('skip_face_detection_quality', False)
            
            if not skip_quality:
                # Adaptive quality threshold based on detection confidence
                base_threshold = self.config['face_detection']['min_quality_threshold']
                if det_score > 0.8:
                    min_quality = base_threshold * 0.6  # More lenient for high confidence
                    self.logger.debug(f"High confidence face (det_score: {det_score:.3f}), using lenient threshold: {min_quality:.3f}")
                elif det_score > 0.6:
                    min_quality = base_threshold  # Standard threshold
                    self.logger.debug(f"Medium confidence face (det_score: {det_score:.3f}), using standard threshold: {min_quality:.3f}")
                else:
                    min_quality = base_threshold * 1.6  # Stricter for low confidence
                    self.logger.debug(f"Low confidence face (det_score: {det_score:.3f}), using strict threshold: {min_quality:.3f}")
                
                # Log detailed quality metrics for analysis
                self.logger.debug(f"Quality scores for {image_source}: {quality_scores}, threshold: {min_quality:.3f}")
                
                if quality_scores['overall'] < min_quality:
                    self.logger.warning(f"Face quality too low in: {image_source} (score: {quality_scores['overall']:.3f} < {min_quality:.3f})")
                    return None
            
            result = {
                'embedding': embedding.astype(np.float32),
                'quality': quality_scores,
                'bbox': best_face.bbox,
                'det_score': getattr(best_face, 'det_score', 0.0),
                'face_confidence': getattr(best_face, 'det_score', 0.0),  # Add face_confidence for easier access
                'face_hash': self.compute_face_hash(embedding),
                'image_source': image_source
            }
            
            # Add saved image path if image was saved
            if save_image and save_path:
                result['saved_image_path'] = save_path
            
            return result
            
        except Exception as e:
            # Check if it's a GPU memory error
            if "CUBLAS_STATUS_ALLOC_FAILED" in str(e) or "CUDA" in str(e):
                self.gpu_memory_error_count += 1
                self.logger.warning(f"GPU memory error processing {image_source}: {e}")
                self.logger.info("Attempting to clear GPU cache and retry...")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except:
                    pass
                
                # If too many GPU memory errors, disable GPU processing
                if self.gpu_memory_error_count > 10:
                    self.gpu_memory_available = False
                    self.logger.warning("Too many GPU memory errors. Switching to CPU processing.")
                
                return None
            else:
                self.logger.error(f"Error processing {image_source}: {e}")
                return None
    
    @monitor_performance
    def add_person(self, name: str, image_source: str, embedding_data: Dict) -> int:
        """
        Add a new person to the database
        
        Args:
            name: Person name
            image_source: Path to person's image or URL
            embedding_data: Face embedding data
            
        Returns:
            Person ID
        """
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Check for duplicate face hash
                cursor.execute('SELECT id FROM persons WHERE face_hash = ?', 
                             (embedding_data['face_hash'],))
                if cursor.fetchone():
                    self.logger.warning(f"Duplicate face detected for: {name}")
                    return -1
                
                # Store person data (without embedding BLOB - now stored in Qdrant)
                quality_score = embedding_data['quality']['overall']
                
                cursor.execute('''
                        INSERT INTO persons (name, image_path, face_quality, face_hash)
                        VALUES (?, ?, ?, ?)
                    ''', (name, image_source, quality_score, embedding_data['face_hash']))
                
                person_id = cursor.lastrowid
                
                # Store detailed quality scores
                cursor.execute('''
                    INSERT INTO face_quality (person_id, quality_score, blur_score, pose_score, lighting_score)
                    VALUES (?, ?, ?, ?, ?)
                ''', (person_id, quality_score, 
                      embedding_data['quality']['blur'],
                      embedding_data['quality']['pose'],
                      embedding_data['quality']['lighting']))
                
                conn.commit()
                
                # Store embedding in Qdrant - NORMALIZE FIRST (critical for cosine similarity)
                embedding_to_store = embedding_data['embedding']
                # Ensure it's normalized (extra safety check)
                embedding_norm = np.linalg.norm(embedding_to_store)
                if embedding_norm > 0:
                    embedding_to_store = embedding_to_store / embedding_norm
                
                success = self.vector_db.add_embedding(
                    person_id=person_id,
                    embedding=embedding_to_store,
                    metadata={
                        'name': name,
                        'quality': quality_score,
                        'image_path': image_source,
                        'face_hash': embedding_data['face_hash']
                    }
                )
                
                if success:
                    self.logger.info(f"Added person: {name} (ID: {person_id}) to both SQLite and Qdrant")
                else:
                    self.logger.error(f"Failed to add embedding to Qdrant for person {person_id}")
                    # Rollback SQLite transaction
                    conn.rollback()
                    return -1
                
                return person_id
                
            except Exception as e:
                self.logger.error(f"Error adding person {name}: {e}")
                conn.rollback()
                return -1
    
    def load_embeddings(self):
        """Initialize vector database connection with restart bug detection"""
        try:
            # Check if Qdrant is working
            embedding_count = self.vector_db.get_embedding_count()
            self.logger.info(f"Vector database initialized with {embedding_count} embeddings")
            
            # Verify embeddings match database - CRITICAL for restart bug detection
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM persons')
                person_count = cursor.fetchone()[0]
            
            self.logger.info(f"Database has {person_count} persons, Qdrant has {embedding_count} embeddings")
            
            # Critical check: if database has persons but no embeddings, this is the restart bug
            if person_count > 0 and embedding_count == 0:
                self.logger.error("🚨 CRITICAL: Database has persons but no embeddings! This is the restart bug.")
                self.logger.error("Qdrant is likely in memory mode and lost data on restart.")
                self.logger.info("Attempting to re-index all embeddings from database...")
                self.reindex_all_embeddings()
            elif person_count != embedding_count:
                self.logger.warning(f"Embedding count mismatch: DB={person_count}, Qdrant={embedding_count}")
                self.logger.info("Attempting to re-index missing embeddings...")
                self.reindex_all_embeddings()
            else:
                self.logger.info("✅ Embedding count matches database - system ready")
            
            # Note: We no longer load all embeddings into memory
            # Qdrant handles the storage and retrieval efficiently
            self.logger.info("Using Qdrant for vector storage and similarity search")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def reindex_all_embeddings(self):
        """Re-index all embeddings from database to Qdrant"""
        try:
            self.logger.info("Starting re-indexing of all embeddings...")
            
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, name, image_path FROM persons')
                persons = cursor.fetchall()
            
            reindexed_count = 0
            for person_id, name, image_path in persons:
                try:
                    if image_path and os.path.exists(image_path):
                        # Extract embedding from saved image
                        embedding_data = self.extract_face_embedding(image_path)
                        if embedding_data:
                            # Add to Qdrant
                            self.vector_db.add_embedding(
                                embedding=embedding_data['embedding'],
                                person_id=person_id,
                                metadata={'name': name, 'image_path': image_path}
                            )
                            reindexed_count += 1
                            self.logger.debug(f"Re-indexed person {person_id}: {name}")
                        else:
                            self.logger.warning(f"Could not extract embedding for person {person_id}: {name}")
                    else:
                        self.logger.warning(f"Image not found for person {person_id}: {image_path}")
                except Exception as e:
                    self.logger.error(f"Error re-indexing person {person_id}: {e}")
            
            self.logger.info(f"Re-indexing completed: {reindexed_count}/{len(persons)} persons processed")
            
        except Exception as e:
            self.logger.error(f"Error during re-indexing: {e}")
    
    def search_person(self, query_embedding: np.ndarray, k=5) -> List[Dict]:
        """
        Search for similar persons using Qdrant vector database
        
        Args:
            query_embedding: Query face embedding
            k: Number of top results to return
            
        Returns:
            List of similar persons with scores
        """
        try:
            # Validate input
            if query_embedding is None or len(query_embedding) == 0:
                self.logger.error("Invalid query embedding")
                return []
            
            # Normalize query embedding for cosine similarity
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                self.logger.error("Query embedding has zero norm")
                return []
            
            query_embedding_normalized = query_embedding / query_norm
            
            # Use Qdrant for similarity search
            results = self.vector_db.search_similar(
                query_embedding=query_embedding_normalized,
                k=k,
                threshold=self.similarity_thresh
            )
            
            # Log results for debugging
            if results:
                self.logger.debug(f"Found {len(results)} similar persons (top similarity: {results[0]['similarity']:.3f})")
            else:
                self.logger.debug(f"No similar persons found above threshold {self.similarity_thresh}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching for similar persons: {e}")
            return []
    
    def update_person_stats(self, person_id: int):
        """Update person statistics (last seen, match count)"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE persons 
            SET last_seen = CURRENT_TIMESTAMP, match_count = match_count + 1
            WHERE id = ?
        ''', (person_id,))
        conn.commit()
        conn.close()
    
    def store_visit_info(self, person_id: int, visit_id: str, customer_id: str, 
                        entry_time: str, image_url: str, saved_image_path: str, similarity: float,
                        branchId: str = '', camera: str = '', entryEventIds: str = ''):
        """Store visit information in database"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO person_visits 
                    (person_id, visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, branchId, camera, entryEventIds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (person_id, visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, branchId, camera, entryEventIds))
                
                conn.commit()
            except Exception as e:
                self.logger.error(f"Error storing visit info: {e}")
                conn.rollback()
                raise
    
    def store_visits_batch(self, visits: List[Dict]):
        """Store multiple visits in a single database transaction"""
        if not visits:
            return
        
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Prepare data for batch insert
                visit_data = []
                for visit in visits:
                    visit_data.append((
                        visit['person_id'],
                        visit['visit_id'],
                        visit['customer_id'],
                        visit['entry_time'],
                        visit['image_url'],
                        visit.get('saved_image_path'),
                        visit['similarity']
                    ))
                
                # Batch insert
                cursor.executemany('''
                    INSERT OR REPLACE INTO person_visits 
                    (person_id, visit_id, customer_id, entry_time, image_url, saved_image_path, similarity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', visit_data)
                
                conn.commit()
                self.logger.info(f"Stored {len(visits)} visits in batch")
                
            except Exception as e:
                self.logger.error(f"Error storing visits batch: {e}")
                conn.rollback()
                raise
    
    def store_low_similarity_image(self, visit_id: str, customer_id: str, entry_time: str, 
                                 image_url: str, saved_image_path: str, similarity: float, 
                                 best_match_name: str = None, reason: str = None):
        """Store low similarity images in a separate table"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                # Create low similarity table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS low_similarity_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        visit_id TEXT,
                        customer_id TEXT,
                        entry_time TEXT,
                        image_url TEXT,
                        saved_image_path TEXT,
                        similarity REAL,
                        best_match_name TEXT,
                        reason TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Add reason column if it doesn't exist (for existing databases)
                try:
                    cursor.execute('ALTER TABLE low_similarity_images ADD COLUMN reason TEXT')
                except sqlite3.OperationalError:
                    # Column already exists, ignore error
                    pass
                
                cursor.execute('''
                    INSERT INTO low_similarity_images 
                        (visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, best_match_name, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, best_match_name, reason))
                
                conn.commit()
            except Exception as e:
                self.logger.error(f"Error storing low similarity image: {e}")
                conn.rollback()
                raise
    
    def store_failed_image(self, visit_id: str, customer_id: str, entry_time: str, 
                          image_url: str, saved_image_path: str, reason: str):
        """Store failed images (no faces detected) in a separate table"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                # Create failed images table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS failed_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        visit_id TEXT,
                        customer_id TEXT,
                        entry_time TEXT,
                        image_url TEXT,
                        saved_image_path TEXT,
                        reason TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Download the image if URL is provided and not already saved
                if image_url and not saved_image_path:
                    try:
                        # Ensure image cache directory exists
                        os.makedirs(self.image_cache_dir, exist_ok=True)
                        
                        # Create a proper save path for the downloaded image
                        filename = f"{visit_id}_{customer_id}.jpg"
                        save_path = os.path.join(self.image_cache_dir, filename)
                        
                        # Download the image and save it
                        downloaded_image = self.download_image_from_url(image_url, save_path=save_path)
                        if downloaded_image is not None:
                            saved_image_path = save_path  # Use the save_path as the saved_image_path
                            self.logger.info(f"Downloaded failed image: {visit_id} -> {saved_image_path}")
                        else:
                            self.logger.warning(f"Failed to download image for {visit_id}")
                            saved_image_path = None
                    except Exception as e:
                        self.logger.warning(f"Failed to download image for {visit_id}: {e}")
                        saved_image_path = None
                
                # Insert the failed image record
                cursor.execute('''
                    INSERT INTO failed_images 
                    (visit_id, customer_id, entry_time, image_url, saved_image_path, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (visit_id, customer_id, entry_time, image_url, saved_image_path, reason))
                
                conn.commit()
                self.logger.info(f"Stored failed image: {visit_id} - {reason}")
                
            except Exception as e:
                self.logger.error(f"Error storing failed image: {e}")
                conn.rollback()
                raise
    
    def get_best_quality_image(self, visits: List[Dict], max_check: int = 20) -> Optional[Dict]:
        """
        Get the best quality image from first N visits for initial person creation
        
        Args:
            visits: List of visits to check
            max_check: Maximum number of visits to check
            
        Returns:
            Best quality visit with embedding data or None
        """
        self.logger.info(f"Selecting best quality image from first {max_check} visits...")
        
        best_visit = None
        best_score = 0.0
        checked = 0
        
        for visit in visits[:max_check]:
            try:
                image_url = visit.get('image') or visit.get('imageUrl')
                if not image_url:
                    continue
                    
                embedding_data = self.extract_face_embedding(image_url, save_image=False)
                if embedding_data and embedding_data.get('embedding') is not None:
                    quality_score = embedding_data['quality']['overall']
                    confidence = embedding_data.get('face_confidence', 0.0)
                    combined_score = quality_score * 0.6 + confidence * 0.4
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_visit = {
                            'visit': visit,
                            'embedding_data': embedding_data
                        }
                        
                    checked += 1
                    if checked >= max_check:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Error checking visit {visit.get('id')}: {e}")
                continue
        
        if best_visit:
            self.logger.info(f"Selected best image with score: {best_score:.3f}")
            return best_visit
        else:
            self.logger.warning("No suitable image found, will use first available")
            return None
    
    @monitor_performance
    def process_visit_data(self, json_file_path: str, output_folder: str = None, max_visits: int = None, save_images: bool = True):
        """
        Process visit data from JSON file with image URLs - main function
        
        Args:
            json_file_path: Path to JSON file containing visit data
            output_folder: Folder to organize recognized persons (optional)
            max_visits: Maximum number of visits to process (for testing)
            save_images: Whether to save downloaded images locally
        """
        self.logger.info(f"Processing visit data from: {json_file_path}")
        
        # Clean up memory before starting
        self.cleanup_memory()
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            # Create images subfolder if saving images
            if save_images:
                images_folder = os.path.join(output_folder, "downloaded_images")
                os.makedirs(images_folder, exist_ok=True)
        
        # Load visit data
        visits = self.load_visit_data(json_file_path)
        if not visits:
            self.logger.warning("No valid visits found in JSON file")
            return
        
        # Limit visits for testing if specified
        if max_visits and max_visits < len(visits):
            visits = visits[:max_visits]
            self.logger.info(f"Processing first {max_visits} visits for testing")
        
        self.logger.info(f"Processing {len(visits)} visits")
        
        # Process each visit
        results = {
            'processed': 0,
            'recognized': 0,
            'new_persons': 0,
            'no_faces': 0,
            'low_quality': 0,
            'download_failed': 0,
            'duplicate_faces': 0,
            'low_similarity': 0
        }
        
        # Thread-safe aggregation for batch webhook
        results_lock = threading.Lock()
        batch_groups = []  # Will store all person groups for final webhook
        
        def process_single_visit(visit_data):
            """Process a single visit - thread-safe function"""
            i, visit = visit_data
            visit_results = {
                'processed': 0,
                'recognized': 0,
                'new_persons': 0,
                'no_faces': 0,
                'low_quality': 0,
                'download_failed': 0,
                'duplicate_faces': 0,
                'low_similarity': 0
            }
            person_group = None
            
            try:
                visit_id = visit.get('id')
                image_url = visit.get('image')
                customer_id = visit.get('customerId')
                entry_time = visit.get('entryTime', '')
                
                # Skip visits without required data
                if not visit_id or not customer_id or not image_url:
                    self.logger.warning(f"Skipping visit {i+1}: missing required data (visit_id: {visit_id}, customer_id: {customer_id}, image: {bool(image_url)})")
                    visit_results['no_faces'] += 1
                    visit_results['processed'] += 1  # Count as processed even if skipped
                    return visit_results, person_group
                
                self.logger.info(f"Processing visit {visit_id} ({i+1}/{len(visits)})")
                self.logger.info(f"Customer: {customer_id}, Time: {entry_time}")
                
                # Extract face embedding from image URL with retry logic
                images_dir = os.path.join(output_folder, "downloaded_images") if (output_folder and save_images) else None
                
                # Retry logic for image processing
                max_retries = self.config['image_processing'].get('retry_attempts', 3)
                retry_delay = self.config['image_processing'].get('retry_delay', 0.5)
                embedding_data = None
                
                for attempt in range(max_retries):
                    try:
                        embedding_data = self.extract_face_embedding(image_url, save_image=save_images, output_dir=images_dir)
                        if embedding_data is not None:
                            break
                        else:
                            self.logger.warning(f"Attempt {attempt + 1} failed for {image_url}: No face detected")
                    except Exception as e:
                        self.logger.warning(f"Attempt {attempt + 1} failed for {image_url}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        else:
                            self.logger.error(f"All {max_retries} attempts failed for {image_url}")
                
                if embedding_data is None:
                    # Store in failed images table
                    self.store_failed_image(visit_id, customer_id, entry_time, 
                                          image_url, None, "No face detected")
                    visit_results['no_faces'] += 1
                    visit_results['processed'] += 1  # Count as processed even if no face detected
                    
                    # Create a special "no face" group to ensure these visits are included in clustering results
                    person_group = {
                        'person_id': f"no_face_{visit_id}",
                        'person_name': f"No_Face_{customer_id}_{int(time.time())}",
                        'visits': [{
                            'id': visit.get('id'),
                            'customerId': visit.get('customerId'),
                            'image': visit.get('image'),
                            'entryTime': visit.get('entryTime'),
                            'similarity': 0.0,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'exitEventIds': visit.get('exitEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {}),
                            'status': 'no_face_detected'
                        }]
                    }
                    
                    return visit_results, person_group
            except Exception as e:
                self.logger.error(f"Error processing visit {i}: {e}")
                visit_results['no_faces'] += 1
                visit_results['processed'] += 1  # Count as processed even if error occurred
                
                # Create a special "error" group to ensure these visits are included in clustering results
                person_group = {
                    'person_id': f"error_{visit.get('id', f'unknown_{i}')}",
                    'person_name': f"Error_{visit.get('customerId', 'unknown')}_{int(time.time())}",
                    'visits': [{
                        'id': visit.get('id'),
                        'customerId': visit.get('customerId'),
                        'image': visit.get('image'),
                        'entryTime': visit.get('entryTime'),
                        'similarity': 0.0,
                        'branchId': visit.get('branchId', ''),
                        'camera': visit.get('camera', ''),
                        'entryEventIds': visit.get('entryEventIds', []),
                        'exitEventIds': visit.get('exitEventIds', []),
                        'customer': visit.get('customer', {}),
                        'results': visit.get('results', {}),
                        'status': 'processing_error',
                        'error': str(e)
                    }]
                }
                
                return visit_results, person_group
            
            # Check for duplicates (disabled for clustering to ensure all visits are processed)
            # try:
            #     if self.is_duplicate_image(image_url, embedding_data['embedding']):
            #         self.logger.info(f"Skipping duplicate image: {image_url}")
            #         visit_results['duplicate_faces'] += 1
            #         return visit_results, person_group
            # except Exception as e:
            #     self.logger.error(f"Error checking duplicates for {image_url}: {e}")
            #     # Continue processing even if duplicate check fails
            
            visit_results['processed'] += 1
            
            # If no persons in DB, add the first image as a new person
            if self.vector_db.get_embedding_count() == 0:
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"First person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, image_url, embedding_data.get('saved_image_path'), 1.0,
                                         visit.get('branchId', ''), visit.get('camera', ''), json.dumps(visit.get('entryEventIds', [])))
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'id': visit.get('id'),
                            'customerId': visit.get('customerId'),
                            'image': visit.get('image'),
                            'entryTime': visit.get('entryTime'),
                            'similarity': 1.0,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    visit_results['duplicate_faces'] += 1
                    return visit_results, person_group

            # Otherwise, do similarity search as usual
            search_results = self.search_person(embedding_data['embedding'], k=5)  # Get more results
            similarity = search_results[0]['similarity'] if search_results else 0.0
            best_match = search_results[0] if search_results else None
            
            # Use a more strict threshold for grouping
            grouping_threshold = self.config['face_recognition']['grouping_threshold_json']
            
            if search_results and similarity >= grouping_threshold:
                # Person recognized (group with best match)
                person_id = best_match['person_id']
                person_name = best_match['name']
                
                self.logger.info(f"Recognized: {person_name} (similarity: {similarity:.3f})")
                
                # Update statistics
                self.update_person_stats(person_id)
                
                # Store visit information in database
                self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                    image_url, embedding_data.get('saved_image_path'), similarity,
                                    visit.get('branchId', ''), visit.get('camera', ''), json.dumps(visit.get('entryEventIds', [])))
                
                # Create person group for batch webhook
                person_group = {
                    'person_id': person_id,
                    'person_name': person_name,
                    'visits': [{
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'customerId': visit.get('customerId', customer_id),
                        'image_url': image_url,
                        'image': visit.get('image', image_url),
                        'entry_time': entry_time,
                        'entryTime': visit.get('entryTime', entry_time),
                        'similarity': similarity,
                        'branchId': visit.get('branchId', ''),
                        'camera': visit.get('camera', ''),
                        'entryEventIds': visit.get('entryEventIds', []),
                        'customer': visit.get('customer', {}),
                        'results': visit.get('results', {})
                    }]
                }
                
                # Save visit info if output folder specified
                if output_folder:
                    person_folder = os.path.join(output_folder, f"{person_name}_{person_id}")
                    os.makedirs(person_folder, exist_ok=True)
                    
                    # Save visit metadata
                    visit_info = {
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'entry_time': entry_time,
                        'image_url': image_url,
                        'saved_image_path': embedding_data.get('saved_image_path'),
                        'similarity': similarity,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    visit_file = os.path.join(person_folder, f"visit_{visit_id}.json")
                    with open(visit_file, 'w') as f:
                        json.dump(visit_info, f, indent=2)
                
                visit_results['recognized'] += 1
            else:
                # Not recognized (low similarity) or no match - create new person
                self.logger.info(f"Low similarity: {similarity:.3f} (grouping threshold: {grouping_threshold:.3f}) - creating new person.")
                
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"New person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                        image_url, embedding_data.get('saved_image_path'), similarity)
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'id': visit.get('id'),
                            'customerId': visit.get('customerId'),
                            'image': visit.get('image'),
                            'entryTime': visit.get('entryTime'),
                            'similarity': similarity,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    self.logger.warning(f"Failed to add new person: {person_name}")
                    visit_results['duplicate_faces'] += 1
            
            return visit_results, person_group
        
        # Process visits using multi-threading with enhanced performance
        max_workers = min(self.config['image_processing']['max_workers'], len(visits))
        self.logger.info(f"Processing {len(visits)} visits with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all visits for processing
            future_to_visit = {executor.submit(process_single_visit, (i, visit)): (i, visit) for i, visit in enumerate(visits)}
            
            # Collect results as they complete with memory management
            processed_count = 0
            for future in as_completed(future_to_visit):
                try:
                    visit_results, person_group = future.result()
                    
                    # Thread-safe aggregation of results
                    with results_lock:
                        for key in results:
                            results[key] += visit_results[key]
                        
                        # Add person group to batch webhook data
                        if person_group:
                            batch_groups.append(person_group)
                    
                    processed_count += 1
                    
                    # Memory cleanup every 50 images to prevent OOM
                    if processed_count % 50 == 0:
                        self.logger.info(f"Processed {processed_count}/{len(visits)} visits, cleaning memory...")
                        self.cleanup_memory()
                        self.clear_gpu_memory()
                            
                except Exception as e:
                    i, visit = future_to_visit[future]
                    self.logger.error(f"Error processing visit {i}: {e}")
                    with results_lock:
                        results['no_faces'] += 1
        
        # Print summary
        self.logger.info("Processing completed!")
        self.logger.info(f"Results: {results}")
        
        # Save clustering results to JSON file
        if batch_groups:
            try:
                self.logger.info(f"Saving clustering results with {len(batch_groups)} groups")
                
                # Calculate total processed including no_face visits
                total_processed = results['processed'] + results['no_faces']
                
                # Save clustering results to JSON
                success = save_clustering_results(
                    groups=batch_groups,
                    total_processed=total_processed,
                    results=results
                )
                
                if success:
                    self.logger.info("Clustering results saved successfully")
                else:
                    self.logger.warning("Failed to save clustering results")
                    
            except Exception as e:
                self.logger.error(f"Error saving clustering results: {e}")
        else:
            self.logger.info("No groups to save in clustering results")
        
        # Clean up memory after processing
        self.cleanup_memory()
        
        return results
    
    @monitor_performance
    def process_visit_data_from_json(self, json_data: dict, output_folder: str = None, max_visits: int = None, save_images: bool = True, clear_existing: bool = False):
        """
        Process visit data from JSON data (not file) - main function
        
        Args:
            json_data: Dictionary containing visit data
            output_folder: Folder to organize recognized persons (optional)
            max_visits: Maximum number of visits to process (for testing)
            save_images: Whether to save downloaded images locally
            clear_existing: Whether to clear existing data before processing
        """
        self.logger.info(f"Processing visit data from JSON data")
        
        # Clean up memory before starting
        self.cleanup_memory()
        
        # Clear existing data if requested
        if clear_existing:
            self.logger.info("Clearing existing data before processing new JSON...")
            self.clear_all_data()
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            # Create images subfolder if saving images
            if save_images:
                images_folder = os.path.join(output_folder, "downloaded_images")
                os.makedirs(images_folder, exist_ok=True)
        
        # Extract visits from JSON data
        visits = json_data.get('visits', [])
        if not visits:
            self.logger.warning("No visits found in JSON data")
            return {
                'processed': 0,
                'recognized': 0,
                'new_persons': 0,
                'no_faces': 0,
                'low_quality': 0,
                'download_failed': 0,
                'duplicate_faces': 0
            }
        
        # Filter visits with valid image URLs
        valid_visits = []
        for visit in visits:
            if visit.get('image') and visit.get('image').startswith('http'):
                valid_visits.append(visit)
        
        self.logger.info(f"Found {len(valid_visits)} visits with valid image URLs")
        
        # Limit visits for testing if specified
        if max_visits and max_visits < len(valid_visits):
            valid_visits = valid_visits[:max_visits]
            self.logger.info(f"Processing first {max_visits} visits for testing")
        
        self.logger.info(f"Processing {len(valid_visits)} visits")
        
        # Process each visit
        results = {
            'processed': 0,
            'recognized': 0,
            'new_persons': 0,
            'no_faces': 0,
            'low_quality': 0,
            'download_failed': 0,
            'duplicate_faces': 0,
            'low_similarity': 0
        }
        
        # Thread-safe aggregation for batch webhook
        results_lock = threading.Lock()
        batch_groups = []  # Will store all person groups for final webhook
        person_groups_dict = {}  # Dictionary to group by person_id for merging duplicates
        
        def process_single_visit_json(visit_data):
            """Process a single visit from JSON - thread-safe function"""
            i, visit = visit_data
            visit_results = {
                'processed': 0,
                'recognized': 0,
                'new_persons': 0,
                'no_faces': 0,
                'low_quality': 0,
                'download_failed': 0,
                'duplicate_faces': 0,
                'low_similarity': 0
            }
            person_group = None
            
            visit_id = visit.get('id')
            image_url = visit.get('image')
            customer_id = visit.get('customerId')
            entry_time = visit.get('entryTime', '')
            
            # Skip visits without required data
            if not visit_id or not customer_id or not image_url:
                self.logger.warning(f"Skipping visit {i+1}: missing required data (visit_id: {visit_id}, customer_id: {customer_id}, image: {bool(image_url)})")
                visit_results['no_faces'] += 1
                return visit_results, person_group
            
            self.logger.info(f"Processing visit {visit_id} ({i+1}/{len(valid_visits)})")
            self.logger.info(f"Customer: {customer_id}, Time: {entry_time}")
            
            # Clear GPU memory every 10 images to prevent memory issues
            if i % 10 == 0:
                self.clear_gpu_memory()
            
            # Check for side face using JSON bbox data first
            is_side_face, side_reason, bbox_data = self.check_side_face_from_json_bbox(visit)
            if is_side_face:
                self.logger.info(f"Side face detected from JSON bbox: {side_reason}")
                # Store in low similarity table as unusable image
                self.store_low_similarity_image(visit_id, customer_id, entry_time, 
                                              image_url, None, 0.0, f"Side face: {side_reason}")
                visit_results['no_faces'] += 1
                return visit_results, person_group
            
            # Extract face embedding from image URL
            images_dir = os.path.join(output_folder, "downloaded_images") if (output_folder and save_images) else None
            embedding_data = self.extract_face_embedding(image_url, save_image=save_images, output_dir=images_dir)
            if embedding_data is None:
                # Store in failed images table
                self.store_failed_image(visit_id, customer_id, entry_time, 
                                      image_url, None, "No face detected")
                visit_results['no_faces'] += 1
                
                # Create a special "no face" group to ensure these visits are included in clustering results
                person_group = {
                    'person_id': f"no_face_{visit_id}",
                    'person_name': f"No_Face_{customer_id}_{int(time.time())}",
                    'visits': [{
                        'id': visit.get('id'),
                        'customerId': visit.get('customerId'),
                        'image': visit.get('image'),
                        'entryTime': visit.get('entryTime'),
                        'similarity': 0.0,
                        'branchId': visit.get('branchId', ''),
                        'camera': visit.get('camera', ''),
                        'entryEventIds': visit.get('entryEventIds', []),
                        'exitEventIds': visit.get('exitEventIds', []),
                        'customer': visit.get('customer', {}),
                        'results': visit.get('results', {}),
                        'status': 'no_face_detected'
                    }]
                }
                
                return visit_results, person_group
            
            # Check for duplicates (disabled for clustering to ensure all visits are processed)
            # try:
            #     if self.is_duplicate_image(image_url, embedding_data['embedding']):
            #         self.logger.info(f"Skipping duplicate image: {image_url}")
            #         visit_results['duplicate_faces'] += 1
            #         return visit_results, person_group
            # except Exception as e:
            #     self.logger.error(f"Error checking duplicates for {image_url}: {e}")
            #     # Continue processing even if duplicate check fails
            
            visit_results['processed'] += 1
            
            # If no persons in DB, add the first image as a new person
            if self.vector_db.get_embedding_count() == 0:
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"First person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, image_url, embedding_data.get('saved_image_path'), 1.0,
                                         visit.get('branchId', ''), visit.get('camera', ''), json.dumps(visit.get('entryEventIds', [])))
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'id': visit.get('id'),
                            'customerId': visit.get('customerId'),
                            'image': visit.get('image'),
                            'entryTime': visit.get('entryTime'),
                            'similarity': 1.0,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    visit_results['duplicate_faces'] += 1
                return visit_results, person_group

            # Otherwise, do similarity search as usual
            search_results = self.search_person(embedding_data['embedding'], k=5)  # Get more results
            similarity = search_results[0]['similarity'] if search_results else 0.0
            best_match = search_results[0] if search_results else None
            
            # Use a more strict threshold for grouping
            grouping_threshold = self.config['face_recognition']['grouping_threshold_json']
            
            if search_results and similarity >= grouping_threshold:
                # Person recognized (group with best match)
                person_id = best_match['person_id']
                person_name = best_match['name']
                
                self.logger.info(f"Recognized: {person_name} (similarity: {similarity:.3f})")
                
                # Update statistics
                self.update_person_stats(person_id)
                
                # Store visit information in database
                self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                    image_url, embedding_data.get('saved_image_path'), similarity,
                                    visit.get('branchId', ''), visit.get('camera', ''), json.dumps(visit.get('entryEventIds', [])))
                
                # Create person group for batch webhook
                person_group = {
                    'person_id': person_id,
                    'person_name': person_name,
                    'visits': [{
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'customerId': visit.get('customerId', customer_id),
                        'image_url': image_url,
                        'image': visit.get('image', image_url),
                        'entry_time': entry_time,
                        'entryTime': visit.get('entryTime', entry_time),
                        'similarity': similarity,
                        'branchId': visit.get('branchId', ''),
                        'camera': visit.get('camera', ''),
                        'entryEventIds': visit.get('entryEventIds', []),
                        'customer': visit.get('customer', {}),
                        'results': visit.get('results', {})
                    }]
                }
                
                # Save visit info if output folder specified
                if output_folder:
                    person_folder = os.path.join(output_folder, f"{person_name}_{person_id}")
                    os.makedirs(person_folder, exist_ok=True)
                    
                    # Save visit metadata
                    visit_info = {
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'entry_time': entry_time,
                        'image_url': image_url,
                        'saved_image_path': embedding_data.get('saved_image_path'),
                        'similarity': similarity,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    visit_file = os.path.join(person_folder, f"visit_{visit_id}.json")
                    with open(visit_file, 'w') as f:
                        json.dump(visit_info, f, indent=2)
                
                visit_results['recognized'] += 1
            else:
                # Not recognized (low similarity) or no match - create new person
                self.logger.info(f"Low similarity: {similarity:.3f} (grouping threshold: {grouping_threshold:.3f}) - creating new person.")
                
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"New person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                        image_url, embedding_data.get('saved_image_path'), similarity)
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'id': visit.get('id'),
                            'customerId': visit.get('customerId'),
                            'image': visit.get('image'),
                            'entryTime': visit.get('entryTime'),
                            'similarity': similarity,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    self.logger.warning(f"Failed to add new person: {person_name}")
                    visit_results['duplicate_faces'] += 1
            
            return visit_results, person_group
        
        # Process visits using multi-threading
        max_workers = min(self.config['image_processing']['max_workers'], len(valid_visits))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all visits for processing
            future_to_visit = {executor.submit(process_single_visit_json, (i, visit)): (i, visit) for i, visit in enumerate(valid_visits)}
            
            # Collect results as they complete
            for future in as_completed(future_to_visit):
                try:
                    visit_results, person_group = future.result()
                    
                    # Thread-safe aggregation of results
                    with results_lock:
                        for key in results:
                            results[key] += visit_results[key]
                        
                        # Add person group to batch webhook data with merging logic
                        if person_group:
                            person_id = person_group['person_id']
                            
                            if person_id in person_groups_dict:
                                # Merge with existing group - add visit to existing person
                                existing_group = person_groups_dict[person_id]
                                existing_group['visits'].extend(person_group['visits'])
                                # Update visit count
                                existing_group['visit_count'] = len(existing_group['visits'])
                                # Update group score to highest similarity
                                if person_group['visits']:
                                    new_similarity = person_group['visits'][0].get('similarity', 0)
                                    existing_group['group_score'] = max(existing_group.get('group_score', 0), new_similarity)
                                # Update group_id to reflect it's a merged group
                                existing_group['group_id'] = f"group_{person_id}"
                                self.logger.info(f"Merged visit into existing person group {person_id} (total visits: {existing_group['visit_count']})")
                            else:
                                # New person group
                                person_group['visit_count'] = len(person_group['visits'])
                                person_group['group_score'] = person_group['visits'][0].get('similarity', 1.0) if person_group['visits'] else 1.0
                                # Update group_id to reflect it's a person group
                                person_group['group_id'] = f"group_{person_id}"
                                person_groups_dict[person_id] = person_group
                                self.logger.info(f"Created new person group {person_id}")
                            
                except Exception as e:
                    i, visit = future_to_visit[future]
                    self.logger.error(f"Error processing visit {i}: {e}")
                    with results_lock:
                        results['no_faces'] += 1
        
        # Convert person_groups_dict to batch_groups list
        batch_groups = list(person_groups_dict.values())
        
        # Print summary
        self.logger.info("Processing completed!")
        self.logger.info(f"Results: {results}")
        self.logger.info(f"Total person groups after merging: {len(batch_groups)}")
        
        # Save clustering results to JSON file
        if batch_groups:
            try:
                self.logger.info(f"Saving clustering results with {len(batch_groups)} groups")
                
                # Calculate total processed including no_face visits
                total_processed = results['processed'] + results['no_faces']
                
                # Save clustering results to JSON
                success = save_clustering_results(
                    groups=batch_groups,
                    total_processed=total_processed,
                    results=results
                )
                
                if success:
                    self.logger.info("Clustering results saved successfully")
                else:
                    self.logger.warning("Failed to save clustering results")
                    
            except Exception as e:
                self.logger.error(f"Error saving clustering results: {e}")
        else:
            self.logger.info("No groups to save in clustering results")
        
        # Clean up memory after processing
        self.cleanup_memory()
        
        return results
    
    # Removed incomplete process_visit_data_optimized function (references undefined extract_face_embedding_from_image)
    
    @monitor_performance
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total persons
            cursor.execute('SELECT COUNT(*) FROM persons')
            total_persons = cursor.fetchone()[0]
            
            # Average quality
            cursor.execute('SELECT AVG(face_quality) FROM persons')
            avg_quality = cursor.fetchone()[0] or 0
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM persons 
                WHERE last_seen > datetime('now', '-1 day')
            ''')
            recent_activity = cursor.fetchone()[0]
            
        return {
            'total_persons': total_persons,
            'average_quality': float(avg_quality),
            'recent_activity': recent_activity,
            'embeddings_loaded': self.vector_db.get_embedding_count()
        }
    
    def validate_real_customer_data(self, data):
        """Validate that data contains real customer information"""
        if not data:
            return False
        
        # Check for real customer IDs (not test data)
        for group in data:
            for visit in group.get('images', []):
                customer_id = visit.get('customer_id', '')
                # Real customer IDs are typically longer and more complex
                if customer_id and len(customer_id) > 10 and not customer_id.startswith('customer_') and not customer_id.startswith('Person_'):
                    return True  # Found real customer ID
                # Also check for visitor IDs (real format)
                if customer_id and customer_id.startswith('visitor-'):
                    return True  # Found real visitor ID
        
        return False  # Only test data found

    def get_person_groups_for_web(self) -> List[Dict]:
        """
        Get person groups with images for web interface
        
        Returns:
            List of person groups with visit information
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check if person_visits table exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_visits'")
        visits_table_exists = cursor.fetchone() is not None
        
        if not visits_table_exists:
            # If no visits table, just return persons with their main image
            cursor.execute('''
                SELECT id, name, image_path, face_quality, match_count, last_seen
                FROM persons
                ORDER BY match_count DESC, last_seen DESC
            ''')
            
            persons = cursor.fetchall()
            person_groups = []
            
            for person in persons:
                person_id, name, image_path, face_quality, match_count, last_seen = person
                
                person_groups.append({
                    'person_id': person_id,
                    'name': name,
                    'image_path': image_path,
                    'face_quality': face_quality,
                    'match_count': match_count,
                    'last_seen': last_seen,
                    'visit_count': 0,
                    'avg_quality': face_quality,
                    'images': [{
                        'visit_id': f'person_{person_id}',
                        'customer_id': name,
                        'entry_time': last_seen or '',
                        'image_url': image_path,
                        'image_path': image_path,
                        'similarity': 1.0
                    }] if image_path else []
                })
            
            conn.close()
            return person_groups
        
        # Get all persons with their visit information
        cursor.execute('''
            SELECT 
                p.id,
                p.name,
                p.image_path,
                p.face_quality,
                p.match_count,
                p.last_seen,
                COUNT(v.visit_id) as visit_count
            FROM persons p
            LEFT JOIN (
                SELECT DISTINCT 
                    person_id,
                    visit_id,
                    entry_time,
                    image_url,
                    saved_image_path,
                    branchId,
                    camera,
                    entryEventIds
                FROM person_visits
            ) v ON p.id = v.person_id
            GROUP BY p.id, p.name, p.image_path, p.face_quality, p.match_count, p.last_seen
            ORDER BY p.match_count DESC, p.last_seen DESC
        ''')
        
        persons = cursor.fetchall()
        
        # Get visit details for each person
        person_groups = []
        for person in persons:
            person_id, name, image_path, face_quality, match_count, last_seen, visit_count = person
            
            # Get all visits for this person
            cursor.execute('''
                SELECT 
                    visit_id,
                    customer_id,
                    entry_time,
                    image_url,
                    saved_image_path,
                    similarity,
                    branchId,
                    camera,
                    entryEventIds
                FROM person_visits
                WHERE person_id = ?
                ORDER BY entry_time DESC
            ''', (person_id,))
            
            visits = cursor.fetchall()
            
            # Prepare images list, include all images for this person
            images = []
            for visit in visits:
                visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, branchId, camera, entryEventIds = visit
                if similarity is not None:
                    # Use saved image path if available, otherwise cache the URL
                    if saved_image_path and os.path.exists(saved_image_path):
                        display_path = saved_image_path
                    elif image_url and image_url.startswith('http'):
                        cached_path = self.get_cached_image_path(image_url)
                        display_path = cached_path if cached_path else image_url
                    else:
                        display_path = image_url
                    images.append({
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'entry_time': entry_time,
                        'image_url': image_url,
                        'image_path': display_path,
                        'similarity': similarity,
                        'branchId': branchId or '',
                        'camera': camera or '',
                        'entryEventIds': entryEventIds or ''
                    })
            
            # If no visits but person exists, add their main image
            if not images and image_path:
                images.append({
                    'visit_id': f'person_{person_id}',
                    'customer_id': name,
                    'entry_time': last_seen or '',
                    'image_url': image_path,
                    'image_path': image_path,
                    'similarity': 1.0
                })
            
            person_groups.append({
                'person_id': person_id,
                'name': name,
                'image_path': image_path,
                'face_quality': face_quality,
                'match_count': match_count,
                'last_seen': last_seen,
                'visit_count': visit_count,
                'avg_quality': face_quality,
                'images': images
            })
        
        conn.close()
        return person_groups
    
    def get_web_stats(self) -> Dict:
        """
        Get statistics for web interface
        
        Returns:
            Dictionary with web statistics
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Total persons
        cursor.execute('SELECT COUNT(*) FROM persons')
        total_persons = cursor.fetchone()[0]
        
        # Check if person_visits table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_visits'")
        visits_table_exists = cursor.fetchone() is not None
        
        if visits_table_exists:
            # Total visits
            cursor.execute('SELECT COUNT(DISTINCT visit_id) FROM person_visits')
            total_visits = cursor.fetchone()[0]
            
            # Total images (unique image URLs)
            cursor.execute('SELECT COUNT(DISTINCT image_url) FROM person_visits')
            total_images = cursor.fetchone()[0]
        else:
            # If no visits table, count persons as images
            total_visits = 0
            cursor.execute('SELECT COUNT(*) FROM persons WHERE image_path IS NOT NULL')
            total_images = cursor.fetchone()[0]
        
        # Low similarity images count (calculated dynamically from clustering results)
        try:
            person_groups = self.get_person_groups_for_web()
            low_similarity_count = 0
            for group in person_groups:
                visits = group.get('visits', [])
                for visit in visits:
                    similarity = visit.get('similarity', 0)
                    if similarity <= 0.3:
                        low_similarity_count += 1
        except Exception as e:
            self.logger.warning(f"Error calculating low similarity count: {e}")
            low_similarity_count = 0
        
        # Failed images count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='failed_images'")
        failed_table_exists = cursor.fetchone() is not None
        
        if failed_table_exists:
            cursor.execute('SELECT COUNT(*) FROM failed_images')
            failed_images_count = cursor.fetchone()[0]
        else:
            failed_images_count = 0
        
        # Total unusable images (low similarity + failed)
        total_unusable = low_similarity_count + failed_images_count
        
        # Recent activity (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM persons 
            WHERE last_seen > datetime('now', '-1 day')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_persons': total_persons,
            'total_visits': total_visits,
            'total_images': total_images,
            'low_similarity_count': low_similarity_count,
            'failed_images_count': failed_images_count,
            'total_unusable': total_unusable,
            'recent_activity': recent_activity
        }
    
    def get_low_similarity_images(self) -> List[Dict]:
        """
        Get low similarity images for web interface
        
        Returns:
            List of low similarity images with details
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check if low similarity table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='low_similarity_images'")
            if not cursor.fetchone():
                conn.close()
                return []
            
            # Get low similarity images
            cursor.execute('''
                SELECT 
                    visit_id,
                    customer_id,
                    entry_time,
                    image_url,
                    saved_image_path,
                    similarity,
                    best_match_name,
                    reason,
                    processed_at
                FROM low_similarity_images
                ORDER BY similarity DESC, processed_at DESC
            ''')
            
            images = []
            for row in cursor.fetchall():
                visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, best_match_name, reason, processed_at = row
                
                # Use saved image path if available, otherwise cache the URL
                if saved_image_path and os.path.exists(saved_image_path):
                    display_path = saved_image_path
                elif image_url and image_url.startswith('http'):
                    # Try to get cached version of the URL
                    cached_path = self.get_cached_image_path(image_url)
                    display_path = cached_path if cached_path else image_url
                else:
                    display_path = image_url
                
                images.append({
                    'visit_id': visit_id,
                    'customer_id': customer_id,
                    'entry_time': entry_time,
                    'image_url': image_url,
                    'image_path': display_path,
                    'similarity': similarity if similarity is not None else 0.0,  # Keep as raw float (0.0-1.0)
                    'similarity_percentage': (similarity * 100) if similarity is not None else 0.0,  # Add percentage field
                    'best_match_name': best_match_name,
                    'reason': reason or 'Low similarity',
                    'processed_at': processed_at
                })
            
            conn.close()
            return images
        except Exception as e:
            self.logger.error(f"Error getting low similarity images: {e}")
            return []

    def is_duplicate_image(self, image_url: str, embedding: np.ndarray) -> bool:
        """Check if this image has already been processed"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            try:
                # Check if image URL already exists
                cursor.execute('SELECT COUNT(*) FROM person_visits WHERE image_url = ?', (image_url,))
                if cursor.fetchone()[0] > 0:
                    return True
                
                # Check if image URL exists in low similarity table
                cursor.execute('SELECT COUNT(*) FROM low_similarity_images WHERE image_url = ?', (image_url,))
                if cursor.fetchone()[0] > 0:
                    return True
                
                # Check if embedding is too similar to existing ones using Qdrant
                if self.vector_db.get_embedding_count() > 0:
                    duplicate_threshold = self.config['face_recognition']['duplicate_similarity_threshold']
                    similar_faces = self.vector_db.search_similar(
                        query_embedding=embedding,
                        k=1,  # Only need the most similar
                        threshold=duplicate_threshold
                    )
                    
                    if similar_faces and len(similar_faces) > 0:
                        return True
                
                return False
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"Error checking for duplicate image: {e}")
            return False  # If there's an error, assume it's not a duplicate

    def clear_all_data(self, clear_clustering_files=False):
        """Clear all data from database and optionally clustering results"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Clear all tables
            cursor.execute('DELETE FROM person_visits')
            cursor.execute('DELETE FROM persons')
            cursor.execute('DELETE FROM failed_images')
            cursor.execute('DELETE FROM face_quality')
            
            # Try to clear low_similarity_images table if it exists
            try:
                cursor.execute('DELETE FROM low_similarity_images')
            except sqlite3.OperationalError:
                pass  # Table doesn't exist, skip it
            
            conn.commit()
            self.logger.info("All data cleared from database")
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            conn.rollback()
        finally:
            conn.close()
        
        # Clear clustering results JSON files only if requested
        if clear_clustering_files:
            try:
                import glob
                import os
                
                clustering_files = glob.glob("clustering_results/clustering_results_*.json")
                files_deleted = 0
                
                for file_path in clustering_files:
                    try:
                        os.remove(file_path)
                        files_deleted += 1
                        self.logger.info(f"Deleted clustering results file: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not delete file {file_path}: {e}")
                
                self.logger.info(f"Cleared {files_deleted} clustering results files")
                
            except Exception as e:
                self.logger.error(f"Error clearing clustering results files: {e}")
        
        # Always clear low similarity images when clearing database (if table exists)
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='low_similarity_images'")
            if cursor.fetchone():
                cursor.execute('DELETE FROM low_similarity_images')
                conn.commit()
                self.logger.info("Cleared low similarity images from database")
            else:
                self.logger.info("low_similarity_images table does not exist, skipping")
            conn.close()
        except Exception as e:
            self.logger.error(f"Error clearing low similarity images: {e}")
        
        # Clear Qdrant vector database
        try:
            self.vector_db.clear_all()
            self.logger.info("Qdrant vector database cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing Qdrant database: {e}")
        
        # Reset in-memory data
        self.face_quality_cache = {}
        self.image_cache = {}
    
    # Webhook functionality removed - now using JSON storage

    def merge_duplicate_persons(self, person_id1: int, person_id2: int):
        """
        Merge two persons that are actually the same person
        
        Args:
            person_id1: ID of the person to keep
            person_id2: ID of the person to merge and delete
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Update all visits from person_id2 to person_id1
            cursor.execute('''
                UPDATE person_visits 
                SET person_id = ? 
                WHERE person_id = ?
            ''', (person_id1, person_id2))
            
            # Update match count for person_id1
            cursor.execute('''
                UPDATE persons 
                SET match_count = match_count + (
                    SELECT match_count FROM persons WHERE id = ?
                )
                WHERE id = ?
            ''', (person_id2, person_id1))
            
            # Delete the duplicate person from SQLite
            cursor.execute('DELETE FROM persons WHERE id = ?', (person_id2,))
            
            conn.commit()
            self.logger.info(f"Merged person {person_id2} into person {person_id1} in SQLite")
            
            # Delete the duplicate person from Qdrant
            try:
                self.vector_db.delete_embedding(person_id2)
                self.logger.info(f"Deleted person {person_id2} from Qdrant")
            except Exception as e:
                self.logger.error(f"Error deleting person {person_id2} from Qdrant: {e}")
            
        except Exception as e:
            self.logger.error(f"Error merging persons {person_id1} and {person_id2}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def find_and_merge_duplicates(self, similarity_threshold: float = None):
        """
        Find and merge duplicate persons based on high similarity using Qdrant
        
        Args:
            similarity_threshold: Threshold for considering persons as duplicates
        """
        try:
            # Use config default if threshold not provided
            if similarity_threshold is None:
                similarity_threshold = self.config['face_recognition']['merge_duplicate_threshold']
            
            # Get all person IDs from SQLite
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM persons ORDER BY id')
            persons = cursor.fetchall()
            conn.close()
            
            if len(persons) < 2:
                self.logger.info("Not enough persons to find duplicates")
                return
            
            self.logger.info(f"Searching for duplicate persons among {len(persons)} persons...")
            
            # Check each person against others using Qdrant
            processed_pairs = set()
            duplicates_found = 0
            
            for i, (person_id1, name1) in enumerate(persons):
                # Get embedding for person1
                embedding1 = self.vector_db.get_embedding(person_id1)
                if embedding1 is None:
                    continue
                
                # Search for similar persons using Qdrant
                similar_persons = self.vector_db.search_similar(
                    query_embedding=embedding1,
                    k=len(persons),  # Get all persons
                    threshold=similarity_threshold
                )
                
                # Check each similar person
                for similar in similar_persons:
                    person_id2 = similar['person_id']
                    similarity = similar['similarity']
                    
                    # Skip self and already processed pairs
                    if (person_id1 >= person_id2 or 
                        (person_id1, person_id2) in processed_pairs or 
                        (person_id2, person_id1) in processed_pairs):
                        continue
                    
                    # Mark this pair as processed
                    processed_pairs.add((person_id1, person_id2))
                    
                    # Get name for person2
                    name2 = similar['name']
                    
                    self.logger.info(f"Found duplicate persons: {name1} (ID: {person_id1}) and {name2} (ID: {person_id2}) (similarity: {similarity:.3f})")
                    
                    # Merge person2 into person1
                    self.merge_duplicate_persons(person_id1, person_id2)
                    duplicates_found += 1
                    
                    # Update the persons list to reflect the merge
                    persons = [(pid, name) for pid, name in persons if pid != person_id2]
            
            self.logger.info(f"Duplicate detection completed. Found and merged {duplicates_found} duplicate pairs.")
            
        except Exception as e:
            self.logger.error(f"Error finding duplicates: {e}")


def main():
    # Load configuration from JSON file
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("Configuration loaded from config.json")
    except FileNotFoundError:
        print("Configuration file config.json not found, using defaults")
        config = {
            'system': {'database_path': 'face_database.db', 'model_name': 'buffalo_l', 'gpu_id': 0},
            'face_detection': {'confidence_threshold': 0.5, 'quality_threshold': 0.3},
            'face_recognition': {'similarity_threshold': 0.4},
            'processing': {'max_visits_default': 500}
        }
    
    # Configuration - Real-time data only
    OUTPUT_FOLDER = "processed_visits"  # Folder to organize recognized persons
    JSON_FILE_PATH = "visit-cluster.json"  # JSON file with visit data
    DATABASE_PATH = config['system']['database_path']
    MODEL_NAME = config['system']['model_name']
    GPU_ID = config['system']['gpu_id']
    SIMILARITY_THRESH = config['face_recognition']['similarity_threshold']
    CONFIDENCE_THRESH = config['face_detection']['confidence_threshold']
    QUALITY_THRESH = config['face_detection']['quality_threshold']
    MAX_VISITS = config['processing']['max_visits_default']
    
    print("🌟 Smart Face Recognition System - Real-time Data Processing")
    print("=" * 60)
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"GPU ID: {GPU_ID}")
    print(f"Similarity threshold: {SIMILARITY_THRESH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESH}")
    print(f"Quality threshold: {QUALITY_THRESH}")
    print(f"Max visits to process: {MAX_VISITS if MAX_VISITS else 'All'}")
    print("=" * 60)
    
    # Initialize system
    face_recognition = SmartFaceRecognition(
        database_path=DATABASE_PATH,
        model_name=MODEL_NAME,
        gpu_id=GPU_ID,
        confidence_thresh=CONFIDENCE_THRESH,
        similarity_thresh=SIMILARITY_THRESH,
        quality_thresh=QUALITY_THRESH
    )
    
    # Process visit data from JSON
    results = face_recognition.process_visit_data(JSON_FILE_PATH, OUTPUT_FOLDER, MAX_VISITS, save_images=True)
    
    # Print database stats
    stats = face_recognition.get_database_stats()
    print(f"\n📊 Database Statistics:")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Average quality: {stats['average_quality']:.3f}")
    print(f"Recent activity: {stats['recent_activity']}")
    print(f"Embeddings loaded: {stats['embeddings_loaded']}")
    
    # Print processing results
    print(f"\n📈 Processing Results:")
    print(f"Visits processed: {results['processed']}")
    print(f"Persons recognized: {results['recognized']}")
    print(f"New persons added: {results['new_persons']}")
    print(f"No faces found: {results['no_faces']}")
    print(f"Low quality faces: {results['low_quality']}")
    print(f"Duplicate faces: {results['duplicate_faces']}")
    print(f"Download failures: {results['download_failed']}")


# FastAPI Web Interface
app = FastAPI(title="🌟 Smart Face Recognition System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global face recognition instance
face_recognition = None

@app.on_event("startup")
async def startup_event():
    """Initialize the face recognition system on startup"""
    global face_recognition
    try:
        # Load configuration for web server
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            print("Configuration loaded from config.json for web server")
        except FileNotFoundError:
            print("Configuration file config.json not found, using defaults for web server")
            config = {
                'system': {'database_path': 'face_database.db', 'model_name': 'buffalo_l', 'gpu_id': 0},
                'face_detection': {'confidence_threshold': 0.4, 'quality_threshold': 0.3},
                'face_recognition': {'similarity_threshold': 0.7}
            }
        
        face_recognition = SmartFaceRecognition(
            database_path=config['system']['database_path'],
            model_name=config['system']['model_name'],
            gpu_id=config['system']['gpu_id'],
            confidence_thresh=config['face_detection']['confidence_threshold'],
            similarity_thresh=config['face_recognition']['similarity_threshold'],
            quality_thresh=config['face_detection']['quality_threshold']
        )
        print("🌟 Face Recognition System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing face recognition system: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        stats = face_recognition.get_web_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/config")
async def get_api_config():
    """Get API configuration from config file"""
    try:
        config = load_api_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {str(e)}")

@app.get("/api/person-groups")
async def get_person_groups():
    """Get person groups with images"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        groups = face_recognition.get_person_groups_for_web()
        return groups
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting person groups: {str(e)}")

@app.get("/api/clustering-results")
async def get_clustering_results():
    """Get clustering results from database (real-time data)"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Get real-time data from database
        person_groups = face_recognition.get_person_groups_for_web()
        
        # Convert database format to clustering results format
        groups = []
        for group in person_groups:
            # Extract visits and convert to clustering format
            visits = []
            for visit in group.get('images', []):
                # Extract eventId from entryEventIds
                entry_events = json.loads(visit.get('entryEventIds', '[]')) if visit.get('entryEventIds') else []
                event_id = entry_events[0].get('eventId', '') if entry_events else ''
                
                visits.append({
                    'customerId': visit.get('customer_id', ''),
                    'eventId': event_id,
                    'visitId': visit.get('visit_id', 0),
                    'branchId': visit.get('branchId', ''),
                    'camera': visit.get('camera', ''),
                    'entryTime': visit.get('entry_time', ''),
                    'image': visit.get('image_url', ''),
                    'similarity': visit.get('similarity', 0.0),
                    'personId': group.get('person_id', ''),
                    'personName': group.get('name', '')
                })
            
            # Create group in clustering format
            groups.append({
                'group_id': f"group_{group.get('person_id', 'unknown')}",
                'person_id': group.get('person_id', ''),
                'person_name': group.get('name', f"Person_{group.get('person_id', 'unknown')}"),
                'group_score': group.get('avg_quality', 0.0),
                'visit_count': group.get('visit_count', len(visits)),
                'branchId': group.get('branchId', ''),
                'camera': group.get('camera', ''),
                'visits': visits
            })
        
        # Validate that we have real customer data
        has_real_data = face_recognition.validate_real_customer_data(person_groups)
        
        return {
            "groups": groups,
            "total_groups": len(groups),
            "data_source": "database",
            "generated_at": datetime.now().isoformat(),
            "has_real_data": has_real_data,
            "message": f"Real-time data: {len(groups)} groups from database" + (" (Real customer data)" if has_real_data else " (Test data - no real customers found)")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading clustering results: {str(e)}")

@app.get("/api/low-similarity-images")
async def get_low_similarity_images():
    """Get low similarity images from database (real-time data)"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Get real-time data from database
        person_groups = face_recognition.get_person_groups_for_web()
        
        # Extract low similarity images from all groups
        low_similarity_images = []
        for group in person_groups:
            visits = group.get('images', [])
            person_name = group.get('name', '')
            
            for visit in visits:
                similarity = visit.get('similarity', 0)
                # Consider images with similarity <= 0.3 as low similarity
                if similarity <= 0.3:
                    # Determine reason based on similarity
                    if similarity == 0.0 or similarity is None:
                        reason = "New Person (No Match Found)"
                        best_match_name = None
                    elif similarity < 0.15:
                        reason = "Very Low Similarity"
                        best_match_name = person_name if person_name else "Unknown"
                    elif similarity < 0.25:
                        reason = "Low Similarity"
                        best_match_name = person_name if person_name else "Unknown"
                    else:
                        reason = "Borderline Similarity"
                        best_match_name = person_name if person_name else "Unknown"
                    
                    low_similarity_images.append({
                        "visit_id": visit.get('visit_id', ''),
                        "customer_id": visit.get('customer_id', ''),
                        "image_url": visit.get('image_url', ''),
                        "image_path": visit.get('image_url', ''),  # Add image_path field for frontend compatibility
                        "entry_time": visit.get('entry_time', ''),
                        "similarity": similarity if similarity is not None else 0.0,
                        "person_id": group.get('person_id', ''),
                        "person_name": person_name,
                        "best_match_name": best_match_name,
                        "reason": reason,
                        "group_id": f"group_{group.get('person_id', 'unknown')}",
                        "branchId": visit.get('branchId', ''),
                        "camera": visit.get('camera', ''),
                        "eventId": visit.get('eventId', '')
                    })
        
        return low_similarity_images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting low similarity images: {str(e)}")

@app.post("/api/merge-duplicates")
async def merge_duplicates():
    """Find and merge duplicate persons"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.find_and_merge_duplicates()
        return {"message": "Duplicate detection and merging completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error merging duplicates: {str(e)}")

@app.post("/api/clear-database")
async def clear_database():
    """Clear all data from database"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.clear_all_data(clear_clustering_files=False)
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.post("/api/clear-clustering-files")
async def clear_clustering_files():
    """Clear clustering results JSON files"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        import glob
        import os
        
        clustering_files = glob.glob("clustering_results/clustering_results_*.json")
        files_deleted = 0
        
        for file_path in clustering_files:
            try:
                os.remove(file_path)
                files_deleted += 1
            except Exception as e:
                pass  # Continue with other files
        
        return {"message": f"Cleared {files_deleted} clustering results files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing clustering files: {str(e)}")

@app.post("/api/clear-all-data")
async def clear_all_data():
    """Clear all data from database and clustering files"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.clear_all_data(clear_clustering_files=True)
        return {"message": "All data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing all data: {str(e)}")

@app.post("/api/update-json-files")
async def update_json_files():
    """Update JSON files with real-time database data"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        import time
        from datetime import timezone
        
        # Get real-time data from database
        person_groups = face_recognition.get_person_groups_for_web()
        
        # Convert to clustering results format
        groups = []
        for group in person_groups:
            visits = []
            for visit in group.get('images', []):
                visits.append({
                    'customerId': visit.get('customer_id', ''),
                    'eventId': visit.get('eventId', ''),
                    'visitId': visit.get('visit_id', 0),
                    'branchId': visit.get('branchId', ''),
                    'camera': visit.get('camera', ''),
                    'entryTime': visit.get('entry_time', ''),
                    'image': visit.get('image_url', ''),
                    'similarity': visit.get('similarity', 0.0),
                    'personId': group.get('person_id', ''),
                    'personName': group.get('name', '')
                })
            
            groups.append({
                'group_id': f"group_{group.get('person_id', 'unknown')}",
                'person_id': group.get('person_id', ''),
                'person_name': group.get('name', f"Person_{group.get('person_id', 'unknown')}"),
                'group_score': group.get('avg_quality', 0.0),
                'visit_count': group.get('visit_count', len(visits)),
                'branchId': group.get('branchId', ''),
                'camera': group.get('camera', ''),
                'visits': visits
            })
        
        # Create clustering results JSON
        clustering_data = {
            "job_id": f"realtime_{int(time.time())}",
            "status": "finished",
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "total_processed": sum(len(group.get('visits', [])) for group in groups),
            "total_groups": len(groups),
            "results": {
                "processed": sum(len(group.get('visits', [])) for group in groups),
                "recognized": len(groups),
                "new_persons": len(groups),
                "no_faces": 0,
                "low_quality": 0,
                "download_failed": 0,
                "duplicate_faces": 0,
                "low_similarity": len([v for group in groups for v in group.get('visits', []) if v.get('similarity', 0) <= 0.3])
            },
            "message": f"Real-time data: Created {len(groups)} groups from database",
            "groups": groups
        }
        
        # Save to clustering results directory
        os.makedirs("clustering_results", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"clustering_results_{timestamp}_realtime.json"
        filepath = os.path.join("clustering_results", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clustering_data, f, indent=2, ensure_ascii=False)
        
        # Also create customer groups format
        customer_groups_data = {}
        group_count = 1
        for group in groups:
            visits = group.get('visits', [])
            if visits:
                customer_groups_data[f"group{group_count}"] = visits
                group_count += 1
        
        customer_groups_filename = f"customer_groups_by_face_recognition_{timestamp}.json"
        customer_groups_filepath = os.path.join(".", customer_groups_filename)
        
        with open(customer_groups_filepath, 'w', encoding='utf-8') as f:
            json.dump(customer_groups_data, f, indent=2, ensure_ascii=False)
        
        return {
            "message": "JSON files updated with real-time data",
            "clustering_file": filepath,
            "customer_groups_file": customer_groups_filepath,
            "total_groups": len(groups),
            "total_visits": sum(len(group.get('visits', [])) for group in groups)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating JSON files: {str(e)}")

# Webhook API endpoint removed - now using JSON storage

@app.get("/api/person/{person_id}")
async def get_person_details(person_id: int):
    """Get detailed information about a specific person"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        conn = sqlite3.connect(face_recognition.database_path)
        cursor = conn.cursor()
        
        # Get person details
        cursor.execute('''
            SELECT id, name, image_path, face_quality, match_count, last_seen, created_at
            FROM persons WHERE id = ?
        ''', (person_id,))
        
        person = cursor.fetchone()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Get all visits for this person
        cursor.execute('''
            SELECT visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, processed_at
            FROM person_visits
            WHERE person_id = ?
            ORDER BY entry_time DESC
        ''', (person_id,))
        
        visits = cursor.fetchall()
        conn.close()
        
        return {
            'person_id': person[0],
            'name': person[1],
            'image_path': person[2],
            'face_quality': person[3],
            'match_count': person[4],
            'last_seen': person[5],
            'created_at': person[6],
            'visits': [
                {
                    'visit_id': visit[0],
                    'customer_id': visit[1],
                    'entry_time': visit[2],
                    'image_url': visit[3],
                    'saved_image_path': visit[4],
                    'similarity': visit[5],
                    'processed_at': visit[6]
                }
                for visit in visits
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting person details: {str(e)}")

@app.get("/api/failed-images")
async def get_failed_images():
    """Get failed images (no faces detected) for unused tab"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        conn = sqlite3.connect(face_recognition.database_path)
        cursor = conn.cursor()
        
        # Get failed images
        cursor.execute('''
            SELECT visit_id, customer_id, entry_time, image_url, saved_image_path, reason, processed_at
            FROM failed_images
            ORDER BY processed_at DESC
        ''')
        
        failed_images = cursor.fetchall()
        conn.close()
        
        return {
            'failed_images': [
                {
                    'visit_id': img[0],
                    'customer_id': img[1],
                    'entry_time': img[2],
                    'image_url': img[3],
                    'saved_image_path': img[4],
                    'reason': img[5],
                    'processed_at': img[6]
                }
                for img in failed_images
            ],
            'total_count': len(failed_images)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting failed images: {str(e)}")

@app.get("/api/unusable-images")
async def get_unusable_images():
    """Get all unusable images (failed + low similarity) combined"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        conn = sqlite3.connect(face_recognition.database_path)
        cursor = conn.cursor()
        
        # Get failed images (no faces detected)
        cursor.execute('''
            SELECT visit_id, customer_id, entry_time, image_url, saved_image_path, reason, processed_at
            FROM failed_images
            ORDER BY processed_at DESC
        ''')
        failed_images = cursor.fetchall()
        
        # Get low similarity images
        cursor.execute('''
            SELECT visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, processed_at
            FROM low_similarity_images
            WHERE reason IS NULL
            ORDER BY processed_at DESC
        ''')
        low_sim_images = cursor.fetchall()
        
        conn.close()
        
        result = []
        
        # Process failed images
        for img in failed_images:
            visit_id, customer_id, entry_time, image_url, saved_image_path, reason, processed_at = img
            result.append({
                'visit_id': visit_id,
                'customer_id': customer_id,
                'entry_time': entry_time,
                'image_url': image_url,
                'saved_image_path': saved_image_path,
                'reason': reason,
                'similarity': None,
                'processed_at': processed_at,
                'type': 'failed'
            })
        
        # Process low similarity images
        for img in low_sim_images:
            visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, processed_at = img
            result.append({
                'visit_id': visit_id,
                'customer_id': customer_id,
                'entry_time': entry_time,
                'image_url': image_url,
                'saved_image_path': saved_image_path,
                'reason': None,
                'similarity': similarity,
                'processed_at': processed_at,
                'type': 'low_similarity'
            })
        
        # Sort by processed_at descending
        result.sort(key=lambda x: x['processed_at'], reverse=True)
        
        return {
            "unusable_images": result,
            "failed_count": len([img for img in result if img['type'] == 'failed']),
            "low_similarity_count": len([img for img in result if img['type'] == 'low_similarity']),
            "total_count": len(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting unusable images: {str(e)}")

@app.post("/api/process-visits")
async def process_visits(request_data: dict = None):
    """Process visits from JSON data (real-time data only)"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Extract parameters from request
        json_data = request_data.get('json_data') if request_data else None
        if not json_data:
            raise HTTPException(status_code=400, detail="JSON data is required. Use /api/process-visits-from-api for real-time data.")
        
        # Auto-detect max visits from JSON data
        max_visits = request_data.get('max_visits', None) if request_data else None
        if max_visits is None and json_data:
            max_visits = len(json_data.get('visits', []))
        elif max_visits is None:
            max_visits = face_recognition.config['processing']['max_visits_fallback']
        save_images = request_data.get('save_images', True) if request_data else True
        clear_existing = request_data.get('clear_existing', False) if request_data else False
        
        # Process JSON data from web interface
        results = face_recognition.process_visit_data_from_json(
            json_data=json_data,
            output_folder="processed_visits",
            max_visits=max_visits,
            save_images=save_images,
            clear_existing=clear_existing
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing visits: {str(e)}")


@app.post("/api/process-real-data")
async def process_real_data(request_data: dict = None):
    """Process real data from existing visit-cluster.json file"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        import json
        
        # Load real data from visit-cluster.json
        with open("visit-cluster.json", 'r', encoding='utf-8') as f:
            real_data = json.load(f)
        
        # Extract parameters from request
        max_visits = request_data.get('max_visits', 50) if request_data else 50
        save_images = request_data.get('save_images', True) if request_data else True
        clear_existing = request_data.get('clear_existing', False) if request_data else False
        
        # Process real data
        results = face_recognition.process_visit_data_from_json(
            json_data=real_data,
            output_folder="processed_visits",
            max_visits=max_visits,
            save_images=save_images,
            clear_existing=clear_existing
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing real data: {str(e)}")

@app.post("/api/process-face-comparisons-from-api")
async def process_face_comparisons_from_api(request_data: dict):
    """Process face comparisons from external API - New structure with url1 and url2"""
    try:
        # Initialize the face comparison system
        face_comparison = FaceComparisonFromAPI()
        
        # Extract API parameters from request
        api_url = request_data.get('api_url')
        if not api_url:
            raise HTTPException(status_code=400, detail="API URL is required")
        
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        page = request_data.get('page', 0)
        limit = request_data.get('limit', 100)
        start_time = request_data.get('start_time')
        end_time = request_data.get('end_time')
        all_branch = request_data.get('all_branch', True)
        max_comparisons = request_data.get('max_comparisons', limit)
        api_key = request_data.get('api_key')
        auth_token = request_data.get('auth_token')
        
        # Fetch face comparison data from API with pagination support
        comparison_records = face_comparison.fetch_face_comparison_data_from_api(
            api_url=api_url,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            all_branch=all_branch,
            api_key=api_key,
            auth_token=auth_token,
            max_comparisons=max_comparisons
        )
        
        if not comparison_records:
            return {
                "message": "No face comparison records found from API",
                "total_comparisons": 0,
                "processed": 0,
                "same_person": 0,
                "different_person": 0,
                "errors": 0,
                "results": []
            }
        
        # Process the face comparisons
        results = face_comparison.process_face_comparisons(
            comparison_records=comparison_records,
            max_comparisons=max_comparisons
        )
        
        # Add API-specific information to results
        results['api_info'] = {
            'api_url': api_url,
            'fetched_records': len(comparison_records),
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'page': page,
                'limit': limit,
                'start_time': start_time,
                'end_time': end_time,
                'all_branch': all_branch
            }
        }
        
        # Save results to JSON file
        try:
            import json
            from datetime import datetime
            
            print(f"🔧 Starting JSON file creation...")
            print(f"🔧 Results keys: {list(results.keys())}")
            print(f"🔧 Results length: {len(results.get('results', []))}")
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_comparison_results_{timestamp}.json"
            print(f"🔧 Creating JSON file: {filename}")
            
            # Prepare data for JSON file (simplified format as requested)
            json_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_comparisons': results.get('total_comparisons', 0),
                    'same_person': results.get('same_person', 0),
                    'different_person': results.get('different_person', 0),
                    'errors': results.get('errors', 0),
                    'accuracy_vs_api': results.get('accuracy_vs_api', 0)
                },
                'comparisons': []
            }
            
            # Add each comparison result in the requested format - only include specified fields
            for comparison in results.get('results', []):
                # Extract fileName, event, and camera from the raw_data if available
                raw_data = comparison.get('raw_data', {})
                entry_event_ids = raw_data.get('entryEventIds', [])
                
                # Get data from first entryEventId if available
                fileName = ''
                event = ''
                camera = ''
                eventId = ''
                
                if entry_event_ids and len(entry_event_ids) > 0:
                    event_data = entry_event_ids[0]
                    fileName = event_data.get('fileName', '')
                    event = event_data.get('event', '')
                    camera = event_data.get('camera', '')
                    eventId = event_data.get('eventId', '')
                else:
                    # Fallback: try to get eventId from the comparison's event_id if it's a string
                    event_id = comparison.get('event_id', '')
                    if isinstance(event_id, str):
                        eventId = event_id
                
                comparison_data = {
                    'fileName': fileName,
                    'event': event,
                    'camera': camera,
                    'eventId': eventId,
                    'approve': comparison.get('api_approve', False),
                    'match_status': comparison.get('match_status', 'UNKNOWN'),
                    'branch_id': comparison.get('branch_id', '')
                }
                json_data['comparisons'].append(comparison_data)
            
            # Write to JSON file
            print(f"🔧 Writing to JSON file: {filename}")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"🔧 JSON file written successfully")
            
            # Add file info to results
            results['json_file'] = {
                'filename': filename,
                'path': os.path.abspath(filename),
                'size': os.path.getsize(filename)
            }
            
            print(f"✅ Face comparison results saved to: {filename}")
            print(f"✅ File size: {os.path.getsize(filename)} bytes")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save results to JSON file: {e}")
            results['json_file'] = {'error': str(e)}
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing face comparisons from API: {str(e)}")

@app.post("/api/run-clustering")
async def run_clustering():
    """Run face recognition clustering on processed visits"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Get person groups from database
        person_groups = face_recognition.get_person_groups_for_web()
        
        if not person_groups:
            return {
                "message": "No person groups found to cluster. Process some visits first.",
                "groups_created": 0
            }
        
        # Convert to clustering format and save
        from json_storage import save_clustering_results
        
        # Convert database format to clustering format
        groups = []
        total_processed = 0
        
        for group in person_groups:
            visits = []
            for visit in group.get('images', []):
                visits.append({
                    'id': visit.get('visit_id'),
                    'customerId': visit.get('customer_id'),
                    'image': visit.get('image_url'),
                    'entryTime': visit.get('entry_time'),
                    'branchId': visit.get('branchId', ''),
                    'camera': visit.get('camera', ''),
                    'entryEventIds': json.loads(visit.get('entryEventIds', '[]')) if visit.get('entryEventIds') else [],
                    'similarity': visit.get('similarity', 0.0)
                })
                total_processed += 1
            
            if visits:
                groups.append({
                    'person_id': group.get('person_id'),
                    'person_name': group.get('name'),
                    'visits': visits
                })
        
        # Save clustering results
        results = {
            'processed': total_processed,
            'recognized': len(groups),
            'new_persons': len(groups),
            'no_faces': 0,
            'low_quality': 0,
            'download_failed': 0,
            'duplicate_faces': 0,
            'low_similarity': 0
        }
        
        success = save_clustering_results(groups, total_processed, results)
        
        if success:
            return {
                "message": f"Clustering completed successfully. Created {len(groups)} groups from {total_processed} visits.",
                "groups_created": len(groups),
                "total_processed": total_processed
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save clustering results")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running clustering: {str(e)}")

@app.post("/api/process-visits-from-api")
async def process_visits_from_api(request_data: dict):
    """Process visits from external API"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Extract API parameters from request
        api_url = request_data.get('api_url')
        if not api_url:
            raise HTTPException(status_code=400, detail="API URL is required")
        
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        page = request_data.get('page', 0)
        limit = request_data.get('limit', 100)
        start_time = request_data.get('start_time')
        end_time = request_data.get('end_time')
        all_branch = request_data.get('all_branch', True)
        selected_branch = request_data.get('selected_branch', 'EM-DEPT')
        max_visits = request_data.get('max_visits', limit)
        # Use max_visits for total visits to fetch, limit for page size
        if 'max_visits' in request_data and request_data['max_visits'] > 0:
            max_visits = request_data['max_visits']
        
        # Log the parameters for debugging
        face_recognition.logger.info(f"API processing parameters: limit={limit}, max_visits={max_visits}, page={page}, selected_branch={selected_branch}")
        save_images = request_data.get('save_images', True)
        clear_existing = request_data.get('clear_existing', False)  # Default to False to preserve data
        api_key = request_data.get('api_key')
        auth_token = request_data.get('auth_token')
        
        # Fetch data from API with pagination support
        visits = face_recognition.fetch_visit_data_from_api(
            api_url=api_url,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            all_branch=all_branch,
            selected_branch=selected_branch,
            api_key=api_key,
            auth_token=auth_token,
            max_visits=max_visits
        )
        
        if not visits:
            return {
                "message": "No visits found from API",
                "processed": 0,
                "recognized": 0,
                "new_persons": 0,
                "no_faces": 0,
                "low_quality": 0,
                "download_failed": 0,
                "duplicate_faces": 0,
                "low_similarity": 0
            }
        
        # Limit visits if max_visits is specified
        if max_visits and len(visits) > max_visits:
            visits = visits[:max_visits]
        
        # Create JSON data structure for processing
        json_data = {
            "visits": visits,
            "total_visits": len(visits),
            "api_url": api_url,
            "fetched_at": datetime.now().isoformat()
        }
        
        # Process the visits using existing logic
        results = face_recognition.process_visit_data_from_json(
            json_data=json_data,
            output_folder="processed_visits",
            max_visits=max_visits,
            save_images=save_images,
            clear_existing=clear_existing
        )
        
        # Store full JSON response in reference file (additional backup)
        face_recognition._store_full_json_response(visits, api_url, start_date, end_date, page, limit)
        
        # Add API-specific information to results
        results['api_info'] = {
            'api_url': api_url,
            'fetched_visits': len(visits),
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'page': page,
                'limit': limit,
                'start_time': start_time,
                'end_time': end_time,
                'all_branch': all_branch
            }
        }
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing visits from API: {str(e)}")

@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """Serve local images with proper caching and error handling"""
    try:
        # Decode the image path
        decoded_path = image_path.replace('%3A', ':').replace('%2F', '/').replace('%5C', '\\')
        
        # Check if file exists
        if not os.path.exists(decoded_path):
            # Try to serve the no-image placeholder
            no_image_path = "static/no-image.png"
            if os.path.exists(no_image_path):
                return FileResponse(no_image_path)
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Return the image file with proper headers for caching
        response = FileResponse(decoded_path)
        cache_max_age = face_recognition.config['web_interface']['cache_control_max_age']
        response.headers["Cache-Control"] = f"public, max-age={cache_max_age}"
        response.headers["Content-Type"] = "image/jpeg"
        return response
    except Exception as e:
        # Try to serve the no-image placeholder on error
        no_image_path = "static/no-image.png"
        if os.path.exists(no_image_path):
            return FileResponse(no_image_path)
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")


@app.get("/api/image-proxy/{image_url:path}")
async def proxy_external_image(image_url: str):
    """Proxy external images to avoid CORS issues with retry logic"""
    try:
        # Decode the image URL
        decoded_url = image_url.replace('%3A', ':').replace('%2F', '/').replace('%5C', '\\')
        
        # Validate that it's a proper URL
        if not decoded_url.startswith('http'):
            raise HTTPException(status_code=400, detail="Invalid URL")
        
        # Set headers to request image content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        }
        
        # Try to download the image with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(decoded_url, timeout=10, stream=True, headers=headers)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'application/json' in content_type:
                    # Server returned JSON error instead of image
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                        continue
                    raise HTTPException(status_code=404, detail="Image not available")
                
                # Return the image with proper headers
                from fastapi.responses import StreamingResponse
                import io
                
                def generate():
                    for chunk in response.iter_content(chunk_size=8192):
                        yield chunk
                
                return StreamingResponse(
                    generate(),
                    media_type=response.headers.get('content-type', 'image/jpeg'),
                    headers={
                        "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                        "Access-Control-Allow-Origin": "*"
                    }
                )
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait longer for timeout retries
                    continue
                raise HTTPException(status_code=408, detail="Image request timeout")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                raise HTTPException(status_code=404, detail=f"Image not found: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        # Return no-image placeholder on error
        no_image_path = "static/no-image.png"
        if os.path.exists(no_image_path):
            return FileResponse(no_image_path)
        raise HTTPException(status_code=500, detail=f"Error proxying image: {str(e)}")


@app.get("/api/image-base64/{image_path:path}")
async def serve_image_base64(image_path: str):
    """Serve images as base64 for better web performance"""
    try:
        # Decode the image path
        decoded_path = image_path.replace('%3A', ':').replace('%2F', '/').replace('%5C', '\\')
        
        # Check if file exists
        if not os.path.exists(decoded_path):
            return {"error": "Image not found", "base64": None}
        
        # Process image for web
        if face_recognition:
            base64_image = face_recognition.process_image_for_web(decoded_path)
            if base64_image:
                return {"base64": base64_image}
        
        # Fallback to regular file serving
        return {"error": "Could not process image", "base64": None}
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}", "base64": None}

@app.post("/api/clear-cache")
async def clear_image_cache():
    """Clear the image cache"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        import shutil
        cache_dir = face_recognition.image_cache_dir
        
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            face_recognition.logger.info("Image cache cleared successfully")
            return {"message": "Cache cleared successfully"}
        else:
            return {"message": "Cache directory does not exist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/api/clear-database")
async def clear_database():
    """Clear the face recognition database"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.clear_database()
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.post("/api/analyze-missing-data")
async def analyze_missing_data(request_data: dict):
    """Analyze what data is missing from API responses"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Extract API parameters from request
        api_url = request_data.get('api_url')
        if not api_url:
            raise HTTPException(status_code=400, detail="API URL is required")
        
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        page = request_data.get('page', 0)
        limit = request_data.get('limit', 100)
        start_time = request_data.get('start_time')
        end_time = request_data.get('end_time')
        all_branch = request_data.get('all_branch', True)
        selected_branch = request_data.get('selected_branch', 'EM-DEPT')
        max_visits = request_data.get('max_visits', 1000)
        api_key = request_data.get('api_key')
        auth_token = request_data.get('auth_token')
        
        # Fetch data with detailed analysis
        analysis_result = face_recognition.analyze_api_data_completeness(
            api_url=api_url,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            all_branch=all_branch,
            selected_branch=selected_branch,
            api_key=api_key,
            auth_token=auth_token,
            max_visits=max_visits
        )
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing missing data: {str(e)}")

@app.get("/api/analyze-existing-data")
async def analyze_existing_data():
    """Analyze existing reference files to understand data completeness"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        import glob
        from datetime import datetime
        
        # Find all reference files
        reference_files = glob.glob("visit-cluster-reference_*.json")
        reference_files.sort(reverse=True)  # Most recent first
        
        analysis_result = {
            "summary": {
                "total_reference_files": len(reference_files),
                "total_visits_across_files": 0,
                "total_visits_with_images": 0,
                "total_visits_missing_images": 0,
                "missing_data_percentage": 0.0,
                "date_range": {"earliest": None, "latest": None},
                "file_sizes": []
            },
            "file_analysis": [],
            "recommendations": []
        }
        
        all_visits = []
        seen_visit_ids = set()
        seen_image_urls = set()
        
        for file_path in reference_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                visits = data.get('visits', [])
                metadata = data.get('metadata', {})
                
                file_analysis = {
                    "filename": file_path,
                    "generated_at": metadata.get('generated_at', 'Unknown'),
                    "api_url": metadata.get('api_url', 'Unknown'),
                    "total_visits": len(visits),
                    "visits_with_images": 0,
                    "visits_missing_images": 0,
                    "duplicate_visits": 0,
                    "invalid_timestamps": 0,
                    "empty_events": 0
                }
                
                # Analyze each visit in this file
                for visit in visits:
                    visit_id = visit.get('id', visit.get('visitId', visit.get('visit_id')))
                    
                    # Check for duplicates across all files
                    if visit_id in seen_visit_ids:
                        file_analysis["duplicate_visits"] += 1
                    else:
                        seen_visit_ids.add(visit_id)
                    
                    # Check for valid timestamps
                    entry_time = visit.get('entryTime', visit.get('entry_time', visit.get('timestamp')))
                    if not entry_time or entry_time == 'null' or entry_time == '':
                        file_analysis["invalid_timestamps"] += 1
                    
                    # Check for images
                    has_image = False
                    image_url = visit.get('image', '')
                    
                    if image_url and image_url.startswith('http'):
                        if image_url not in seen_image_urls:
                            seen_image_urls.add(image_url)
                            has_image = True
                    
                    # Also check entry events for images
                    if not has_image:
                        entry_events = visit.get('entryEventIds', [])
                        if not entry_events or len(entry_events) == 0:
                            file_analysis["empty_events"] += 1
                        else:
                            for event in entry_events:
                                if isinstance(event, dict) and event.get('fileName'):
                                    image_url = f"https://cdn.analytics.thefusionapps.com/v11/{event['fileName']}"
                                    if image_url not in seen_image_urls:
                                        seen_image_urls.add(image_url)
                                        has_image = True
                                        break
                    
                    if has_image:
                        file_analysis["visits_with_images"] += 1
                    else:
                        file_analysis["visits_missing_images"] += 1
                
                # Calculate file-level percentages
                if file_analysis["total_visits"] > 0:
                    file_analysis["missing_percentage"] = (file_analysis["visits_missing_images"] / file_analysis["total_visits"]) * 100
                else:
                    file_analysis["missing_percentage"] = 0.0
                
                analysis_result["file_analysis"].append(file_analysis)
                analysis_result["summary"]["total_visits_across_files"] += file_analysis["total_visits"]
                analysis_result["summary"]["total_visits_with_images"] += file_analysis["visits_with_images"]
                analysis_result["summary"]["total_visits_missing_images"] += file_analysis["visits_missing_images"]
                
                # Track file size
                file_size = os.path.getsize(file_path)
                analysis_result["summary"]["file_sizes"].append({
                    "filename": file_path,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })
                
                # Track date range
                if file_analysis["generated_at"] != 'Unknown':
                    try:
                        file_date = datetime.fromisoformat(file_analysis["generated_at"].replace('Z', '+00:00'))
                        if analysis_result["summary"]["date_range"]["earliest"] is None or file_date < analysis_result["summary"]["date_range"]["earliest"]:
                            analysis_result["summary"]["date_range"]["earliest"] = file_date.isoformat()
                        if analysis_result["summary"]["date_range"]["latest"] is None or file_date > analysis_result["summary"]["date_range"]["latest"]:
                            analysis_result["summary"]["date_range"]["latest"] = file_date.isoformat()
                    except:
                        pass
                
            except Exception as e:
                analysis_result["file_analysis"].append({
                    "filename": file_path,
                    "error": str(e)
                })
        
        # Calculate overall missing data percentage
        total_visits = analysis_result["summary"]["total_visits_across_files"]
        if total_visits > 0:
            missing_count = analysis_result["summary"]["total_visits_missing_images"]
            analysis_result["summary"]["missing_data_percentage"] = (missing_count / total_visits) * 100
        
        # Generate recommendations
        if analysis_result["summary"]["missing_data_percentage"] > 30:
            analysis_result["recommendations"].append("High missing data percentage - consider re-fetching with different parameters")
        
        if len(analysis_result["file_analysis"]) > 1:
            analysis_result["recommendations"].append("Multiple reference files found - consider consolidating data")
        
        duplicate_count = sum(f.get("duplicate_visits", 0) for f in analysis_result["file_analysis"])
        if duplicate_count > 0:
            analysis_result["recommendations"].append(f"Found {duplicate_count} duplicate visits across files - consider deduplication")
        
        return analysis_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing existing data: {str(e)}")

def run_web_server(host: str = None, port: int = None):
    """Run the FastAPI web server"""
    # Load configuration for web server
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("Configuration loaded from config.json for web server")
    except FileNotFoundError:
        print("Configuration file config.json not found, using defaults for web server")
        config = {
            'web_interface': {'host': '0.0.0.0', 'port': 8000}
        }
    
    # Use config defaults if not provided
    if host is None:
        host = config['web_interface']['host']
    if port is None:
        port = config['web_interface']['port']
    
    print(f"🌟 Starting Smart Face Recognition Web Server...")
    print(f"🌐 Server will be available at: http://{host}:{port}")
    print(f"📱 Web Interface: http://{host}:{port}")
    print(f"📊 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Run web server
        run_web_server()
    else:
        # Run original main function
        main()
# pm2 start main.py --interpreter=python3 --name=boxmot-server
