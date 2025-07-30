import os
import json
import logging
import csv
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import vision
from google.cloud.vision import AnnotateImageResponse
from PIL import Image
from io import BytesIO
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ocr_multimodel_eval.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Initialize Google Vision client
try:
    vision_client = vision.ImageAnnotatorClient()
    logging.info("Google Vision client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Google Vision client: {e}")
    vision_client = None

# Initialize Gemini client
gemini_api_key = os.getenv('GEMINI_API_KEY')
gemini_client = None
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_client = genai
        logging.info("Gemini client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        gemini_client = None
else:
    logging.warning("GEMINI_API_KEY not found in environment variables. Gemini models will not be available.")

# Supported models
SUPPORTED_MODELS = {
    'google_vision': 'Google Cloud Vision API',
    'gemini_2_5_pro': 'gemini-2.5-pro',
    'gemini_2_5_flash': 'gemini-2.5-flash',
    'gemini_2_5_flash_lite': 'gemini-2.5-flash-lite',
    'gemini_2_0_flash': 'gemini-2.0-flash',
    'gemini_2_0_flash_lite': 'gemini-2.0-flash-lite'
}

class OCRMultiModelEvaluator:
    """Multi-model OCR evaluator supporting Google Vision and Gemini models."""
    
    def __init__(self):
        self.vision_client = vision_client
        self.gemini_client = gemini_client
        

    
    def google_vision_ocr(self, image_path: str, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        """Perform OCR using Google Cloud Vision API."""
        try:
            image = vision.Image()
            
            # Load local image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image.content = content
            
            features = [
                {
                    "type_": vision.Feature.Type.DOCUMENT_TEXT_DETECTION,
                    "model": "builtin/weekly",
                }
            ]
            
            image_context = {}
            if lang_hint:
                image_context["language_hints"] = [lang_hint]
            
            response = self.vision_client.annotate_image({
                "image": image,
                "features": features,
                "image_context": image_context
            })
            
            response_json = AnnotateImageResponse.to_json(response)
            response_dict = json.loads(response_json)
            
            # Extract text from response
            if "error" in response_dict:
                text = ""
                error = response_dict["error"]
            elif "fullTextAnnotation" in response_dict:
                text = response_dict["fullTextAnnotation"].get("text", "")
                error = None
            elif "textAnnotations" in response_dict and response_dict["textAnnotations"]:
                text = response_dict["textAnnotations"][0].get("description", "")
                error = None
            else:
                text = ""
                error = None
            
            return {
                "model": "google_vision",
                "text": text.strip(),
                "raw_response": response_dict,
                "error": error
            }
            
        except Exception as e:
            logging.error(f"Google Vision OCR error: {e}")
            return {
                "model": "google_vision",
                "text": "",
                "raw_response": None,
                "error": str(e)
            }
    
    def gemini_ocr(self, image_path: str, model_name: str, lang_hint: Optional[str] = None) -> Dict[str, Any]:
        """Perform OCR using Gemini models."""
        try:
            if not self.gemini_client:
                raise ValueError("Gemini client not initialized. Check GEMINI_API_KEY in environment variables")
            
            # Get the actual model name
            if model_name not in SUPPORTED_MODELS:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}")
            
            actual_model_name = SUPPORTED_MODELS[model_name]
            
            # Load local image file
            image = Image.open(image_path)
            
            # Create prompt for OCR - very specific to get only raw Tibetan text
            prompt = "Extract only the Tibetan text from this image. Return ONLY the raw Tibetan characters exactly as they appear in the image, with no English explanations, no formatting, no line numbers, no headers, and no additional commentary. Just the pure Tibetan text."
            if lang_hint:
                prompt += f" The text is in {lang_hint} language."
            
            # Generate response using Gemini API
            model = self.gemini_client.GenerativeModel(actual_model_name)
            response = model.generate_content([prompt, image])
            
            return {
                "model": model_name,
                "text": response.text.strip() if response.text else "",
                "raw_response": {
                    "candidates": [{
                        "content": {
                            "parts": [{
                                "text": response.text
                            }]
                        }
                    }] if response.text else []
                },
                "error": None
            }
            
        except Exception as e:
            logging.error(f"Gemini OCR error with {model_name}: {e}")
            return {
                "model": model_name,
                "text": "",
                "raw_response": None,
                "error": str(e)
            }
    
    def run_ocr(self, image_path: str, model_name: str, lang_hint: Optional[str] = None, save_to_csv: bool = True) -> Dict[str, Any]:
        """Run OCR using the specified model.
        
        Args:
            image_path: Path to local image file
            model_name: Name of the model to use (see SUPPORTED_MODELS)
            lang_hint: Optional language hint (e.g., 'bo' for Tibetan)
            save_to_csv: Whether to save results to CSV file (default: True)
            
        Returns:
            Dictionary containing OCR results
        """
        logging.info(f"Running OCR with model: {model_name} on image: {image_path}")
        
        if model_name not in SUPPORTED_MODELS:
            result = {
                "model": model_name,
                "text": "",
                "raw_response": None,
                "error": f"Unsupported model: {model_name}. Supported models: {list(SUPPORTED_MODELS.keys())}"
            }
        elif model_name == 'google_vision':
            result = self.google_vision_ocr(image_path, lang_hint)
        else:
            result = self.gemini_ocr(image_path, model_name, lang_hint)
        
        # Save to CSV if requested
        if save_to_csv:
            self.save_result_to_csv(image_path, result, model_name)
        
        return result
    
    def save_result_to_csv(self, image_path: str, result: Dict[str, Any], model_name: str):
        """Save OCR result to CSV file.
        
        Args:
            image_path: Path to image file
            result: OCR result dictionary
            model_name: Name of the model used
        """
        try:
            # Extract filename from path
            filename = os.path.basename(image_path)
            image_name_without_ext = os.path.splitext(filename)[0]
            
            # Create CSV filename with model name and image filename
            csv_filename = f"{model_name}_{image_name_without_ext}.csv"
            
            # Prepare data for CSV with proper Tibetan text handling
            inference_text = result.get('text', '')
            error_text = result.get('error', '')
            
            # Clean and normalize Tibetan text
            if inference_text:
                # Remove any extra whitespace and normalize
                inference_text = ' '.join(inference_text.split())
            
            csv_data = {
                'image_filename': filename,
                'inference_result': inference_text,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'error': error_text
            }
            
            # Check if CSV file exists
            file_exists = os.path.isfile(csv_filename)
            
            # Write to CSV with UTF-8 BOM for better Excel compatibility
            with open(csv_filename, 'a', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['image_filename', 'inference_result', 'model_used', 'timestamp', 'error']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                    logging.info(f"Created new CSV file with UTF-8 BOM: {csv_filename}")
                
                writer.writerow(csv_data)
                logging.info(f"Saved Tibetan text result to CSV: {csv_filename}")
                
        except Exception as e:
            logging.error(f"Error saving to CSV: {e}")
    
    def process_batch_images(self, image_list: list, model_name: str, lang_hint: Optional[str] = None) -> list:
        """Process a batch of images with the specified model.
        
        Args:
            image_list: List of image paths
            model_name: Name of the model to use
            lang_hint: Optional language hint
            
        Returns:
            List of OCR results
        """
        results = []
        total_images = len(image_list)
        
        logging.info(f"Processing {total_images} images with model: {model_name}")
        
        for i, image_path in enumerate(image_list, 1):
            logging.info(f"Processing image {i}/{total_images}: {image_path}")
            
            result = self.run_ocr(image_path, model_name, lang_hint, save_to_csv=True)
            results.append(result)
            
            # Log progress
            if i % 10 == 0 or i == total_images:
                logging.info(f"Completed {i}/{total_images} images")
        
        logging.info(f"Batch processing completed for model: {model_name}")
        return results
    
    def process_multimodel_images(self, model_name: str, lang_hint: Optional[str] = None, images_folder: str = "multimodel-images") -> list:
        """Process all images in the multimodel-images folder with the specified model.
        
        Args:
            model_name: Name of the model to use
            lang_hint: Optional language hint (e.g., 'bo' for Tibetan)
            images_folder: Folder containing images (default: 'multimodel-images')
            
        Returns:
            List of OCR results
        """
        try:
            # Get list of image files from the folder
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            for filename in os.listdir(images_folder):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(images_folder, filename)
                    image_files.append(image_path)
            
            if not image_files:
                logging.warning(f"No image files found in {images_folder}")
                return []
            
            logging.info(f"Found {len(image_files)} images in {images_folder}")
            logging.info(f"Processing with model: {model_name}")
            
            # Process all images
            return self.process_batch_images(image_files, model_name, lang_hint)
            
        except FileNotFoundError:
            logging.error(f"Folder {images_folder} not found")
            return []
        except Exception as e:
            logging.error(f"Error processing multimodel images: {e}")
            return []


def main():
    """Example usage of the OCR Multi-Model Evaluator."""
    evaluator = OCRMultiModelEvaluator()
    
    # Example usage
    print("OCR Multi-Model Evaluator")
    print("Supported models:", list(SUPPORTED_MODELS.keys()))
    
    # Example: Run OCR on a single image with a specific model
    # result = evaluator.run_ocr("multimodel-images/img_1.jpg", "google_vision", lang_hint="bo")
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Example: Process all images in multimodel-images folder
    results = evaluator.process_multimodel_images("gemini_2_5_pro", lang_hint=None, images_folder="multimodel-images")
    print(f"Processed {len(results)} images from multimodel-images folder")
    
    # Example: Process specific images
    # image_list = ["multimodel-images/img_1.jpg", "multimodel-images/img_2.png"]
    # results = evaluator.process_batch_images(image_list, "google_vision", lang_hint="bo")
    # print(f"Processed {len(results)} specific images")


if __name__ == "__main__":
    main()